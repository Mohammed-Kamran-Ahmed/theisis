"""
common.py - helpers shared by train_lmdb.py (dual encoder) and
train_baseline.py (fused ResNet-50), so both run under the SAME protocol.
"""

import ast
import math
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reben_pipeline.metrics import BEN_CLASSES_19


# ---------------------------------------------------------------------------
# Reproducibility / checkpoint helpers
# ---------------------------------------------------------------------------

def set_all_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
        except Exception as exc:  # different GPU count etc.
            print(f"[resume] could not restore CUDA RNG state: {exc}")


def atomic_save(obj, path) -> None:
    """Write to a temp file first so a crash/shutdown never corrupts a checkpoint."""
    path = str(path)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_checkpoint(path):
    # Always load to CPU: avoids moving RNG-state tensors to the GPU.
    return torch.load(str(path), map_location="cpu", weights_only=False)


def to_jsonable(obj):
    """Recursively convert numpy / torch / Path / NaN values into JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def build_data_module(lmdb_dir: str, parquet: str, batch_size: int, num_workers: int = 0):
    """Official ConfigILM BENv2 LMDB data module (12 channels: 2 SAR + 10 optical)."""
    from configilm.extra.DataModules.BENv2_DataModule import BENv2DataModule

    data_dirs = {
        "images_lmdb": str(lmdb_dir),
        "metadata_parquet": str(parquet),
        "metadata_snow_cloud_parquet": None,
    }
    dm = BENv2DataModule(
        data_dirs=data_dirs,
        batch_size=batch_size,
        img_size=(12, 120, 120),
        num_workers_dataloader=num_workers,
    )
    dm.setup("fit")
    return dm


def split_bands(x: torch.Tensor):
    """(B,12,H,W) -> S1 (B,2,H,W) [VV,VH], S2 (B,10,H,W)."""
    return x[:, :2], x[:, 2:]


# ---------------------------------------------------------------------------
# Class frequencies, class-order check, pos_weight
# ---------------------------------------------------------------------------

def class_frequencies_from_parquet(parquet_path: str, split: str = "train") -> np.ndarray:
    """Fraction of `split` patches containing each class, in BEN_CLASSES_19 order."""
    df = pd.read_parquet(parquet_path)
    for col in ("split", "labels"):
        if col not in df.columns:
            raise KeyError(f"Parquet has no '{col}' column (columns: {list(df.columns)})")
    sub = df[df["split"] == split]
    if len(sub) == 0:
        raise ValueError(f"No rows with split == '{split}' in {parquet_path}")

    index = {c: i for i, c in enumerate(BEN_CLASSES_19)}
    counts = np.zeros(len(BEN_CLASSES_19), dtype=np.float64)
    unknown = set()
    for raw in sub["labels"]:
        if isinstance(raw, str):
            try:
                raw = ast.literal_eval(raw.strip())
            except (ValueError, SyntaxError):
                raw = [raw]
        for lbl in {str(l).strip() for l in raw}:
            if lbl in index:
                counts[index[lbl]] += 1.0
            else:
                unknown.add(lbl)
    if unknown:
        print(f"[warn] parquet label names not in BEN_CLASSES_19: {sorted(unknown)[:5]} ...")
    if counts.sum() == 0:
        raise ValueError("No parquet labels matched BEN_CLASSES_19; check the label column format.")
    return counts / len(sub)


def check_label_alignment(freq: np.ndarray, loader, n_batches: int = 10, min_corr: float = 0.9):
    """
    The loader yields multi-hot vectors in ConfigILM's class order, while metrics /
    pos_weight / per-class names use BEN_CLASSES_19. If those orders differ, every
    per-class number is silently wrong. Compare class frequencies seen in real
    loader batches against the parquet frequencies; raise on mismatch.
    """
    counts = np.zeros(len(freq), dtype=np.float64)
    n = 0
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        y = batch[1].float()
        counts += y.sum(dim=0).numpy()
        n += y.shape[0]
    obs = counts / max(n, 1)
    if np.std(obs) < 1e-9 or np.std(freq) < 1e-9:
        raise RuntimeError("Class-order check impossible: constant label frequencies.")
    corr = float(np.corrcoef(obs, freq)[0, 1])
    print(f"[check] class-order alignment: corr(loader freq, parquet freq) = {corr:.3f} "
          f"over {n} patches")
    if corr < min_corr:
        raise RuntimeError(
            f"Class order mismatch between data loader and BEN_CLASSES_19 (corr={corr:.3f}). "
            "Per-class metrics and pos_weight would be wrong - fix before training."
        )
    return corr


def make_pos_weight(freq: np.ndarray, cap: float = 10.0, device="cpu") -> torch.Tensor:
    """BCE pos_weight = (1-f)/f, capped so rare classes cannot explode the loss."""
    f = np.clip(freq, 1e-3, 0.999)
    pw = np.minimum((1.0 - f) / f, cap)
    return torch.tensor(pw, dtype=torch.float32, device=device)