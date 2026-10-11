"""
smoke_test.py - plumbing check on FAKE data (no LMDB, no GPU, ~1-2 min on CPU).

    python -m reben_pipeline.smoke_test

Verifies: variants train end-to-end, IERC fits and feeds the loss, an interrupted run
resumes with the correct LR schedule, existing output dirs are protected, the baseline
runs under the same protocol, and compare_runs can read the summaries.
It does NOT say anything about model quality - the data is synthetic.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from reben_pipeline import compare_runs, train_baseline, train_lmdb
from reben_pipeline.common import load_checkpoint
from reben_pipeline.metrics import BEN_CLASSES_19

N_TRAIN, N_VAL, N_TEST, SIZE = 96, 48, 48, 64
CLASS_P = np.array([0.05, 0.6, 0.02, 0.3, 0.02, 0.15, 0.35, 0.05, 0.1, 0.03,
                    0.04, 0.02, 0.2, 0.06, 0.1, 0.4, 0.03, 0.12, 0.25])


def make_split(n, rng, W):
    y = (rng.random((n, 19)) < CLASS_P).astype(np.float32)
    for i in range(n):  # at least one label per patch
        if y[i].sum() == 0:
            y[i, rng.integers(19)] = 1.0
    x = rng.normal(0, 0.3, (n, 12, SIZE, SIZE)).astype(np.float32)
    x += (y @ W)[:, :, None, None]  # learnable signal
    return torch.from_numpy(x), torch.from_numpy(y)


class FakeDM:
    def __init__(self, batch_size, seed=0):
        rng = np.random.default_rng(seed)
        W = rng.normal(0, 0.8, (19, 12)).astype(np.float32)
        self.data = {s: make_split(n, rng, W) for s, n in
                     (("train", N_TRAIN), ("val", N_VAL), ("test", N_TEST))}
        self.bs = batch_size
        self.train_ds = TensorDataset(*self.data["train"])

    def setup(self, stage=None):
        pass

    def _loader(self, split, shuffle):
        return DataLoader(TensorDataset(*self.data[split]), batch_size=self.bs, shuffle=shuffle)

    def train_dataloader(self):
        return self._loader("train", True)

    def val_dataloader(self):
        return self._loader("val", False)

    def test_dataloader(self):
        return self._loader("test", False)


def write_fake_parquet(path, dm):
    y = dm.data["train"][1].numpy()
    rows = [{"split": "train", "labels": [BEN_CLASSES_19[c] for c in np.where(r > 0)[0]]} for r in y]
    rows += [{"split": "val", "labels": [BEN_CLASSES_19[1]]}, {"split": "test", "labels": [BEN_CLASSES_19[1]]}]
    pd.DataFrame(rows).to_parquet(path)


def main():
    tmp = Path(tempfile.mkdtemp(prefix="reben_smoke_"))
    try:
        dm_holder = {}

        def fake_build(lmdb_dir, parquet, batch_size, num_workers=0):
            dm_holder["dm"] = FakeDM(batch_size)
            return dm_holder["dm"]

        train_lmdb.build_data_module = fake_build
        train_baseline.build_data_module = fake_build
        parquet = tmp / "meta.parquet"
        write_fake_parquet(parquet, FakeDM(16))

        common = ["--parquet", str(parquet), "--lmdb_dir", "unused", "--backbone", "resnet18",
                  "--no_pretrained", "--batch_size", "16", "--lr", "1e-3", "--seed", "1"]

        # 1) full method, uninterrupted
        print("\n=== 1/6 cahn_ierc, uninterrupted (4 epochs) ===")
        a = train_lmdb.main(common + ["--variant", "cahn_ierc", "--epochs", "4", "--ierc_start", "2",
                                      "--ierc_interval", "1", "--output_dir", str(tmp / "a")])
        assert a is not None and np.isfinite(a["test"]["fusion"]["ap_macro"])
        ck = load_checkpoint(tmp / "a" / "last.pt")
        assert ck["ierc_state"]["clusterer"] is not None, "IERC never fitted"
        assert any(h["train_anchor_frac"] > 0 for h in a["history"]), "CAHN saw no valid anchors"

        # 2) interrupted + resumed run must follow the same LR schedule
        print("\n=== 2/6 cahn_ierc, interrupted after epoch 2, then --resume ===")
        args_b = common + ["--variant", "cahn_ierc", "--epochs", "4", "--ierc_start", "2",
                           "--ierc_interval", "1", "--output_dir", str(tmp / "b")]
        assert train_lmdb.main(args_b + ["--stop_after_epoch", "2"]) is None
        b = train_lmdb.main(args_b + ["--resume"])
        lr_a = [h["lr"] for h in a["history"]]
        lr_b = [h["lr"] for h in b["history"]]
        assert np.allclose(lr_a, lr_b), f"LR schedule differs after resume: {lr_a} vs {lr_b}"
        assert [h["epoch"] for h in b["history"]] == [1, 2, 3, 4]
        print(f"   LR schedule identical after resume: {np.round(lr_b, 6).tolist()}")
        print(f"   final val AP-macro uninterrupted {a['history'][-1]['val_ap_macro']:.2f} | "
              f"resumed {b['history'][-1]['val_ap_macro']:.2f}")

        # 3) protection against silent overwrite
        print("\n=== 3/6 existing output dir is protected ===")
        try:
            train_lmdb.main(args_b)
            raise AssertionError("expected SystemExit")
        except SystemExit as e:
            print(f"   refused as expected: {str(e)[:70]}...")

        # 4) ablation variants
        print("\n=== 4/6 variants cls / align / cahn (2 epochs each) ===")
        for v in ("cls", "align", "cahn"):
            s = train_lmdb.main(common + ["--variant", v, "--epochs", "2", "--output_dir", str(tmp / v)])
            assert np.isfinite(s["test"]["fusion"]["ap_macro"])
            if v in ("cls", "align"):
                assert s["history"][-1]["train_cahn"] == 0.0

        # 5) baseline
        print("\n=== 5/6 baseline (2 epochs) ===")
        train_baseline.main(["--parquet", str(parquet), "--lmdb_dir", "unused", "--no_pretrained",
                             "--batch_size", "16", "--lr", "1e-3", "--seed", "1", "--epochs", "2",
                             "--output_dir", str(tmp / "base")])

        # 6) compare
        print("\n=== 6/6 compare_runs ===")
        compare_runs.main([str(tmp / d) for d in ("a", "cls", "align", "cahn", "base")])

        print("\nSMOKE TEST PASSED (plumbing only - these numbers say nothing about model quality)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()