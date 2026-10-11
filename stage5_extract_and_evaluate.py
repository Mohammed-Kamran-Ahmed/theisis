"""
stage5_extract_and_evaluate.py
================================
Rewritten to actually produce what Table 3 of the paper needs: Silhouette /
Calinski-Harabasz / Davies-Bouldin for BOTH the baseline and IERC
checkpoints, split into TRAIN and TEST using the 'split' column, in TWO
embedding spaces.

ARCHITECTURE NOTE: stage4_contrastive_learning.ModalityEncoder has no real
128-dimensional intermediate stage -- it goes straight from the ResNet-18's
512-dim pooled features to a `proj_dim`-dim (64) projection. There is no
literal "128-dim backbone embedding h" in the current code. As a practical,
clearly-labeled substitute (NOT a retrained architecture change):
  - "z" (projection space, 64-dim)  = the fused/averaged, L2-normalized
    projection (z_sar + z_opt)/2 -- this matches what compute_test_silhouette
    already uses everywhere else in the pipeline.
  - "h" (concat space, 128-dim)     = the raw concatenation [z_sar, z_opt]
    BEFORE averaging/fusion -- the best available 128-dim stand-in given the
    current encoder, but not the same thing the paper's methodology text
    describes. State this substitution explicitly if you report these
    numbers as "backbone space h".

Usage: run once per checkpoint you want evaluated. Produces one row per
(run_label, space, split) combination, appended to
stage5_full_evaluation.csv.

    python stage5_extract_and_evaluate.py --checkpoint stage4_best.pth --label baseline
    python stage5_extract_and_evaluate.py --checkpoint stage4b_ierc_best.pth --label IERC
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import rasterio

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

from stage4_contrastive_learning import ModalityEncoder
from seed_lock import set_seed  # <-- ADDED

PATCH_DIR = "./downloaded_patches"
MODEL_DIR = "./saved_models"
OUTPUT_CSV = "./stage5_full_evaluation.csv"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Which (metadata file, label column) each run should use. This is the fix:
# the "baseline" checkpoint was trained against the ORIGINAL gmm_cluster
# labels and must be evaluated against those -- not against IERC's
# pseudo_label column, even though patch_metadata_reclustered.csv (written
# by Stage 4b) has both columns present for every row. Loading the wrong
# file/column here silently mislabels the baseline embeddings and produces
# a baseline silhouette that does not match what training actually logged.
LABEL_SOURCE = {
    "baseline": {"file": "patch_metadata_clustered.csv", "column": "gmm_cluster"},
    "IERC": {"file": "patch_metadata_reclustered.csv", "column": "pseudo_label"},
}


class FinalEmbeddingDataset(Dataset):
    def __init__(self, df, label_column, patch_dir=PATCH_DIR, target_size=(224, 224)):
        self.df = df.reset_index(drop=True)
        self.label_column = label_column
        self.patch_dir = patch_dir
        self.target_size = target_size

    def __len__(self):
        return len(self.df)

    def resolve_path(self, row, idx):
        columns = ["filepath", "local_path", "patch_path", "export_name", "filename"]
        for col in columns:
            if col in row and pd.notna(row[col]):
                value = str(row[col])
                if os.path.exists(value):
                    return value
                candidate = os.path.join(self.patch_dir, os.path.basename(value))
                if os.path.exists(candidate):
                    return candidate
        patch_id = row.get("patch_id", None)
        if patch_id is not None:
            for ext in [".tif", ".tiff", ".png"]:
                candidate = os.path.join(self.patch_dir, f"{patch_id}{ext}")
                if os.path.exists(candidate):
                    return candidate
        raise FileNotFoundError(f"Patch not found for index {idx}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = self.resolve_path(row, idx)

        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        if data.shape[0] >= 5:
            sar, optical = data[:2], data[2:5]
        elif data.shape[0] >= 3:
            sar, optical = data[:2], data[:3]
        else:
            sar, optical = data, data

        sar = torch.from_numpy(sar)
        optical = torch.from_numpy(optical)
        if sar.shape[0] < 3:
            sar = sar.repeat(3, 1, 1)[:3]
        if optical.shape[0] < 3:
            optical = optical.repeat(3, 1, 1)[:3]

        sar = F.interpolate(sar.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)
        optical = F.interpolate(optical.unsqueeze(0), size=self.target_size, mode="bilinear", align_corners=False).squeeze(0)

        label = int(row[self.label_column])
        split = str(row.get("split", "unknown"))
        return sar, optical, label, split


def evaluate_latent_space(embeddings, labels):
    if len(np.unique(labels)) < 2 or len(embeddings) < 3:
        return {"Silhouette": float("nan"), "Calinski-Harabasz": float("nan"), "Davies-Bouldin": float("nan")}
    return {
        "Silhouette": silhouette_score(embeddings, labels),
        "Calinski-Harabasz": calinski_harabasz_score(embeddings, labels),
        "Davies-Bouldin": davies_bouldin_score(embeddings, labels),
    }


def main():
    set_seed(SEED)  # <-- ADDED

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                         help="Checkpoint filename in ./saved_models, e.g. stage4_best.pth or stage4b_ierc_best.pth")
    parser.add_argument("--label", required=True,
                         help="Run label for the output rows, e.g. baseline or IERC")
    args = parser.parse_args()

    metadata = "patch_metadata_reclustered.csv"
    if not os.path.exists(metadata):
        metadata = "patch_metadata_clustered.csv"
    if not os.path.exists(metadata):
        raise FileNotFoundError("No metadata file found.")

    df = pd.read_csv(metadata)
    df = df[df.get("valid", True) == True].copy().reset_index(drop=True)

    if "split" not in df.columns:
        raise ValueError("No 'split' column in metadata -- cannot evaluate train/test separately.")

    if args.label not in LABEL_SOURCE:
        raise ValueError(f"--label must be one of {list(LABEL_SOURCE.keys())} so the correct "
                          f"label column is used (see LABEL_SOURCE at top of file).")
    label_column = LABEL_SOURCE[args.label]["column"]
    if label_column not in df.columns:
        raise ValueError(f"Expected label column '{label_column}' for --label {args.label} "
                          f"not found in {metadata}. Columns present: {df.columns.tolist()}")
    print(f"Using label column '{label_column}' from {metadata} for run '{args.label}'.\n")

    checkpoint_path = os.path.join(MODEL_DIR, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    sar_enc = ModalityEncoder(proj_dim=64).to(DEVICE)
    opt_enc = ModalityEncoder(proj_dim=64).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    sar_enc.load_state_dict(state["sar_state"])
    opt_enc.load_state_dict(state["opt_state"])
    sar_enc.eval()
    opt_enc.eval()

    dataset = FinalEmbeddingDataset(df, label_column=label_column)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    z_all, h_all, labels_all, splits_all = [], [], [], []

    with torch.no_grad():
        for sar, optical, lbl, split in loader:
            sar, optical = sar.to(DEVICE), optical.to(DEVICE)
            z_sar = sar_enc(sar)
            z_opt = opt_enc(optical)

            z_fused = F.normalize((z_sar + z_opt) / 2.0, p=2, dim=1)
            h_concat = torch.cat([z_sar, z_opt], dim=1)  # 128-dim stand-in, see module docstring

            z_all.append(z_fused.cpu().numpy())
            h_all.append(h_concat.cpu().numpy())
            labels_all.extend(lbl.numpy())
            splits_all.extend(split)

    z_all = np.nan_to_num(np.vstack(z_all))
    h_all = np.nan_to_num(np.vstack(h_all))
    labels_all = np.asarray(labels_all)
    splits_all = np.asarray(splits_all)

    rows = []
    for space_name, embeddings in [("z (64-dim, fused)", z_all), ("h (128-dim, concat)", h_all)]:
        for split_name in ["train", "test"]:
            mask = splits_all == split_name
            n = int(mask.sum())
            if n == 0:
                continue
            metrics = evaluate_latent_space(embeddings[mask], labels_all[mask])
            rows.append({
                "run_label": args.label,
                "space": space_name,
                "split": split_name,
                "n": n,
                "silhouette": metrics["Silhouette"],
                "calinski_harabasz": metrics["Calinski-Harabasz"],
                "davies_bouldin": metrics["Davies-Bouldin"],
            })
            print(f"[{args.label}] {space_name} / {split_name} (n={n}): "
                  f"silhouette={metrics['Silhouette']:.4f}  "
                  f"C-H={metrics['Calinski-Harabasz']:.2f}  "
                  f"D-B={metrics['Davies-Bouldin']:.4f}")

    new_df = pd.DataFrame(rows)
    if os.path.exists(OUTPUT_CSV):
        existing = pd.read_csv(OUTPUT_CSV)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    combined.to_csv(OUTPUT_CSV, index=False)
    print(f"\nAppended to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()