"""
view_single_output.py
=======================
Shows the ACTUAL final output of both trained decoders for one patch,
in one figure. Run this any time you just want to SEE what the model
produces, without digging through evaluation scripts.

Usage:
    python view_single_output.py                 # shows test patch #0
    python view_single_output.py --index 5        # shows test patch #5
    python view_single_output.py --patch_id XYZ   # shows a specific patch by id
"""

import argparse
import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_decoders import ThesisRealDataset, NORM_STATS_PATH, TARGET_FEATURE_COLUMNS
from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="index within the test split")
    parser.add_argument("--patch_id", type=str, default=None, help="specific patch_id to look up")
    args = parser.parse_args()

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    test_dataset = ThesisRealDataset(
        embeddings_csv="patch_embeddings_stage5.csv",
        metadata_csv="patch_metadata_clustered.csv",
        patches_dir="downloaded_patches",
        split="test",
        norm_stats=norm_stats,
    )

    idx = args.index
    if args.patch_id is not None:
        idx = test_dataset.patch_ids.index(args.patch_id)

    z, true_feat, true_img = test_dataset[idx]
    patch_id = test_dataset.patch_ids[idx]

    feat_model = PhysicalFeatureDecoder(latent_dim=z.shape[0], num_features=true_feat.shape[0])
    img_model = OpticalImageDecoder(latent_dim=z.shape[0], out_channels=3)
    feat_model.load_state_dict(torch.load("feature_decoder.pth"))
    img_model.load_state_dict(torch.load("image_decoder.pth"))
    feat_model.eval()
    img_model.eval()

    with torch.no_grad():
        pred_feat = feat_model(z.unsqueeze(0)).squeeze(0).numpy()
        pred_img = img_model(z.unsqueeze(0)).squeeze(0).numpy()

    true_feat = true_feat.numpy()
    pred_img_disp = np.clip(np.transpose(pred_img, (1, 2, 0)), 0, 1)
    true_img_disp = np.clip(np.transpose(true_img.numpy(), (1, 2, 0)), 0, 1)

    print(f"\nPatch: {patch_id}")
    print("=" * 50)
    print(f"{'Feature':<20}{'Actual':>12}{'Predicted':>12}")
    for name, t, p in zip(TARGET_FEATURE_COLUMNS, true_feat, pred_feat):
        print(f"{name:<20}{t:>12.4f}{p:>12.4f}")
    print("=" * 50)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    axes[0].imshow(pred_img_disp)
    axes[0].set_title("Generated Optical\n(from SAR embedding alone)")
    axes[0].axis("off")

    axes[1].imshow(true_img_disp)
    axes[1].set_title("Ground-Truth Optical")
    axes[1].axis("off")

    x = np.arange(len(TARGET_FEATURE_COLUMNS))
    axes[2].bar(x - 0.2, true_feat, width=0.4, label="Actual", color="#4C72B0")
    axes[2].bar(x + 0.2, pred_feat, width=0.4, label="Predicted", color="#DD8452")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([c.replace("_mean", "") for c in TARGET_FEATURE_COLUMNS], rotation=30, fontsize=8)
    axes[2].set_title("Physical Features:\nSAR-only prediction vs. real")
    axes[2].legend()

    fig.suptitle(f"Decoder output for patch: {patch_id}", fontsize=12)
    plt.tight_layout()
    plt.savefig("single_output_view.png", dpi=150, bbox_inches="tight")
    print("\nSaved single_output_view.png -- open it to see the actual output.")
    plt.show()


if __name__ == "__main__":
    main()