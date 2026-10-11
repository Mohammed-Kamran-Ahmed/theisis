"""
view_spatial_output.py
=========================
Same idea as view_single_output.py, but for the new spatial U-Net decoder.

Usage:
    python view_spatial_output.py --index 0
"""

import argparse
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from train_spatial_decoder import (
    SpatialDecoderDataset, INPUT_CSV, PATCH_DIR, SAR_ENCODER_PATH,
    SAR_BAND_INDICES, NORM_STATS_PATH,
)
from models.sar_encoder_spatial import load_frozen_sar_encoder
from models.spatial_unet_decoder import SpatialUNetDecoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)
    test_df = df[(df["valid"] == True) & (df["split"] == "test")].copy().reset_index(drop=True)

    with open(NORM_STATS_PATH) as f:
        norm_stats = json.load(f)

    test_dataset = SpatialDecoderDataset(test_df, PATCH_DIR, norm_stats, augment=False)

    sar_tensor, rgb_tensor = test_dataset[args.index]
    patch_id = test_df.iloc[args.index].get("export_name", f"index_{args.index}")

    sar_encoder = load_frozen_sar_encoder(SAR_ENCODER_PATH, in_channels=len(SAR_BAND_INDICES), device=DEVICE)
    decoder = SpatialUNetDecoder(out_channels=3).to(DEVICE)
    decoder.load_state_dict(torch.load("spatial_image_decoder.pth", map_location=DEVICE))
    decoder.eval()

    with torch.no_grad():
        f1, f2, f3, f4, _ = sar_encoder.forward_spatial(sar_tensor.unsqueeze(0).to(DEVICE))
        gen_img = decoder(f1, f2, f3, f4).squeeze(0).cpu().numpy()

    gen_disp = np.clip(np.transpose(gen_img, (1, 2, 0)), 0, 1)
    true_disp = np.clip(np.transpose(rgb_tensor.numpy(), (1, 2, 0)), 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(9, 5))
    axes[0].imshow(gen_disp)
    axes[0].set_title("Spatial U-Net Decoder Output\n(SAR-only, skip connections)")
    axes[0].axis("off")

    axes[1].imshow(true_disp)
    axes[1].set_title("Ground-Truth Optical")
    axes[1].axis("off")

    fig.suptitle(f"Spatial decoder output for patch: {patch_id}", fontsize=11)
    plt.tight_layout()
    plt.savefig("spatial_output_view.png", dpi=150, bbox_inches="tight")
    print("Saved spatial_output_view.png")
    plt.show()


if __name__ == "__main__":
    main()