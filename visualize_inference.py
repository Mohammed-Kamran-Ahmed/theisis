import torch
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from models.decoders import PhysicalFeatureDecoder, OpticalImageDecoder

def normalize_band(band, p_min=2, p_max=98):
    """Applies percentile stretching to enhance SAR texture contrast."""
    low, high = np.percentile(band, (p_min, p_max))
    if high <= low:
        return np.zeros_like(band)
    stretched = np.clip((band - low) / (high - low), 0, 1)
    return stretched

def visualize_sar_inference(sar_tif_path, z_sar_vector):
    # 1. Load trained decoders
    feat_names = ['NDVI_mean', 'NDWI_mean', 'EVI_mean', 'RVI_mean', 'VH_VV_ratio_mean']
    feat_model = PhysicalFeatureDecoder(latent_dim=64, num_features=len(feat_names))
    img_model = OpticalImageDecoder(latent_dim=64, out_channels=3)

    feat_model.load_state_dict(torch.load("feature_decoder.pth"))
    img_model.load_state_dict(torch.load("image_decoder.pth"))

    feat_model.eval()
    img_model.eval()

    # 2. Run inference
    z_tensor = torch.tensor(z_sar_vector, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        pred_feats = feat_model(z_tensor).squeeze(0).numpy()
        gen_img_tensor = img_model(z_tensor).squeeze(0)

    # 3. Process Generated Array with Percentile Stretch
    gen_np = gen_img_tensor.cpu().numpy() # (3, H, W)
    gen_rgb = np.zeros((gen_np.shape[1], gen_np.shape[2], 3))
    for c in range(3):
        gen_rgb[:, :, c] = normalize_band(gen_np[c])

    # 4. Read SAR .tif and construct Ground Truth False-Color Composite
    with rasterio.open(sar_tif_path) as src:
        data = src.read()
        num_bands = src.count
        
        # Single Band (Grayscale intensity map)
        sar_primary = normalize_band(data[0])
        
        # Create False Color Composite if multi-band
        if num_bands >= 2:
            vv = normalize_band(data[0])
            vh = normalize_band(data[1])
            ratio = normalize_band(data[1] / (data[0] + 1e-6))
            gt_composite = np.dstack([vv, vh, ratio])
        else:
            gt_composite = np.dstack([sar_primary, sar_primary, sar_primary])

    # 5. Plot Outputs
    fig = plt.figure(figsize=(16, 5))
    
    # Subplot 1: Input SAR Patch (Grayscale Intensity)
    ax1 = fig.add_subplot(1, 4, 1)
    ax1.imshow(sar_primary, cmap='inferno')
    ax1.set_title("1. Input SAR Backscatter (VV)")
    ax1.axis('off')

    # Subplot 2: Decoded Physical Features
    ax2 = fig.add_subplot(1, 4, 2)
    bars = ax2.barh(feat_names, pred_feats, color='mediumseagreen', edgecolor='darkgreen')
    ax2.set_xlim([-0.2, 1.2])
    ax2.set_title("2. Decoded Physical Features")
    ax2.grid(True, linestyle='--', alpha=0.5)
    for bar in bars:
        width = bar.get_width()
        ax2.text(width + 0.02, bar.get_y() + bar.get_height()/2, f'{width:.3f}', va='center')

    # Subplot 3: Generated Synthetic Patch
    ax3 = fig.add_subplot(1, 4, 3)
    ax3.imshow(gen_rgb)
    ax3.set_title("3. Generated Patch Composite")
    ax3.axis('off')

    # Subplot 4: Ground Truth False-Color Composite
    ax4 = fig.add_subplot(1, 4, 4)
    ax4.imshow(gt_composite)
    ax4.set_title("4. Ground Truth False-Color")
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig("sample_inference_output.png", dpi=300)
    print("Inference visualization saved as 'sample_inference_output.png'!")
    plt.show()

if __name__ == "__main__":
    emb_df = pd.read_csv("patch_embeddings_stage5.csv")
    emb_cols = [c for c in emb_df.columns if c.startswith('z') or c.isdigit() or 'dim' in c]
    if len(emb_cols) < 64:
        emb_cols = emb_df.select_dtypes(include=[np.number]).columns[:64]

    sample_z = emb_df[emb_cols].iloc[0].values
    
    import glob
    sar_patches = sorted(glob.glob("downloaded_patches/*.tif"))
    if len(sar_patches) > 0:
        visualize_sar_inference(
            sar_tif_path=sar_patches[0], 
            z_sar_vector=sample_z
        )