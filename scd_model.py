"""
Spatially-Conditioned Decoder (SCD) — FiLM-modulated U-Net.

Implements paper Section 5.3 exactly:
  - Retains SAR-encoder skip feature maps (skip1@32x32, skip2@16x16, skip3@8x8)
  - Modulates every decoder block with the global fused embedding z via FiLM (Eq. 12)
  - Trains with combined pixel + structural + perceptual loss (Eq. 13)

PLUG-IN POINTS (search for "TODO" below):
  1. `SAREncoderWithSkips` assumes your SAR backbone is a torchvision resnet18.
     If your existing f_sar is already a resnet18, just swap in its state_dict.
  2. `SCDDataset` / the training loop assume a dataloader yielding
     (sar_img, opt_img, z_embedding) triples. Replace with your existing
     dataset class — you already compute z (the frozen fused 64-dim embedding)
     from Stage B/CAHN/IERC, so just cache those alongside patch ids.

Run:
    python scd_model.py --data_dir /path/to/patches --z_cache /path/to/z_embeddings.npz \
        --epochs 120 --out_dir ./scd_runs
"""
import argparse
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvm


# --------------------------------------------------------------------------
# 1. SAR encoder that also returns intermediate skip feature maps
# --------------------------------------------------------------------------
class SAREncoderWithSkips(nn.Module):
    """ResNet-18 backbone that returns (embedding_h, [skip1, skip2, skip3]).

    For a 128x128 input, resnet18's stages produce:
      layer1 -> 32x32 x  64ch   (skip1)
      layer2 -> 16x16 x 128ch   (skip2)
      layer3 ->  8x8  x 256ch   (skip3)
      layer4 ->  4x4  x 512ch   (pooled -> h, 128-dim after projection to match paper's h)
    """

    def __init__(self, pretrained_backbone: nn.Module = None):
        super().__init__()
        backbone = pretrained_backbone if pretrained_backbone is not None else tvm.resnet18(weights=None)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem(x)
        s1 = self.layer1(x)   # 32x32 x 64
        s2 = self.layer2(s1)  # 16x16 x 128
        s3 = self.layer3(s2)  #  8x8  x 256
        x4 = self.layer4(s3)  #  4x4  x 512
        h = self.pool(x4).flatten(1)  # 512-dim backbone feature
        return h, [s1, s2, s3]

    def load_pretrained_encoder(self, state_dict_path: str):
        """TODO: point this at your already-trained f_sar (post-IERC) checkpoint."""
        sd = torch.load(state_dict_path, map_location="cpu")
        self.load_state_dict(sd, strict=False)


# --------------------------------------------------------------------------
# 2. FiLM block
# --------------------------------------------------------------------------
class FiLMGenerator(nn.Module):
    """MLP(z) -> per-channel (gamma, beta) for a target feature map with C channels."""

    def __init__(self, z_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, channels * 2),
        )
        self.channels = channels

    def forward(self, z):
        out = self.net(z)  # (B, 2C)
        gamma, beta = out[:, :self.channels], out[:, self.channels:]
        return gamma, beta


class FiLMDecoderBlock(nn.Module):
    """Upsample -> conv -> FiLM(z) -> concat skip -> conv -> ReLU  (Eq. 12: F' = gamma(z)*F + beta(z))"""

    def __init__(self, in_ch, skip_ch, out_ch, z_dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv_in = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.film = FiLMGenerator(z_dim, out_ch)
        self.conv_merge = nn.Conv2d(out_ch + skip_ch, out_ch, 3, padding=1)
        self.norm = nn.InstanceNorm2d(out_ch, affine=False)

    def forward(self, x, z, skip=None):
        x = self.up(x)
        x = self.conv_in(x)
        x = self.norm(x)
        gamma, beta = self.film(z)
        x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
        x = F.relu(x, inplace=True)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                skip = F.interpolate(skip, size=x.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = F.relu(self.conv_merge(x), inplace=True)
        return x


# --------------------------------------------------------------------------
# 3. Full SCD: z (+ skips) -> 128x128x3 RGB
# --------------------------------------------------------------------------
class SpatiallyConditionedDecoder(nn.Module):
    def __init__(self, z_dim: int = 64):
        super().__init__()
        self.z_dim = z_dim
        self.fc_in = nn.Linear(z_dim, 512 * 4 * 4)
        # 4x4 -> 8x8 (merge skip3, 256ch) -> 16x16 (merge skip2, 128ch)
        # -> 32x32 (merge skip1, 64ch) -> 64x64 (no skip) -> 128x128 (no skip)
        self.block1 = FiLMDecoderBlock(512, 256, 256, z_dim)  # -> 8x8
        self.block2 = FiLMDecoderBlock(256, 128, 128, z_dim)  # -> 16x16
        self.block3 = FiLMDecoderBlock(128, 64, 64, z_dim)    # -> 32x32
        self.block4 = FiLMDecoderBlock(64, 0, 32, z_dim)      # -> 64x64
        self.block5 = FiLMDecoderBlock(32, 0, 16, z_dim)      # -> 128x128
        self.to_rgb = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, z, skips):
        s1, s2, s3 = skips  # 32x32x64, 16x16x128, 8x8x256
        x = self.fc_in(z).view(-1, 512, 4, 4)
        x = self.block1(x, z, s3)
        x = self.block2(x, z, s2)
        x = self.block3(x, z, s1)
        x = self.block4(x, z, None)
        x = self.block5(x, z, None)
        x = torch.sigmoid(self.to_rgb(x))  # normalized [0,1] RGB target
        return x


# --------------------------------------------------------------------------
# 4. Loss (Eq. 13): MSE + 0.5*L1 + 0.5*(1-SSIM) + 0.2*perceptual(VGG16 first 16 layers)
# --------------------------------------------------------------------------
class SCDLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        vgg = tvm.vgg16(weights=tvm.VGG16_Weights.IMAGENET1K_V1).features[:16].to(device).eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.register_buffer_std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def ssim(self, x, y, C1=0.01 ** 2, C2=0.03 ** 2, window=11):
        pad = window // 2
        mu_x = F.avg_pool2d(x, window, 1, pad)
        mu_y = F.avg_pool2d(y, window, 1, pad)
        sigma_x = F.avg_pool2d(x * x, window, 1, pad) - mu_x ** 2
        sigma_y = F.avg_pool2d(y * y, window, 1, pad) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x * y, window, 1, pad) - mu_x * mu_y
        ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
        )
        return ssim_map.mean()

    def perceptual(self, x, y):
        xn = (x - self.register_buffer_mean) / self.register_buffer_std
        yn = (y - self.register_buffer_mean) / self.register_buffer_std
        return F.l1_loss(self.vgg(xn), self.vgg(yn))

    def forward(self, pred, target):
        mse = F.mse_loss(pred, target)
        l1 = F.l1_loss(pred, target)
        ssim_val = self.ssim(pred, target)
        perc = self.perceptual(pred, target)
        total = mse + 0.5 * l1 + 0.5 * (1 - ssim_val) + 0.2 * perc
        return total, {"mse": mse.item(), "l1": l1.item(), "ssim": ssim_val.item(), "perc": perc.item()}


# --------------------------------------------------------------------------
# 5. PSNR metric
# --------------------------------------------------------------------------
def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target).item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


# --------------------------------------------------------------------------
# 6. Dataset stub — REPLACE with your real dataset class
# --------------------------------------------------------------------------
class SCDDataset(Dataset):
    """TODO: replace this stub.

    Expected to return:
        sar_img  : FloatTensor (5, 128, 128)  -- your 5-channel SAR patch
        opt_img  : FloatTensor (3, 128, 128)  -- normalized RGB target, [0,1]
        z        : FloatTensor (64,)          -- frozen fused embedding from IERC stage

    Simplest integration: precompute z for every patch once (you already do this
    for the CCA benchmark), save as a .npz {patch_id: z}, and load it here keyed
    by the same patch_id your existing SAR/optical loader uses.
    """

    def __init__(self, data_dir, z_cache_path, split="train"):
        self.data_dir = data_dir
        z_cache = np.load(z_cache_path, allow_pickle=True)
        self.patch_ids = list(z_cache["patch_ids"])
        self.z_all = z_cache["z"]  # (N, 64)
        self.split = split
        # TODO: load your actual sar/opt patch arrays here (e.g. from .npy/.tif per id)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        raise NotImplementedError(
            "Wire this up to your existing SAR/optical patch loader "
            "(same one used for Stage B / CAHN / IERC)."
        )


# --------------------------------------------------------------------------
# 7. Training loop
# --------------------------------------------------------------------------
def train_scd(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SCD] device = {device}")

    sar_encoder = SAREncoderWithSkips().to(device)
    if args.sar_encoder_ckpt:
        sar_encoder.load_pretrained_encoder(args.sar_encoder_ckpt)
    sar_encoder.eval()
    for p in sar_encoder.parameters():
        p.requires_grad = False  # frozen: SCD only trains the decoder, per paper §5.3

    decoder = SpatiallyConditionedDecoder(z_dim=64).to(device)
    loss_fn = SCDLoss(device).to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    train_ds = SCDDataset(args.data_dir, args.z_cache, split="train")
    test_ds = SCDDataset(args.data_dir, args.z_cache, split="test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    os.makedirs(args.out_dir, exist_ok=True)
    log = {"train_loss": [], "test_ssim": [], "test_psnr": []}

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        epoch_loss = 0.0
        t0 = time.time()
        for sar_img, opt_img, z in train_loader:
            sar_img, opt_img, z = sar_img.to(device), opt_img.to(device), z.to(device)
            with torch.no_grad():
                _, skips = sar_encoder(sar_img)
            pred = decoder(z, skips)
            loss, parts = loss_fn(pred, opt_img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * sar_img.size(0)
        sched.step()
        epoch_loss /= len(train_ds)
        log["train_loss"].append(epoch_loss)

        if epoch % 10 == 0 or epoch == args.epochs:
            decoder.eval()
            ssim_sum, psnr_sum, n = 0.0, 0.0, 0
            with torch.no_grad():
                for sar_img, opt_img, z in test_loader:
                    sar_img, opt_img, z = sar_img.to(device), opt_img.to(device), z.to(device)
                    _, skips = sar_encoder(sar_img)
                    pred = decoder(z, skips)
                    ssim_sum += loss_fn.ssim(pred, opt_img).item() * sar_img.size(0)
                    psnr_sum += psnr(pred, opt_img) * sar_img.size(0)
                    n += sar_img.size(0)
            test_ssim, test_psnr = ssim_sum / n, psnr_sum / n
            log["test_ssim"].append(test_ssim)
            log["test_psnr"].append(test_psnr)
            print(f"[epoch {epoch:3d}] train_loss={epoch_loss:.4f} "
                  f"test_SSIM={test_ssim:.4f} test_PSNR={test_psnr:.2f}dB "
                  f"({time.time()-t0:.1f}s)")

    torch.save(decoder.state_dict(), os.path.join(args.out_dir, "scd_decoder.pth"))
    with open(os.path.join(args.out_dir, "scd_training_log.json"), "w") as f:
        json.dump(log, f, indent=2)
    print(f"[SCD] done. Final test SSIM={log['test_ssim'][-1]:.4f}, "
          f"PSNR={log['test_psnr'][-1]:.2f}dB. "
          f"Compare directly against flat-decoder baseline: SSIM=0.5186, PSNR=15.91dB.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--z_cache", required=True, help="npz with patch_ids + fused 64-dim z embeddings")
    p.add_argument("--sar_encoder_ckpt", default=None, help="path to your trained/IERC'd f_sar weights")
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--out_dir", default="./scd_runs")
    args = p.parse_args()
    train_scd(args)
