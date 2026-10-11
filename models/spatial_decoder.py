"""
models/spatial_decoder.py
==========================
Novelty 3 (PRIMARY CONTRIBUTION) — Spatially-Conditioned SAR→Optical Decoder.

Problem with the existing OpticalImageDecoder:
  The baseline decoder takes a single 64-dim projection vector, projects it to
  256×16×16 via a fully-connected layer, then upsamples. ALL spatial structure
  must be hallucinated from the bottleneck vector — the decoder has no access
  to where vegetation, water, or bare soil patches were in the SAR image. This
  is why the baseline achieves only SSIM≈0.52 and PSNR≈15.9 dB.

Our solution — Spatially-Conditioned Decoder (SCD):
  We re-use the SPATIAL FEATURE MAPS from intermediate SAR encoder layers via
  U-Net-style skip connections. The SAR image already encodes spatial structure
  (field boundaries, texture gradients, water bodies) in its intermediate
  feature maps — we just need to pass them directly to the decoder.

  Additionally, the global 64-dim patch embedding is injected at the bottleneck
  using FiLM (Feature-wise Linear Modulation), giving the decoder semantic
  context (which cluster / season / crop type) while skip connections provide
  pixel-level spatial guidance.

Architecture (input: 64×64 SAR patches, output: 128×128 optical RGB):
  Encoder spatial maps (from SAR ResNet-18 with 64×64 input):
    bottleneck : [B, 512,  2,  2]  (after layer4 + avgpool not applied)
    skip3      : [B, 256,  4,  4]  (after layer3)
    skip2      : [B, 128,  8,  8]  (after layer2)
    skip1      : [B,  64, 16, 16]  (after layer1)

  Decoder upsampling path:
    [B,512,2,2] → FiLM(embed) → up2x →
    [B,512,4,4] + skip3[B,256,4,4] → ConvBlock → [B,256,4,4]
    [B,256,4,4] → up2x →
    [B,256,8,8] + skip2[B,128,8,8] → ConvBlock → [B,128,8,8]
    [B,128,8,8] → up2x →
    [B,128,16,16] + skip1[B,64,16,16] → ConvBlock → [B,64,16,16]
    [B,64,16,16] → up2x → ConvBlock → [B,32,32,32]
    [B,32,32,32] → up2x → ConvBlock → [B,16,64,64]
    [B,16,64,64] → up2x → ConvBlock → [B,8,128,128]
    [B,8,128,128] → Conv1×1 → Sigmoid → [B,3,128,128]

References:
  - Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image
    Segmentation", MICCAI 2015.
  - Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer",
    AAAI 2018.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# BUILDING BLOCKS
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Standard U-Net double-conv block:  Conv→BN→ReLU → Conv→BN→ReLU.
    Optional Dropout2d for regularisation in deeper blocks.
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout2d(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FiLMConditioner(nn.Module):
    """
    FiLM (Feature-wise Linear Modulation) conditioning layer.

    Projects the global 64-dim patch embedding to channel-wise scale (γ)
    and shift (β) parameters and applies them to a spatial feature map:
        out[b,c,h,w] = feat[b,c,h,w] * (1 + γ_c) + β_c

    This allows the decoder to modulate its behaviour based on semantic
    context (cluster type, season) without discarding spatial information.
    """
    def __init__(self, embed_dim: int, spatial_channels: int):
        super().__init__()
        self.gamma_fc = nn.Linear(embed_dim, spatial_channels)
        self.beta_fc  = nn.Linear(embed_dim, spatial_channels)
        # Initialise to near-identity: gamma≈0, beta≈0 → minimal distortion at start
        nn.init.zeros_(self.gamma_fc.weight)
        nn.init.zeros_(self.gamma_fc.bias)
        nn.init.zeros_(self.beta_fc.weight)
        nn.init.zeros_(self.beta_fc.bias)

    def forward(self, feat_map: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat_map  : [B, C, H, W] spatial feature tensor.
            embedding : [B, embed_dim] global semantic embedding.
        Returns:
            FiLM-conditioned [B, C, H, W].
        """
        gamma = self.gamma_fc(embedding).unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]
        beta  = self.beta_fc(embedding).unsqueeze(-1).unsqueeze(-1)   # [B,C,1,1]
        return feat_map * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# MAIN DECODER: SpatiallyConditionedDecoder
# ---------------------------------------------------------------------------

class SpatiallyConditionedDecoder(nn.Module):
    """
    U-Net decoder with FiLM embedding conditioning.

    Takes SAR encoder spatial feature maps (skip connections) and the
    global 64-dim projection embedding, and reconstructs a 128×128 RGB
    optical image that is spatially consistent with the input SAR patch.

    Args:
        embed_dim   : Dimension of the global embedding (default 64).
        out_channels: Number of output image channels (default 3, RGB).
    """

    def __init__(self, embed_dim: int = 64, out_channels: int = 3):
        super().__init__()

        # FiLM: inject global embedding into bottleneck [B,512,2,2]
        self.film = FiLMConditioner(embed_dim, spatial_channels=512)

        # ── Upsampling path ─────────────────────────────────────────────
        # Level 4: 2→4, cat with skip3 [256ch] → [256ch]
        self.up4   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = ConvBlock(512 + 256, 256, dropout=0.2)

        # Level 3: 4→8, cat with skip2 [128ch] → [128ch]
        self.up3   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = ConvBlock(256 + 128, 128, dropout=0.1)

        # Level 2: 8→16, cat with skip1 [64ch] → [64ch]
        self.up2   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = ConvBlock(128 + 64, 64)

        # Level 1: 16→32, no skip → [32ch]
        self.up1   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = ConvBlock(64, 32)

        # Level 0: 32→64, no skip → [16ch]
        self.up0   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv0 = ConvBlock(32, 16)

        # Level -1: 64→128, no skip → [8ch]
        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv_final = ConvBlock(16, 8)

        # Output projection [B,8,128,128] → [B,out_channels,128,128]
        self.out_conv = nn.Conv2d(8, out_channels, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,   # [B, 512, 2, 2]
        skip3:      torch.Tensor,   # [B, 256, 4, 4]
        skip2:      torch.Tensor,   # [B, 128, 8, 8]
        skip1:      torch.Tensor,   # [B,  64, 16, 16]
        embedding:  torch.Tensor,   # [B, embed_dim]
    ) -> torch.Tensor:
        """Returns [B, out_channels, 128, 128] optical image prediction."""

        # FiLM-condition the bottleneck with global patch embedding
        x = self.film(bottleneck, embedding)   # [B, 512, 2, 2]

        # 2 → 4, concat skip3
        x = self.up4(x)                        # [B, 512, 4, 4]
        x = torch.cat([x, skip3], dim=1)       # [B, 768, 4, 4]
        x = self.conv4(x)                      # [B, 256, 4, 4]

        # 4 → 8, concat skip2
        x = self.up3(x)                        # [B, 256, 8, 8]
        x = torch.cat([x, skip2], dim=1)       # [B, 384, 8, 8]
        x = self.conv3(x)                      # [B, 128, 8, 8]

        # 8 → 16, concat skip1
        x = self.up2(x)                        # [B, 128, 16, 16]
        x = torch.cat([x, skip1], dim=1)       # [B, 192, 16, 16]
        x = self.conv2(x)                      # [B,  64, 16, 16]

        # 16 → 32, no skip
        x = self.up1(x)                        # [B,  64, 32, 32]
        x = self.conv1(x)                      # [B,  32, 32, 32]

        # 32 → 64, no skip
        x = self.up0(x)                        # [B,  32, 64, 64]
        x = self.conv0(x)                      # [B,  16, 64, 64]

        # 64 → 128, no skip
        x = self.up_final(x)                   # [B,  16, 128, 128]
        x = self.conv_final(x)                 # [B,   8, 128, 128]

        return torch.sigmoid(self.out_conv(x)) # [B,   3, 128, 128]


# ---------------------------------------------------------------------------
# SAR ENCODER WRAPPER:  exposes skip connections
# ---------------------------------------------------------------------------

class SAREcoderWithSkips(nn.Module):
    """
    Thin wrapper around a trained MultiModalEncoder that exposes the
    intermediate spatial feature maps needed by SpatiallyConditionedDecoder.

    The weights are shared — this wrapper does NOT copy parameters.

    Returns:
        embedding   : [B, 128]    backbone embedding (h)
        projection  : [B,  64]    L2-normalised projection (z)
        bottleneck  : [B, 512, 2, 2]  layer4 spatial output (no avgpool)
        skip3       : [B, 256, 4, 4]  layer3 output
        skip2       : [B, 128, 8, 8]  layer2 output
        skip1       : [B,  64, 16, 16] layer1 output
    """

    def __init__(self, base_encoder: nn.Module):
        super().__init__()
        # Share all sub-modules from the existing encoder (no param copy)
        self.conv1    = base_encoder.conv1
        self.bn1      = base_encoder.bn1
        self.relu     = base_encoder.relu
        self.maxpool  = base_encoder.maxpool
        self.layer1   = base_encoder.layer1
        self.layer2   = base_encoder.layer2
        self.layer3   = base_encoder.layer3
        self.layer4   = base_encoder.layer4
        self.avgpool  = base_encoder.avgpool
        self.fc_embed = base_encoder.fc_embed
        self.proj_head = base_encoder.proj_head

    def forward(self, x: torch.Tensor):
        # Encoder stem
        x = self.conv1(x)     # [B, 64,  32, 32]  (input 64×64, stride=2)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)   # [B, 64,  16, 16]  (stride=2)

        skip1      = self.layer1(x)          # [B,  64, 16, 16]
        skip2      = self.layer2(skip1)      # [B, 128,  8,  8]
        skip3      = self.layer3(skip2)      # [B, 256,  4,  4]
        bottleneck = self.layer4(skip3)      # [B, 512,  2,  2]

        pooled    = self.avgpool(bottleneck) # [B, 512,  1,  1]
        pooled    = torch.flatten(pooled, 1) # [B, 512]

        embedding  = self.fc_embed(pooled)               # [B, 128]
        projection = F.normalize(
            self.proj_head(embedding), dim=1
        )                                                # [B,  64]

        return embedding, projection, bottleneck, skip3, skip2, skip1
