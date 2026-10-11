"""
models/sar_encoder_spatial.py
================================
Same architecture as MultiModalEncoder in stage4_contrastive_learning.py
(identical layer names -> identical state_dict keys), but forward() returns
the intermediate spatial feature maps (before global average pooling)
instead of just the pooled embedding/projection.

This is a SEPARATE class, not a modification of your Stage 4 script. It
exists purely so we can load your ALREADY-TRAINED sar_encoder.pth weights
into something that exposes spatial structure for the new decoder --
your original encoder class and checkpoint are untouched and still work
exactly as before for everything else (Stage 5 embedding export, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class MultiModalEncoderSpatial(nn.Module):
    def __init__(self, in_channels, embed_dim=128, proj_dim=64, pretrained=True):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        self.fc_embed = nn.Linear(512, embed_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward_spatial(self, x):
        """Returns intermediate spatial feature maps for skip connections,
        plus the final pooled embedding for reference/sanity checks.

        For a 64x64 input (matching Stage 4's TARGET_SIZE), the spatial
        shapes are approximately:
            f1 (layer1): 64  x 16 x 16
            f2 (layer2): 128 x 8  x 8
            f3 (layer3): 256 x 4  x 4
            f4 (layer4): 512 x 2  x 2   <- bottleneck for the decoder
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        pooled = self.avgpool(f4)
        pooled = torch.flatten(pooled, 1)
        embedding = self.fc_embed(pooled)

        return f1, f2, f3, f4, embedding

    def forward(self, x):
        # Kept for compatibility if anything ever loads this class and
        # expects the original (embedding, projection) signature.
        f1, f2, f3, f4, embedding = self.forward_spatial(x)
        projection = self.proj_head(embedding)
        return embedding, F.normalize(projection, dim=1)


def load_frozen_sar_encoder(checkpoint_path, in_channels=5, embed_dim=128, proj_dim=64, device="cpu"):
    """Loads your already-trained sar_encoder.pth into the spatial-aware
    class and freezes it -- this encoder is NOT retrained here, only the
    new decoder is."""
    encoder = MultiModalEncoderSpatial(in_channels=in_channels, embed_dim=embed_dim,
                                        proj_dim=proj_dim, pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location=device)
    encoder.load_state_dict(state_dict)
    encoder.to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder