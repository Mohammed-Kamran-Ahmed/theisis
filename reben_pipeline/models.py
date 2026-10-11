"""
models.py - Dual-Encoder Architecture with CAHN/IERC & Multi-Label Classification
==================================================================================
Implements:
1. S1 SAR Encoder branch (2 bands: VV, VH).
2. S2 Optical Encoder branch (10 Sentinel-2 bands).
3. Projection heads for the contrastive embedding space (z_sar, z_opt).
4. Fused cross-modal representation and multi-label classifier heads (S1, S2, Fusion).

Change vs. previous version: the nan_to_num() after normalisation was removed. It
turned a diverged (NaN) forward pass into silent zeros; a NaN now propagates to the
loss, where the training loop raises a FloatingPointError.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm


def create_backbone(backbone_type: str = "resnet50", pretrained: bool = True, in_channels: int = 3):
    """Torchvision ResNet backbone (no fc) with a customised first convolution."""
    if backbone_type == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = 512
    elif backbone_type == "resnet50":
        model = tvm.resnet50(weights=tvm.ResNet50_Weights.DEFAULT if pretrained else None)
        feat_dim = 2048
    elif backbone_type == "resnet101":
        model = tvm.resnet101(weights=tvm.ResNet101_Weights.DEFAULT if pretrained else None)
        feat_dim = 2048
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}")

    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if in_channels < 3:
                new_conv.weight.data[:, :in_channels] = old_conv.weight.data[:, :in_channels]
            else:
                repeat_factor = (in_channels + 2) // 3
                repeated_w = old_conv.weight.data.repeat(1, repeat_factor, 1, 1)[:, :in_channels]
                new_conv.weight.data = repeated_w * (3.0 / in_channels)
        model.conv1 = new_conv

    stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
    encoder = nn.ModuleDict({
        "stem": stem,
        "layer1": model.layer1,
        "layer2": model.layer2,
        "layer3": model.layer3,
        "layer4": model.layer4,
        "pool": nn.AdaptiveAvgPool2d(1),
    })
    return encoder, feat_dim


class ModalityBranch(nn.Module):
    """Encoder branch for one modality (SAR or optical)."""

    def __init__(self, in_channels: int, backbone: str = "resnet50", proj_dim: int = 128,
                 pretrained: bool = True):
        super().__init__()
        self.encoder, self.feat_dim = create_backbone(
            backbone, pretrained=pretrained, in_channels=in_channels
        )
        self.projection_head = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        """Returns (feat (B, feat_dim), L2-normalised projection (B, proj_dim))."""
        x = self.encoder["stem"](x)
        x = self.encoder["layer1"](x)
        x = self.encoder["layer2"](x)
        x = self.encoder["layer3"](x)
        x = self.encoder["layer4"](x)
        feat = self.encoder["pool"](x).flatten(1)
        proj = self.projection_head(feat)
        return feat, F.normalize(proj, p=2, dim=1)


class CAHNIERCDualEncoder(nn.Module):
    """
    Dual encoder: SAR (B,2,H,W) -> f_sar, optical (B,10,H,W) -> f_opt, fused
    multi-label classifier, plus per-modality classifiers.
    """

    def __init__(
        self,
        sar_channels: int = 2,
        opt_channels: int = 10,
        num_classes: int = 19,
        backbone: str = "resnet50",
        proj_dim: int = 128,
        fusion_type: str = "concat",  # 'concat' or 'add'
        pretrained: bool = True,
    ):
        super().__init__()
        self.sar_branch = ModalityBranch(sar_channels, backbone=backbone, proj_dim=proj_dim,
                                         pretrained=pretrained)
        self.opt_branch = ModalityBranch(opt_channels, backbone=backbone, proj_dim=proj_dim,
                                         pretrained=pretrained)
        self.feat_dim = self.sar_branch.feat_dim
        self.proj_dim = proj_dim
        self.num_classes = num_classes
        self.fusion_type = fusion_type

        self.sar_classifier = nn.Linear(self.feat_dim, num_classes)
        self.opt_classifier = nn.Linear(self.feat_dim, num_classes)

        fusion_dim = self.feat_dim * 2 if fusion_type == "concat" else self.feat_dim
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, sar_img: torch.Tensor = None, opt_img: torch.Tensor = None):
        res = {}
        if sar_img is not None:
            feat_s, z_s = self.sar_branch(sar_img)
            res.update({"feat_sar": feat_s, "z_sar": z_s,
                        "logits_sar": self.sar_classifier(feat_s)})

        if opt_img is not None:
            feat_o, z_o = self.opt_branch(opt_img)
            res.update({"feat_opt": feat_o, "z_opt": z_o,
                        "logits_opt": self.opt_classifier(feat_o)})

        if sar_img is not None and opt_img is not None:
            res["z_fused"] = F.normalize((res["z_sar"] + res["z_opt"]) / 2.0, p=2, dim=1)
            if self.fusion_type == "concat":
                fused_feats = torch.cat([res["feat_sar"], res["feat_opt"]], dim=1)
            else:
                fused_feats = res["feat_sar"] + res["feat_opt"]
            res["logits_fused"] = self.fusion_classifier(fused_feats)

        return res