"""
models/pretrained_backbone.py
================================
Fix #1 for the train/test overfitting gap: initialize both encoder branches
from ImageNet-pretrained weights instead of random init.

Why this matters for you specifically:
  - Your dual ResNet-18 (11M+ params/branch) is currently trained from
    scratch (`pretrained=False`) on only 454 samples.
  - The optical branch is already RGB, so `pretrained=True` works directly.
  - The SAR branch has 5 input channels (VV, VH, VH_VV, entropy, RVI), so
    torchvision's stock `resnet18(weights=...)` can't be loaded as-is --
    conv1 expects 3 input channels. This module "inflates" the pretrained
    conv1 filters from 3 -> N channels by averaging across the RGB filters
    and replicating, which is the standard approach used in multispectral/
    SAR remote-sensing transfer learning.

Usage (see integration notes at the bottom of this file):

    from models.pretrained_backbone import build_pretrained_resnet18

    sar_backbone = build_pretrained_resnet18(in_channels=5)
    opt_backbone = build_pretrained_resnet18(in_channels=3)  # already RGB
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def _inflate_conv1(conv1: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Replace a pretrained 3-channel conv1 with an in_channels-channel conv1,
    preserving the pretrained filter statistics.

    Method: average the 3 pretrained input-channel filters together to get
    a single "grayscale-equivalent" filter per output channel, then tile
    that across all `in_channels` new input channels. This is the same
    trick used in torchvision's own flow/optical-flow multi-frame models
    and in most SAR/multispectral transfer-learning papers (e.g. inflating
    RGB weights for 4-band Sentinel-2 or 5-band SAR-feature stacks).
    """
    old_weight = conv1.weight.data  # shape: [out_ch, 3, kH, kW]
    out_ch, _, kH, kW = old_weight.shape

    # Mean across the original 3 input channels -> [out_ch, 1, kH, kW]
    mean_weight = old_weight.mean(dim=1, keepdim=True)

    # Tile to the new channel count -> [out_ch, in_channels, kH, kW]
    new_weight = mean_weight.repeat(1, in_channels, 1, 1)

    # Rescale so the effective activation magnitude stays roughly similar
    # (averaging N copies of a similar-magnitude signal would otherwise
    # inflate activations by ~sqrt(in_channels) once real data flows through)
    new_weight = new_weight * (3.0 / in_channels)

    new_conv1 = nn.Conv2d(
        in_channels, out_ch,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=(conv1.bias is not None),
    )
    new_conv1.weight.data = new_weight
    if conv1.bias is not None:
        new_conv1.bias.data = conv1.bias.data.clone()

    return new_conv1


def build_pretrained_resnet18(in_channels: int, freeze_stem: bool = False) -> nn.Module:
    """
    Returns a torchvision ResNet-18 with ImageNet weights, with conv1
    adapted to `in_channels` input channels.

    Args:
        in_channels: number of input channels (5 for your SAR stack, 3 for optical RGB).
        freeze_stem: if True, freezes conv1 + bn1 + layer1 (helps further
            when data is very limited -- keeps the most generic, most
            transferable low-level filters fixed and only fine-tunes
            layer2/3/4). Worth trying as a second experiment if the
            unfrozen version still overfits.

    Returns:
        nn.Module: a ResNet-18 backbone (final fc layer removed --
        use `.children()[:-2]` style slicing already handled: the returned
        module still has `.fc` present but you should ignore/replace it in
        your MultiModalEncoder, exactly as you already do today).
    """
    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    if in_channels != 3:
        backbone.conv1 = _inflate_conv1(backbone.conv1, in_channels)

    if freeze_stem:
        for module in [backbone.conv1, backbone.bn1, backbone.layer1]:
            for p in module.parameters():
                p.requires_grad = False

    return backbone


# ----------------------------------------------------------------------
# INTEGRATION NOTES for stage4_contrastive_learning.py
# ----------------------------------------------------------------------
# Your `MultiModalEncoder` almost certainly builds two `resnet18(...)` calls
# internally (one per modality) using torchvision, since the constructor
# signature you're using is:
#
#   MultiModalEncoder(in_channels=len(SAR_BAND_INDICES), embed_dim=128,
#                      proj_dim=64, pretrained=False)
#
# Two ways to wire this in, pick whichever matches how MultiModalEncoder
# is actually written:
#
# OPTION A -- if MultiModalEncoder already has a `pretrained` flag that
# just forwards to `resnet18(weights=...)` for a 3-channel input, that flag
# will currently CRASH or silently ignore pretrained weights for your
# 5-channel SAR branch (shape mismatch on conv1). Replace whatever it does
# internally for the SAR branch with:
#
#   from models.pretrained_backbone import build_pretrained_resnet18
#   self.sar_backbone = build_pretrained_resnet18(in_channels=in_channels)
#   self.opt_backbone = build_pretrained_resnet18(in_channels=3)
#
# then remove the final `.fc` the same way your code already does today
# (e.g. `nn.Sequential(*list(backbone.children())[:-2])` before your own
# pooling + projection head).
#
# OPTION B -- if you'd rather not touch MultiModalEncoder's internals,
# just build the two backbones here and monkey-patch them in after
# construction:
#
#   encoder = MultiModalEncoder(in_channels=5, embed_dim=128, proj_dim=64,
#                                pretrained=False)
#   encoder.sar_backbone = build_pretrained_resnet18(in_channels=5)
#   encoder.opt_backbone = build_pretrained_resnet18(in_channels=3)
#
# (adjust attribute names `sar_backbone`/`opt_backbone` to whatever your
# class actually calls them -- check the top of MultiModalEncoder.__init__)
#
# Either way, do NOT freeze the whole backbone -- you still want to fine-tune
# on SAR/optical statistics, just starting from useful filters instead of
# noise. Try `freeze_stem=True` only as a follow-up experiment if the
# unfrozen version still shows a large train/test gap.