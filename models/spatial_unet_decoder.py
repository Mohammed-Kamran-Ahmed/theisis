"""
models/spatial_unet_decoder.py
=================================
Decodes an optical-looking RGB image from the SAR encoder's SPATIAL
feature maps (not the pooled 64-dim vector the existing OpticalImageDecoder
uses). Skip connections from each encoder stage let the decoder place
things at approximately the right LOCATION, not just guess an average
color for the whole patch -- this is the actual fix for the "solid green
blob" problem.

Path (for a 64x64 input matching Stage 4's resolution):
    f4 (512, 2, 2)  -- bottleneck
    -> upsample to 4x4,  concat f3 (256, 4, 4)  -> conv
    -> upsample to 8x8,  concat f2 (128, 8, 8)  -> conv
    -> upsample to 16x16, concat f1 (64, 16, 16) -> conv
    -> upsample to 32x32 -> conv
    -> upsample to 64x64 -> final conv -> 3-channel RGB, Sigmoid

Output resolution is 64x64, matching the resolution the SAR encoder was
actually trained at -- not the 128x128 used by the older pooled-vector
decoder. This is an honest choice: upsampling further wouldn't add real
information, just interpolate.
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

    def forward(self, x):
        return self.block(x)


class SpatialUNetDecoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv1 = ConvBlock(512 + 256, 256)   # bottleneck upsampled + f3 skip

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv2 = ConvBlock(256 + 128, 128)   # + f2 skip

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv3 = ConvBlock(128 + 64, 64)     # + f1 skip

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv4 = ConvBlock(64, 32)

        self.up5 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.final = nn.Sequential(
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, f1, f2, f3, f4):
        # f4: (B,512,2,2) f3: (B,256,4,4) f2: (B,128,8,8) f1: (B,64,16,16)
        x = self.up1(f4)                              # -> (B,512,4,4)
        x = torch.cat([x, f3], dim=1)                  # -> (B,768,4,4)
        x = self.conv1(x)                               # -> (B,256,4,4)

        x = self.up2(x)                                 # -> (B,256,8,8)
        x = torch.cat([x, f2], dim=1)                   # -> (B,384,8,8)
        x = self.conv2(x)                                # -> (B,128,8,8)

        x = self.up3(x)                                  # -> (B,128,16,16)
        x = torch.cat([x, f1], dim=1)                    # -> (B,192,16,16)
        x = self.conv3(x)                                 # -> (B,64,16,16)

        x = self.up4(x)                                   # -> (B,64,32,32)
        x = self.conv4(x)                                  # -> (B,32,32,32)

        x = self.up5(x)                                     # -> (B,32,64,64)
        x = self.final(x)                                    # -> (B,3,64,64)
        return x
    