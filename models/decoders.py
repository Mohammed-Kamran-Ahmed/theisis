import torch
import torch.nn as nn

class PhysicalFeatureDecoder(nn.Module):
    def __init__(self, latent_dim=64, num_features=5):
        super(PhysicalFeatureDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_features)
        )

    def forward(self, z_sar):
        return self.decoder(z_sar)


class OpticalImageDecoder(nn.Module):
    def __init__(self, latent_dim=64, out_channels=3):
        super(OpticalImageDecoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 16 * 16),
            nn.BatchNorm1d(256 * 16 * 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.deconv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout2d(0.2),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, z_sar):
        x = self.fc(z_sar)
        x = x.view(-1, 256, 16, 16)
        return self.deconv_blocks(x)