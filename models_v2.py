# models_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseResidualBlock(nn.Module):
    """
    A single dense block with multiple convolutions and residual connections.
    Part of the RRDB structure.
    """
    def __init__(self, in_channels, growth_channels=32):
        super(DenseResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, growth_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(in_channels + growth_channels, growth_channels, 3, 1, 1)
        self.conv3 = nn.Conv2d(in_channels + 2 * growth_channels, growth_channels, 3, 1, 1)
        self.conv4 = nn.Conv2d(in_channels + 3 * growth_channels, growth_channels, 3, 1, 1)
        self.conv5 = nn.Conv2d(in_channels + 4 * growth_channels, in_channels, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Residual scaling for stability
        return x5 * 0.2 + x

class RRDB(nn.Module):
    """
    Residual-in-Residual Dense Block, the core of the ESRGAN generator.
    """
    def __init__(self, in_channels):
        super(RRDB, self).__init__()
        self.drb1 = DenseResidualBlock(in_channels)
        self.drb2 = DenseResidualBlock(in_channels)
        self.drb3 = DenseResidualBlock(in_channels)

    def forward(self, x):
        out = self.drb1(x)
        out = self.drb2(out)
        out = self.drb3(out)
        # Residual scaling for stability
        return out * 0.2 + x

class GeneratorV2(nn.Module):
    """
    ESRGAN-style Generator with RRDB blocks. No Batch Norm.
    """
    def __init__(self, num_residual_blocks=23, num_channels=3, base_channels=64):
        super(GeneratorV2, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(num_channels, base_channels, 3, 1, 1)
        
        # RRDB blocks
        rrdb_blocks = []
        for _ in range(num_residual_blocks):
            rrdb_blocks.append(RRDB(base_channels))
        self.rrdb_body = nn.Sequential(*rrdb_blocks)
        
        # Post-RRDB convolution
        self.conv_body = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        
        # Upsampling layers (PixelShuffle)
        self.upsampling = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Final convolutions for refinement and output
        self.conv_hr = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(base_channels, num_channels, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.rrdb_body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat  # Residual connection over the main body
        
        out = self.upsampling(feat)
        out = self.lrelu(self.conv_hr(out))
        out = self.conv_last(out)
        return out

class DiscriminatorV2(nn.Module):
    """
    SRGAN Discriminator, adapted for Relativistic GAN loss.
    No final sigmoid activation, as BCEWithLogitsLoss is used.
    """
    def __init__(self, num_channels=3, base_channels=64):
        super(DiscriminatorV2, self).__init__()
        
        def discriminator_block(in_channels, out_channels, stride=1, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*block)

        self.features = nn.Sequential(
            # input: 3 x 96 x 96
            nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # state: 64 x 96 x 96
            discriminator_block(base_channels, base_channels, stride=2), # 64 x 48 x 48
            discriminator_block(base_channels, base_channels * 2),       # 128 x 48 x 48
            discriminator_block(base_channels * 2, base_channels * 2, stride=2), # 128 x 24 x 24
            discriminator_block(base_channels * 2, base_channels * 4),         # 256 x 24 x 24
            discriminator_block(base_channels * 4, base_channels * 4, stride=2), # 256 x 12 x 12
            discriminator_block(base_channels * 4, base_channels * 8),         # 512 x 12 x 12
            discriminator_block(base_channels * 8, base_channels * 8, stride=2)  # 512 x 6 x 6
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(base_channels * 8, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(100, 1)
            # No sigmoid here
        )

    def forward(self, x):
        out = self.features(x)
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out