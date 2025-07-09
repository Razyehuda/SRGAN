import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):
    """Residual block with two 3x3 convolutions and batch normalization."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = F.relu(out)
        return out


class Generator(nn.Module):
    """SRGAN Generator network."""
    
    def __init__(self, num_residual_blocks=16, num_channels=3, base_channels=64):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(num_channels, base_channels, kernel_size=9, padding=4)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(base_channels))
        self.residual_layers = nn.Sequential(*res_blocks)
        
        # Post-residual convolution
        self.conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)
        
        # Upsampling layers
        self.upsampling = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.conv3 = nn.Conv2d(base_channels, num_channels, kernel_size=9, padding=4)
        
    def forward(self, x):
        out = F.relu(self.conv1(x))
        residual = out
        out = self.residual_layers(out)
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.upsampling(out)
        out = self.conv3(out)
        return out  # Remove tanh to allow proper residual learning


class Discriminator(nn.Module):
    """SRGAN Discriminator network."""
    
    def __init__(self, num_channels=3, base_channels=64):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, bn=True):
            block = [nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)]
            if bn:
                block.append(nn.BatchNorm2d(out_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*block)
        
        self.conv1 = nn.Conv2d(num_channels, base_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = discriminator_block(base_channels, base_channels, bn=False)
        self.conv3 = discriminator_block(base_channels, base_channels * 2)
        self.conv4 = discriminator_block(base_channels * 2, base_channels * 2)
        self.conv5 = discriminator_block(base_channels * 2, base_channels * 4)
        self.conv6 = discriminator_block(base_channels * 4, base_channels * 4)
        self.conv7 = discriminator_block(base_channels * 4, base_channels * 8)
        self.conv8 = discriminator_block(base_channels * 8, base_channels * 8)
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Dense layers
        self.dense1 = nn.Linear(base_channels * 8 * 4 * 4, 1024)
        self.dense2 = nn.Linear(1024, 1)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.2)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        
        # Use adaptive pooling to get fixed size
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        
        out = F.leaky_relu(self.dense1(out), 0.2)
        out = self.dense2(out)
        return torch.sigmoid(out)


class VGGFeatureExtractor(nn.Module):
    """VGG19 feature extractor for perceptual loss."""
    
    def __init__(self, feature_layer=35):
        super(VGGFeatureExtractor, self).__init__()
        vgg = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Convert from [-1, 1] to ImageNet normalized range [0, 1] -> ImageNet norm
        # First denormalize from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Then apply ImageNet normalization
        x = (x - torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)) / torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return self.features(x)


class SRGAN(nn.Module):
    """Complete SRGAN model."""
    
    def __init__(self, num_residual_blocks=16, num_channels=3, base_channels=64):
        super(SRGAN, self).__init__()
        self.generator = Generator(num_residual_blocks, num_channels, base_channels)
        self.discriminator = Discriminator(num_channels, base_channels)
        self.feature_extractor = VGGFeatureExtractor()
        
    def forward(self, x):
        return self.generator(x) 