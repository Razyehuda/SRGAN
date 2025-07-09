# losses_v2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from utils.losses import calculate_psnr, calculate_ssim # Reuse your existing metrics

class ContentLossV2(nn.Module):
    """Content loss using L1 loss, as recommended by ESRGAN."""
    def __init__(self):
        super(ContentLossV2, self).__init__()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, sr, hr):
        return self.l1_loss(sr, hr)

class PerceptualLossV2(nn.Module):
    """Perceptual loss using VGG19 features from before activation."""
    def __init__(self, feature_layer=34): # Layer 34 is the conv layer before the final ReLU
        super(PerceptualLossV2, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer]).eval()
        
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.l1_loss = nn.L1Loss()
    
    def forward(self, sr, hr):
        sr_features = self.feature_extractor(self._normalize_for_vgg(sr))
        hr_features = self.feature_extractor(self._normalize_for_vgg(hr))
        return self.l1_loss(sr_features, hr_features)
    
    def _normalize_for_vgg(self, x):
        x = (x + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

class AdversarialLossV2(nn.Module):
    """Relativistic Average Adversarial loss for generator."""
    def __init__(self):
        super(AdversarialLossV2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_outputs, fake_outputs):
        real_labels = torch.ones_like(real_outputs)
        fake_labels = torch.zeros_like(fake_outputs)
        
        loss_real = self.bce_loss(real_outputs - torch.mean(fake_outputs), fake_labels)
        loss_fake = self.bce_loss(fake_outputs - torch.mean(real_outputs), real_labels)
        
        return (loss_real + loss_fake) / 2

class DiscriminatorLossV2(nn.Module):
    """Relativistic Average Adversarial loss for discriminator."""
    def __init__(self):
        super(DiscriminatorLossV2, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, real_outputs, fake_outputs):
        real_labels = torch.ones_like(real_outputs)
        fake_labels = torch.zeros_like(fake_outputs)
        
        real_loss = self.bce_loss(real_outputs - torch.mean(fake_outputs), real_labels)
        fake_loss = self.bce_loss(fake_outputs - torch.mean(real_outputs), fake_labels)
        
        return (real_loss + fake_loss) / 2

class SRGANLossV2(nn.Module):
    """Combined loss for ESRGAN fine-tuning."""
    def __init__(self, content_weight=0.01, perceptual_weight=1.0, adversarial_weight=0.005):
        super(SRGANLossV2, self).__init__()
        self.content_loss = ContentLossV2()
        self.perceptual_loss = PerceptualLossV2()
        self.adversarial_loss = AdversarialLossV2()
        
        self.content_weight = content_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
    
    def forward(self, sr, hr, real_outputs, fake_outputs):
        content_loss_val = self.content_loss(sr, hr)
        perceptual_loss_val = self.perceptual_loss(sr, hr)
        adv_loss_val = self.adversarial_loss(real_outputs, fake_outputs)
        
        total_loss = (self.content_weight * content_loss_val + 
                      self.perceptual_weight * perceptual_loss_val + 
                      self.adversarial_weight * adv_loss_val)
        
        return total_loss, content_loss_val, perceptual_loss_val, adv_loss_val