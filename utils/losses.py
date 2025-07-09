import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ContentLoss(nn.Module):
    """Content loss using MSE between generated and target images."""
    
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, sr, hr):
        return self.mse_loss(sr, hr)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG19 features."""
    
    def __init__(self, feature_layer=35):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features.children())[:feature_layer])
        
        # Freeze VGG parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
        self.mse_loss = nn.MSELoss()
    
    def forward(self, sr, hr):
        # Normalize inputs for VGG
        sr_normalized = self._normalize_for_vgg(sr)
        hr_normalized = self._normalize_for_vgg(hr)
        
        # Extract features
        sr_features = self.feature_extractor(sr_normalized)
        hr_features = self.feature_extractor(hr_normalized)
        
        return self.mse_loss(sr_features, hr_features)
    
    def _normalize_for_vgg(self, x):
        """Normalize input for VGG network."""
        # First denormalize from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Then apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
        return (x - mean) / std


class AdversarialLoss(nn.Module):
    """Adversarial loss for generator training."""
    
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, fake_outputs):
        # Create target labels (1 for real images)
        batch_size = fake_outputs.size(0)
        real_labels = torch.ones(batch_size, 1).to(fake_outputs.device)
        
        return self.bce_loss(fake_outputs, real_labels)


class DiscriminatorLoss(nn.Module):
    """Loss for discriminator training."""
    
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
    
    def forward(self, real_outputs, fake_outputs):
        batch_size = real_outputs.size(0)
        
        # Create target labels
        real_labels = torch.ones(batch_size, 1).to(real_outputs.device)
        fake_labels = torch.zeros(batch_size, 1).to(fake_outputs.device)
        
        # Calculate losses
        real_loss = self.bce_loss(real_outputs, real_labels)
        fake_loss = self.bce_loss(fake_outputs, fake_labels)
        
        return real_loss + fake_loss


class SRGANLoss(nn.Module):
    """Combined loss for SRGAN training."""
    
    def __init__(self, content_weight=1.0, perceptual_weight=1.0, adversarial_weight=1e-3):
        super(SRGANLoss, self).__init__()
        self.content_loss = ContentLoss()
        self.perceptual_loss = PerceptualLoss()
        self.adversarial_loss = AdversarialLoss()
        
        self.content_weight = content_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
    
    def forward(self, sr, hr, fake_outputs=None):
        # Content loss
        content_loss = self.content_loss(sr, hr)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(sr, hr)
        
        # Adversarial loss (if provided)
        if fake_outputs is not None:
            adv_loss = self.adversarial_loss(fake_outputs)
        else:
            adv_loss = torch.tensor(0.0).to(sr.device)
        
        # Combined loss
        total_loss = (self.content_weight * content_loss + 
                     self.perceptual_weight * perceptual_loss + 
                     self.adversarial_weight * adv_loss)
        
        return total_loss, content_loss, perceptual_loss, adv_loss


def calculate_psnr(sr, hr, max_val=1.0):
    """Calculate PSNR between super-resolved and high-resolution images.
    
    Args:
        sr: Super-resolved image tensor (should be in [0, 1] range)
        hr: High-resolution image tensor (should be in [0, 1] range)
        max_val: Maximum pixel value (1.0 for normalized images)
    """
    # Ensure inputs are in the correct range
    sr = torch.clamp(sr, 0.0, 1.0)
    hr = torch.clamp(hr, 0.0, 1.0)
    
    mse = torch.mean((sr - hr) ** 2)
    if mse == 0:
        return float('inf')
    
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr


def calculate_ssim(sr, hr, window_size=11, size_average=True):
    """Calculate SSIM between super-resolved and high-resolution images."""
    def gaussian_window(size, sigma):
        gauss = torch.Tensor([torch.exp(torch.tensor(-(x - size//2)**2/float(2*sigma**2))) for x in range(size)])
        return gauss/gauss.sum()
    
    def create_window(window_size, channel):
        _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def _ssim(img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)
    
    (_, channel, _, _) = sr.size()
    window = create_window(window_size, channel).to(sr.device)
    
    return _ssim(sr, hr, window, window_size, channel, size_average) 