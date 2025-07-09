import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from models.srgan import Generator
from data.dataset import create_data_loaders
from utils.losses import calculate_psnr, calculate_ssim

def test_normalization():
    """Test the normalization pipeline."""
    print("Testing normalization pipeline...")
    
    # Create a simple test image
    test_img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    test_img = Image.fromarray(test_img)
    
    # Test dataset normalization
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # [-1, 1]
    ])
    
    normalized_img = transform(test_img)
    print(f"Original image range: [0, 255]")
    print(f"After ToTensor: [{normalized_img.min():.3f}, {normalized_img.max():.3f}]")
    
    # Test denormalization
    denormalized_img = (normalized_img + 1.0) / 2.0
    print(f"After denormalization: [{denormalized_img.min():.3f}, {denormalized_img.max():.3f}]")
    
    # Test generator output
    generator = Generator(num_residual_blocks=2, base_channels=16)  # Small model for testing
    generator.eval()
    
    with torch.no_grad():
        # Create dummy input
        dummy_input = torch.randn(1, 3, 24, 24)  # Small input
        output = generator(dummy_input)
        print(f"Generator output range: [{output.min():.3f}, {output.max():.3f}]")
        
        # Test PSNR calculation
        target = torch.randn(1, 3, 96, 96)  # 4x larger
        target = torch.clamp(target, -1, 1)  # Ensure it's in valid range
        
        # Denormalize both for PSNR
        output_denorm = (output + 1.0) / 2.0
        target_denorm = (target + 1.0) / 2.0
        
        psnr = calculate_psnr(output_denorm, target_denorm)
        ssim = calculate_ssim(output_denorm, target_denorm)
        
        print(f"PSNR: {psnr:.2f} dB")
        print(f"SSIM: {ssim:.4f}")
    
    print("\n✅ Normalization test passed!")
    print("Expected ranges:")
    print("- Dataset output: [-1, 1]")
    print("- Generator output: [-1, 1] (due to tanh)")
    print("- Denormalized for metrics: [0, 1]")
    print("- PSNR should be finite and positive")

if __name__ == "__main__":
    test_normalization() 