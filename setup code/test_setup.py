import torch
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.srgan import Generator, Discriminator
from data.dataset import DIV2KDataset
from utils.losses import SRGANLoss, calculate_psnr, calculate_ssim


def test_models():
    """Test if models can be created and run forward pass."""
    print("Testing model creation and forward pass...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test Generator
    generator = Generator(num_residual_blocks=16, num_channels=3, base_channels=64).to(device)
    lr_input = torch.randn(2, 3, 24, 24).to(device)  # 2 images, 3 channels, 24x24
    sr_output = generator(lr_input)
    print(f"Generator input shape: {lr_input.shape}")
    print(f"Generator output shape: {sr_output.shape}")
    print("✓ Generator test passed!")
    
    # Test Discriminator
    discriminator = Discriminator(num_channels=3, base_channels=64).to(device)
    hr_input = torch.randn(2, 3, 96, 96).to(device)  # 2 images, 3 channels, 96x96
    d_output = discriminator(hr_input)
    print(f"Discriminator input shape: {hr_input.shape}")
    print(f"Discriminator output shape: {d_output.shape}")
    print("✓ Discriminator test passed!")
    
    return True


def test_losses():
    """Test if loss functions work correctly."""
    print("\nTesting loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    sr_imgs = torch.randn(2, 3, 96, 96).to(device)
    hr_imgs = torch.randn(2, 3, 96, 96).to(device)
    
    # Create proper fake outputs (between 0 and 1)
    fake_outputs = torch.sigmoid(torch.randn(2, 1)).to(device)
    
    # Test SRGAN loss
    loss_fn = SRGANLoss().to(device)
    total_loss, content_loss, perceptual_loss, adversarial_loss = loss_fn(sr_imgs, hr_imgs, fake_outputs)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Content loss: {content_loss.item():.4f}")
    print(f"Perceptual loss: {perceptual_loss.item():.4f}")
    print(f"Adversarial loss: {adversarial_loss.item():.4f}")
    print("✓ Loss functions test passed!")
    
    # Test metrics
    psnr = calculate_psnr(sr_imgs, hr_imgs)
    ssim = calculate_ssim(sr_imgs, hr_imgs)
    print(f"PSNR: {psnr.item():.2f}")
    print(f"SSIM: {ssim.item():.4f}")
    print("✓ Metrics test passed!")
    
    return True


def test_dataset():
    """Test if dataset can be created (if data exists)."""
    print("\nTesting dataset creation...")
    
    # Check current directory for dataset
    hr_train_dir = Path("DIV2K_train_HR")
    hr_val_dir = Path("DIV2K_valid_HR")
    
    if not hr_train_dir.exists():
        print("⚠ Dataset not found. Please run download_dataset.py first.")
        print("Creating dummy dataset for testing...")
        
        # Create dummy dataset
        dummy_dir = Path("dummy_data")
        dummy_dir.mkdir(exist_ok=True)
        
        # Create a dummy image
        import numpy as np
        from PIL import Image
        
        dummy_img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
        dummy_img = Image.fromarray(dummy_img)
        dummy_img.save(dummy_dir / "dummy.png")
        
        hr_train_dir = dummy_dir
        hr_val_dir = dummy_dir
    
    try:
        dataset = DIV2KDataset(
            hr_dir=str(hr_train_dir),
            patch_size=96,
            scale_factor=4,
            is_training=True
        )
        
        print(f"Dataset created successfully with {len(dataset)} samples")
        
        # Test data loading
        lr_img, hr_img = dataset[0]
        print(f"LR image shape: {lr_img.shape}")
        print(f"HR image shape: {hr_img.shape}")
        print("✓ Dataset test passed!")
        
        return True
        
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Running SRGAN setup tests...\n")
    
    tests_passed = 0
    total_tests = 3
    
    # Test models
    if test_models():
        tests_passed += 1
    
    # Test losses
    if test_losses():
        tests_passed += 1
    
    # Test dataset
    if test_dataset():
        tests_passed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your SRGAN setup is ready for training.")
        print("\nNext steps:")
        print("1. Download the dataset: python download_dataset.py")
        print("2. Start training: python train.py --hr_train_dir DIV2K_train_HR --hr_val_dir DIV2K_valid_HR")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    print("="*50)


if __name__ == "__main__":
    main() 