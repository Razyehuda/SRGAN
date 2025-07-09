import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import torch.serialization
import argparse

# Import the models and dataset
from models.srgan import Generator
from data.dataset import create_data_loaders

def test_model_output():
    """Test if the SRGAN model is generating high-resolution or low-resolution images."""
    
    print("="*60)
    print("TESTING SRGAN MODEL OUTPUT RESOLUTION")
    print("="*60)
    
    # Default paths
    DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
    DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        hr_train_dir=DEFAULT_TRAIN_HR,
        hr_val_dir=DEFAULT_VAL_HR,
        lr_train_dir=None,
        lr_val_dir=None,
        batch_size=1,
        patch_size=96,
        scale_factor=4,
        num_workers=0
    )
    
    # Get a sample
    lr_imgs, hr_imgs = next(iter(val_loader))
    
    print(f"Input LR shape: {lr_imgs.shape}")
    print(f"Target HR shape: {hr_imgs.shape}")
    
    # Create a generator model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(
        num_residual_blocks=16,
        num_channels=3,
        base_channels=64
    ).to(device)
    
    print(f"Generator created on device: {device}")
    
    # Test with random weights (untrained model)
    print("\n" + "="*40)
    print("TESTING UNTRAINED MODEL")
    print("="*40)
    
    lr_imgs = lr_imgs.to(device)
    hr_imgs = hr_imgs.to(device)
    
    with torch.no_grad():
        # Generate output
        sr_imgs = generator(lr_imgs)
        
        print(f"Generated SR shape: {sr_imgs.shape}")
        print(f"Expected SR shape: {hr_imgs.shape}")
        
        # Check if shapes match
        if sr_imgs.shape == hr_imgs.shape:
            print("✓ Model output has correct high-resolution shape!")
        else:
            print("✗ Model output has wrong shape!")
            print(f"Expected: {hr_imgs.shape}, Got: {sr_imgs.shape}")
        
        # Check scale factor
        lr_h, lr_w = lr_imgs.shape[2], lr_imgs.shape[3]
        sr_h, sr_w = sr_imgs.shape[2], sr_imgs.shape[3]
        hr_h, hr_w = hr_imgs.shape[2], hr_imgs.shape[3]
        
        print(f"\nResolution analysis:")
        print(f"LR: {lr_w}x{lr_h}")
        print(f"SR (generated): {sr_w}x{sr_h}")
        print(f"HR (target): {hr_w}x{hr_h}")
        print(f"Scale factor (SR/LR): {sr_w/lr_w:.1f}x")
        print(f"Expected scale factor: 4x")
        
        if sr_w/lr_w == 4 and sr_h/lr_h == 4:
            print("✓ Model is correctly upscaling by 4x!")
        else:
            print("✗ Model is NOT correctly upscaling!")
    
    # Test with a trained model if available
    print("\n" + "="*40)
    print("TESTING TRAINED MODEL (if available)")
    print("="*40)
    
    checkpoint_path = "checkpoints_best_model/best_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading trained model from {checkpoint_path}")
        # Allowlist argparse.Namespace for safe loading
        with torch.serialization.safe_globals([argparse.Namespace]):
            checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        generator.eval()
        
        with torch.no_grad():
            sr_imgs_trained = generator(lr_imgs)
            
            print(f"Trained model SR shape: {sr_imgs_trained.shape}")
            
            # Compare quality
            sr_imgs_denorm = (sr_imgs_trained + 1.0) / 2.0
            hr_imgs_denorm = (hr_imgs + 1.0) / 2.0
            lr_imgs_denorm = (lr_imgs + 1.0) / 2.0
            
            # Calculate PSNR
            mse = torch.mean((sr_imgs_denorm - hr_imgs_denorm) ** 2)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            
            print(f"PSNR (SR vs HR): {psnr.item():.2f} dB")
            
            # Create visualization
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            
            # Convert to numpy for plotting
            lr_np = lr_imgs_denorm.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            sr_np = sr_imgs_denorm.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            hr_np = hr_imgs_denorm.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Plot images
            axes[0].imshow(lr_np, interpolation='nearest')
            axes[0].set_title(f'Low Resolution\n({lr_np.shape[1]}x{lr_np.shape[0]})')
            axes[0].axis('off')
            
            axes[1].imshow(sr_np, interpolation='nearest')
            axes[1].set_title(f'Super Resolved\n({sr_np.shape[1]}x{sr_np.shape[0]})\nPSNR: {psnr.item():.2f}dB')
            axes[1].axis('off')
            
            axes[2].imshow(hr_np, interpolation='nearest')
            axes[2].set_title(f'High Resolution\n({hr_np.shape[1]}x{hr_np.shape[0]})')
            axes[2].axis('off')
            
            # Difference image
            diff = np.abs(sr_np - hr_np)
            axes[3].imshow(diff, cmap='hot', interpolation='nearest')
            axes[3].set_title(f'Difference\nMax: {diff.max():.3f}')
            axes[3].axis('off')
            
            plt.tight_layout()
            plt.savefig('model_output_test.png', dpi=150, bbox_inches='tight')
            plt.show()
            
            print(f"\nVisualization saved as 'model_output_test.png'")
            
    else:
        print("No trained model found. Run training first.")
    
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)
    print("If the model output shape matches the target HR shape (4x larger than LR),")
    print("then the model architecture is correct and should learn to generate HR images.")
    print("The quality will improve with training.")
    print("="*60)

if __name__ == "__main__":
    test_model_output() 