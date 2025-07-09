import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import the dataset
from data.dataset import create_data_loaders

def check_resolution_differences():
    """Check and visualize the resolution differences between LR and HR images."""
    
    # Default paths
    DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
    DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        hr_train_dir=DEFAULT_TRAIN_HR,
        hr_val_dir=DEFAULT_VAL_HR,
        lr_train_dir=None,  # Will generate LR by downsampling
        lr_val_dir=None,
        batch_size=1,
        patch_size=96,
        scale_factor=4,
        num_workers=0
    )
    
    print("Data loader created successfully!")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    
    # Get a batch
    lr_imgs, hr_imgs = next(iter(val_loader))
    
    print(f"\nImage tensor shapes:")
    print(f"LR images shape: {lr_imgs.shape}")
    print(f"HR images shape: {hr_imgs.shape}")
    
    # Denormalize for visualization
    lr_imgs_denorm = (lr_imgs + 1.0) / 2.0
    hr_imgs_denorm = (hr_imgs + 1.0) / 2.0
    
    # Convert to numpy
    lr_np = lr_imgs_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    hr_np = hr_imgs_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    
    print(f"\nNumpy array shapes after denormalization:")
    print(f"LR numpy shape: {lr_np.shape}")
    print(f"HR numpy shape: {hr_np.shape}")
    
    # Check pixel value ranges
    print(f"\nPixel value ranges:")
    print(f"LR min: {lr_np.min():.3f}, max: {lr_np.max():.3f}")
    print(f"HR min: {hr_np.min():.3f}, max: {hr_np.max():.3f}")
    
    # Verify downsampling
    print(f"\nVerifying downsampling:")
    print(f"Scale factor: 4")
    print(f"Expected LR size: {hr_np.shape[1]//4}x{hr_np.shape[0]//4}")
    print(f"Actual LR size: {lr_np.shape[1]}x{lr_np.shape[0]}")
    
    if lr_np.shape[1] == hr_np.shape[1] // 4 and lr_np.shape[0] == hr_np.shape[0] // 4:
        print("✓ LR image is correctly downsampled")
    else:
        print("✗ LR image is NOT correctly downsampled!")
    
    # Create a better visualization that shows the actual size differences
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Row 1: Show images at their actual sizes
    # Low resolution
    axes[0, 0].imshow(lr_np, interpolation='nearest')
    axes[0, 0].set_title(f'Low Resolution\n({lr_np.shape[1]}x{lr_np.shape[0]} pixels)')
    axes[0, 0].axis('off')
    
    # High resolution
    axes[0, 1].imshow(hr_np, interpolation='nearest')
    axes[0, 1].set_title(f'High Resolution\n({hr_np.shape[1]}x{hr_np.shape[0]} pixels)')
    axes[0, 1].axis('off')
    
    # Upscaled LR for comparison (what the generator should produce)
    from PIL import Image
    lr_pil = Image.fromarray((lr_np * 255).astype(np.uint8))
    lr_upscaled = lr_pil.resize((hr_np.shape[1], hr_np.shape[0]), Image.BICUBIC)
    lr_upscaled_np = np.array(lr_upscaled) / 255.0
    
    axes[0, 2].imshow(lr_upscaled_np, interpolation='nearest')
    axes[0, 2].set_title(f'LR Upscaled (Bicubic)\n({lr_upscaled_np.shape[1]}x{lr_upscaled_np.shape[0]} pixels)')
    axes[0, 2].axis('off')
    
    # Row 2: Show images at the same display size for comparison
    # Low resolution (upscaled for display)
    axes[1, 0].imshow(lr_np, interpolation='bilinear', extent=[0, hr_np.shape[1], 0, hr_np.shape[0]])
    axes[1, 0].set_title('Low Resolution (Display Size)')
    axes[1, 0].axis('off')
    
    # High resolution
    axes[1, 1].imshow(hr_np, interpolation='nearest')
    axes[1, 1].set_title('High Resolution (Display Size)')
    axes[1, 1].axis('off')
    
    # Difference image
    diff = np.abs(hr_np - lr_upscaled_np)
    axes[1, 2].imshow(diff, cmap='hot', interpolation='nearest')
    axes[1, 2].set_title(f'Difference (HR - Upscaled LR)\nMax diff: {diff.max():.3f}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('resolution_differences.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print(f"\n" + "="*50)
    print("RESOLUTION DIFFERENCES SUMMARY")
    print("="*50)
    print(f"Low Resolution: {lr_np.shape[1]}x{lr_np.shape[0]} pixels")
    print(f"High Resolution: {hr_np.shape[1]}x{hr_np.shape[0]} pixels")
    print(f"Scale Factor: {hr_np.shape[1]//lr_np.shape[1]}")
    print(f"Total pixels LR: {lr_np.shape[1] * lr_np.shape[0]}")
    print(f"Total pixels HR: {hr_np.shape[1] * hr_np.shape[0]}")
    print(f"Pixel ratio: {(hr_np.shape[1] * hr_np.shape[0]) / (lr_np.shape[1] * lr_np.shape[0]):.1f}x")
    print("="*50)
    
    return lr_np, hr_np

if __name__ == "__main__":
    check_resolution_differences() 