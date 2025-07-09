import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import the dataset and trainer
from data.dataset import create_data_loaders
from train import SRGANTrainer
import argparse

def debug_image_sizes():
    """Debug function to check image sizes and content."""
    
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
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot LR image
    axes[0].imshow(lr_np)
    axes[0].set_title(f'Low Resolution ({lr_np.shape[1]}x{lr_np.shape[0]})')
    axes[0].axis('off')
    
    # Plot HR image
    axes[1].imshow(hr_np)
    axes[1].set_title(f'High Resolution ({hr_np.shape[1]}x{hr_np.shape[0]})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('debug_image_sizes.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Check if the issue is in the dataset
    print(f"\nChecking dataset behavior:")
    print(f"Scale factor: 4")
    print(f"Expected LR size: {hr_np.shape[1]//4}x{hr_np.shape[0]//4}")
    print(f"Actual LR size: {lr_np.shape[1]}x{lr_np.shape[0]}")
    
    if lr_np.shape[1] == hr_np.shape[1] // 4 and lr_np.shape[0] == hr_np.shape[0] // 4:
        print("✓ LR image is correctly downsampled")
    else:
        print("✗ LR image is NOT correctly downsampled!")
    
    return lr_np, hr_np

if __name__ == "__main__":
    debug_image_sizes() 