import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from data.dataset import create_data_loaders

def test_resolution_downsampling():
    """Test that the dataset correctly downsamples images by 4x."""
    
    # Default paths
    DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
    DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
    
    print("Testing 4x resolution downsampling...")
    print("=" * 50)
    
    # Create data loaders with small batch size for testing
    train_loader, val_loader = create_data_loaders(
        hr_train_dir=DEFAULT_TRAIN_HR,
        hr_val_dir=DEFAULT_VAL_HR,
        batch_size=2,
        patch_size=128,
        scale_factor=4,
        num_workers=0  # Use 0 for debugging
    )
    
    print(f"Number of training batches: {len(train_loader)}")
    print(f"Number of validation batches: {len(val_loader)}")
    
    # Test training data
    print("\n--- Testing Training Data ---")
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(train_loader):
        if batch_idx >= 3:  # Test first 3 batches
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  LR shape: {lr_imgs.shape}")
        print(f"  HR shape: {hr_imgs.shape}")
        
        # Check that LR is exactly 4x smaller than HR
        lr_h, lr_w = lr_imgs.shape[2], lr_imgs.shape[3]
        hr_h, hr_w = hr_imgs.shape[2], hr_imgs.shape[3]
        
        expected_lr_h = hr_h // 4
        expected_lr_w = hr_w // 4
        
        print(f"  LR dimensions: {lr_w} x {lr_h}")
        print(f"  HR dimensions: {hr_w} x {hr_h}")
        print(f"  Expected LR dimensions: {expected_lr_w} x {expected_lr_h}")
        
        if lr_h == expected_lr_h and lr_w == expected_lr_w:
            print(f"  ✓ 4x downsampling correct!")
        else:
            print(f"  ✗ 4x downsampling incorrect!")
            print(f"     Expected: {expected_lr_w} x {expected_lr_h}, Got: {lr_w} x {lr_h}")
    
    # Test validation data
    print("\n--- Testing Validation Data ---")
    for batch_idx, (lr_imgs, hr_imgs) in enumerate(val_loader):
        if batch_idx >= 3:  # Test first 3 batches
            break
            
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  LR shape: {lr_imgs.shape}")
        print(f"  HR shape: {hr_imgs.shape}")
        
        # Check that LR is exactly 4x smaller than HR
        lr_h, lr_w = lr_imgs.shape[2], lr_imgs.shape[3]
        hr_h, hr_w = hr_imgs.shape[2], hr_imgs.shape[3]
        
        expected_lr_h = hr_h // 4
        expected_lr_w = hr_w // 4
        
        print(f"  LR dimensions: {lr_w} x {lr_h}")
        print(f"  HR dimensions: {hr_w} x {hr_h}")
        print(f"  Expected LR dimensions: {expected_lr_w} x {expected_lr_h}")
        
        if lr_h == expected_lr_h and lr_w == expected_lr_w:
            print(f"  ✓ 4x downsampling correct!")
        else:
            print(f"  ✗ 4x downsampling incorrect!")
            print(f"     Expected: {expected_lr_w} x {expected_lr_h}, Got: {lr_w} x {lr_h}")
    
    # Visual test - save sample images
    print("\n--- Visual Test ---")
    print("Saving sample images for visual verification...")
    
    # Get a sample from training data
    lr_imgs, hr_imgs = next(iter(train_loader))
    
    # Convert from [-1, 1] to [0, 1] for visualization
    lr_imgs = (lr_imgs + 1.0) / 2.0
    hr_imgs = (hr_imgs + 1.0) / 2.0
    
    # Convert to numpy and transpose for matplotlib
    lr_np = lr_imgs[0].numpy().transpose(1, 2, 0)
    hr_np = hr_imgs[0].numpy().transpose(1, 2, 0)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(lr_np)
    axes[0].set_title(f'Low Resolution ({lr_np.shape[1]}x{lr_np.shape[0]})')
    axes[0].axis('off')
    
    axes[1].imshow(hr_np)
    axes[1].set_title(f'High Resolution ({hr_np.shape[1]}x{hr_np.shape[0]})')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('resolution_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Sample images saved as 'resolution_test.png'")
    
    # Calculate and display the actual scale factor
    actual_scale_h = hr_np.shape[0] / lr_np.shape[0]
    actual_scale_w = hr_np.shape[1] / lr_np.shape[1]
    
    print(f"\nActual scale factors:")
    print(f"  Height: {actual_scale_h:.1f}x")
    print(f"  Width: {actual_scale_w:.1f}x")
    
    if abs(actual_scale_h - 4.0) < 0.1 and abs(actual_scale_w - 4.0) < 0.1:
        print("✓ Scale factor is approximately 4x as expected!")
    else:
        print("✗ Scale factor is not 4x!")
    
    print("\n" + "=" * 50)
    print("Resolution downsampling test completed!")

if __name__ == '__main__':
    test_resolution_downsampling() 