import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Import the dataset
from data.dataset import create_data_loaders

def test_dataset_fix():
    """Test the dataset fix to ensure proper resolution differences."""
    
    # Default paths
    DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
    DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
    
    print("Testing dataset fix...")
    
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
    
    print(f"✓ Data loaders created successfully")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Test training data (should be cropped)
    train_lr, train_hr = next(iter(train_loader))
    print(f"\nTraining data shapes:")
    print(f"LR: {train_lr.shape}")
    print(f"HR: {train_hr.shape}")
    
    # Test validation data (should be full images)
    val_lr, val_hr = next(iter(val_loader))
    print(f"\nValidation data shapes:")
    print(f"LR: {val_lr.shape}")
    print(f"HR: {val_hr.shape}")
    
    # Check if validation images are properly downsampled
    lr_h, lr_w = val_lr.shape[2], val_lr.shape[3]
    hr_h, hr_w = val_hr.shape[2], val_hr.shape[3]
    
    print(f"\nValidation image dimensions:")
    print(f"LR: {lr_w}x{lr_h}")
    print(f"HR: {hr_w}x{hr_h}")
    print(f"Scale factor: {hr_w//lr_w}")
    
    if hr_w == lr_w * 4 and hr_h == lr_h * 4:
        print("✓ Validation images are correctly downsampled!")
    else:
        print("✗ Validation images are NOT correctly downsampled!")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Denormalize images
    train_lr_denorm = (train_lr + 1.0) / 2.0
    train_hr_denorm = (train_hr + 1.0) / 2.0
    val_lr_denorm = (val_lr + 1.0) / 2.0
    val_hr_denorm = (val_hr + 1.0) / 2.0
    
    # Convert to numpy
    train_lr_np = train_lr_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    train_hr_np = train_hr_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    val_lr_np = val_lr_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    val_hr_np = val_hr_denorm.squeeze(0).numpy().transpose(1, 2, 0)
    
    # Plot training images (cropped)
    axes[0, 0].imshow(train_lr_np, interpolation='nearest')
    axes[0, 0].set_title(f'Training LR ({train_lr_np.shape[1]}x{train_lr_np.shape[0]})')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(train_hr_np, interpolation='nearest')
    axes[0, 1].set_title(f'Training HR ({train_hr_np.shape[1]}x{train_hr_np.shape[0]})')
    axes[0, 1].axis('off')
    
    # Plot validation images (full)
    axes[1, 0].imshow(val_lr_np, interpolation='nearest')
    axes[1, 0].set_title(f'Validation LR ({val_lr_np.shape[1]}x{val_lr_np.shape[0]})')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(val_hr_np, interpolation='nearest')
    axes[1, 1].set_title(f'Validation HR ({val_hr_np.shape[1]}x{val_hr_np.shape[0]})')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('dataset_fix_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Test completed! Check 'dataset_fix_test.png' for visualization.")
    
    return True

if __name__ == "__main__":
    test_dataset_fix() 