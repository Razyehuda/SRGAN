#!/usr/bin/env python3
"""
Quick Start Script for SRGAN Super-Resolution Project

This script demonstrates basic usage of the SRGAN models for image super-resolution.
"""

import os
import sys
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed."""
    try:
        import torch
        import torchvision
        import numpy
        import PIL
        import matplotlib
        import tqdm
        import cv2
        import skimage
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_cuda():
    """Check CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠ CUDA not available, will use CPU")
            return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def create_sample_image():
    """Create a sample image for testing."""
    try:
        import numpy as np
        from PIL import Image
        
        # Create a simple test image
        img_array = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # Save the image
        sample_path = "sample_test_image.png"
        img.save(sample_path)
        print(f"✓ Created sample test image: {sample_path}")
        return sample_path
    except Exception as e:
        print(f"❌ Error creating sample image: {e}")
        return None

def run_quick_test():
    """Run a quick test with a sample image."""
    print("\n" + "="*50)
    print("RUNNING QUICK TEST")
    print("="*50)
    
    # Check if we have a trained model
    checkpoint_path = "checkpoints_v2/best_model_finetune.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ No trained model found at {checkpoint_path}")
        print("Please train a model first or download pre-trained weights")
        return False
    
    # Create sample image
    sample_path = create_sample_image()
    if not sample_path:
        return False
    
    # Run test
    try:
        import subprocess
        cmd = [
            sys.executable, "test_v2_model.py",
            "--checkpoint", checkpoint_path,
            "--mode", "single",
            "--input", sample_path,
            "--output", "quick_test_output.png",
            "--compare",
            "--no_display"
        ]
        
        print("Running test...")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ Quick test completed successfully!")
            print("Output files:")
            print("  - quick_test_output.png (super-resolved image)")
            print("  - quick_test_output_comparison.png (comparison)")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def show_usage_examples():
    """Show usage examples."""
    print("\n" + "="*50)
    print("USAGE EXAMPLES")
    print("="*50)
    
    examples = [
        {
            "title": "Train Original SRGAN",
            "command": "python train.py --hr_train_dir path/to/DIV2K_train_HR --hr_val_dir path/to/DIV2K_valid_HR --batch_size 16 --num_epochs 100"
        },
        {
            "title": "Train SRGAN v2 (Pretrain)",
            "command": "python train_v2.py --mode pretrain --hr_train_dir path/to/DIV2K_train_HR --hr_val_dir path/to/DIV2K_valid_HR --batch_size 16 --num_epochs 50"
        },
        {
            "title": "Train SRGAN v2 (Finetune)",
            "command": "python train_v2.py --mode finetune --hr_train_dir path/to/DIV2K_train_HR --hr_val_dir path/to/DIV2K_valid_HR --batch_size 16 --num_epochs 200"
        },
        {
            "title": "Test Single Image",
            "command": "python test_v2_model.py --checkpoint checkpoints_v2/best_model_finetune.pth --mode single --input your_image.jpg --output sr_output.jpg --compare"
        },
        {
            "title": "Evaluate on Validation Set",
            "command": "python test_v2_model.py --checkpoint checkpoints_v2/best_model_finetune.pth --mode validation --output validation_results"
        },
        {
            "title": "Compute FID Score",
            "command": "python compute_fid_scores.py --checkpoint checkpoints_v2/best_model_finetune.pth --hr_dir path/to/validation_hr --output_dir fid_evaluation"
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}")
        print(f"   {example['command']}")

def main():
    parser = argparse.ArgumentParser(description="Quick Start for SRGAN Project")
    parser.add_argument("--test", action="store_true", help="Run quick test")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("--check", action="store_true", help="Check system setup")
    
    args = parser.parse_args()
    
    print("🚀 SRGAN Super-Resolution Project - Quick Start")
    print("="*50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check CUDA
    check_cuda()
    
    if args.check:
        print("\n✓ System setup check completed")
        return
    
    if args.test:
        run_quick_test()
        return
    
    if args.examples:
        show_usage_examples()
        return
    
    # Default behavior: show all
    print("\n" + "="*50)
    print("SYSTEM CHECK")
    print("="*50)
    
    # Run quick test
    if run_quick_test():
        print("\n🎉 Quick start completed successfully!")
    else:
        print("\n⚠ Quick test failed. Please check the setup.")
    
    # Show examples
    show_usage_examples()
    
    print("\n" + "="*50)
    print("NEXT STEPS")
    print("="*50)
    print("1. Set up your dataset (DIV2K recommended)")
    print("2. Train your model using the examples above")
    print("3. Test your model on images")
    print("4. Evaluate performance with PSNR/SSIM/FID")
    print("\nFor detailed documentation, see README.md")

if __name__ == "__main__":
    main() 