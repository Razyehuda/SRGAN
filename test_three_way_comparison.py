#!/usr/bin/env python3
"""
Simple test script to demonstrate three-way comparison functionality.
This script shows how to use test_v2_model.py to create side-by-side comparisons
of: Original HR image, LR input (4x downsampled), and SR output.
"""

import os
import subprocess
import sys

def main():
    # Example usage of the updated test_v2_model.py
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints_v2/best_model_finetune.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please make sure you have a trained model checkpoint.")
        return
    
    # Example 1: Test with validation HR images (recommended)
    print("=== Example 1: Testing with validation HR images ===")
    print("This will create three-way comparisons for all validation images.")
    print("Command:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode hr_test --input C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR --output validation_comparisons --compare")
    
    # Example 2: Single image test
    print("\n=== Example 2: Single image test ===")
    print("This will test a single image and create a three-way comparison.")
    print("Command:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode single --input path/to/your/image.jpg --output output/sr_image.jpg --compare")
    
    # Example 3: Batch processing
    print("\n=== Example 3: Batch processing ===")
    print("This will process all images in a directory.")
    print("Command:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode batch --input path/to/input/directory --output path/to/output/directory")
    
    print("\n=== Key Features ===")
    print("✓ Three-way comparison: Original HR, LR input (4x downsampled), SR output")
    print("✓ Memory optimization with --max_size parameter")
    print("✓ GPU/CPU support with --device parameter")
    print("✓ Batch processing for multiple images")
    print("✓ Validation set evaluation with PSNR/SSIM metrics")
    
    print("\n=== Usage Tips ===")
    print("1. For best results, use 'hr_test' mode with validation HR images")
    print("2. Use --max_size 512 if you encounter memory issues")
    print("3. Use --device cpu if GPU memory is insufficient")
    print("4. The --compare flag creates side-by-side comparison images")
    
    # Ask if user wants to run the validation test
    print("\nWould you like to run the validation test now? (y/n)")
    response = input().lower().strip()
    
    if response == 'y':
        # Run the validation test
        cmd = [
            "python", "test_v2_model.py",
            "--checkpoint", checkpoint_path,
            "--mode", "hr_test",
            "--input", r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR",
            "--output", "validation_comparisons",
            "--max_size", "512"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
            print("✓ Validation test completed successfully!")
            print("Check the 'validation_comparisons' directory for results.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running validation test: {e}")
        except FileNotFoundError:
            print("❌ Python not found in PATH. Please run the command manually.")
    else:
        print("You can run the commands manually when ready.")

if __name__ == "__main__":
    main() 