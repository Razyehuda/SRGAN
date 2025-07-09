#!/usr/bin/env python3
"""
Demo script for interactive three-way comparison.
This script demonstrates how to use the updated test_v2_model.py
to display side-by-side comparisons of HR, LR, and SR images.
"""

import os
import subprocess
import sys

def main():
    print("=== SRGAN Interactive Comparison Demo ===\n")
    
    # Check if checkpoint exists
    checkpoint_path = "checkpoints_v2/best_model_finetune.pth"
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Please make sure you have a trained model checkpoint.")
        return
    
    print("This demo will show you how to use the interactive comparison feature.")
    print("The script will display three images side by side:")
    print("1. Original High-Resolution image")
    print("2. Low-Resolution input (4x downsampled)")
    print("3. SRGAN Super-Resolution output")
    print()
    
    # Example 1: Interactive validation test
    print("=== Example 1: Interactive Validation Test ===")
    print("This will process validation images and show each comparison interactively.")
    print("You can press Enter to continue to the next image.")
    print()
    
    print("Command to run:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode hr_test --input C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR --output validation_comparisons --max_size 512")
    print()
    
    # Example 2: Single image with display
    print("=== Example 2: Single Image with Display ===")
    print("This will process a single image and show the comparison.")
    print()
    print("Command to run:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode single --input path/to/your/image.jpg --output output/sr_image.jpg --compare")
    print()
    
    # Example 3: Save only (no display)
    print("=== Example 3: Save Only (No Display) ===")
    print("This will save comparison images without showing them interactively.")
    print()
    print("Command to run:")
    print(f"python test_v2_model.py --checkpoint {checkpoint_path} --mode hr_test --input C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR --output validation_comparisons --max_size 512 --no_display")
    print()
    
    print("=== Key Features ===")
    print("✓ Interactive display with plt.show()")
    print("✓ Press Enter to continue to next image")
    print("✓ Automatic saving of comparison images")
    print("✓ Option to disable display with --no_display")
    print("✓ Memory optimization with --max_size")
    print()
    
    # Ask if user wants to run the interactive demo
    print("Would you like to run the interactive validation test now? (y/n)")
    response = input().lower().strip()
    
    if response == 'y':
        # Run the interactive validation test
        cmd = [
            "python", "test_v2_model.py",
            "--checkpoint", checkpoint_path,
            "--mode", "hr_test",
            "--input", r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR",
            "--output", "validation_comparisons",
            "--max_size", "512"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        print("\nThe script will:")
        print("1. Load your trained model")
        print("2. Process validation images")
        print("3. Show each comparison interactively")
        print("4. Save comparison images to 'validation_comparisons' folder")
        print("5. Wait for you to press Enter between images")
        print()
        
        try:
            subprocess.run(cmd, check=True)
            print("✓ Interactive validation test completed successfully!")
            print("Check the 'validation_comparisons' directory for saved images.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error running validation test: {e}")
        except FileNotFoundError:
            print("❌ Python not found in PATH. Please run the command manually.")
    else:
        print("You can run the commands manually when ready.")
        print("\nTips:")
        print("- Use --no_display if you only want to save images without showing them")
        print("- Use --max_size 256 if you encounter memory issues")
        print("- The comparison images are automatically saved even when using interactive display")

if __name__ == "__main__":
    main() 