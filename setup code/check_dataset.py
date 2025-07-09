import os
from pathlib import Path


def check_dataset_structure():
    """Check the current directory structure and show what's available."""
    print("Checking dataset structure...")
    print("=" * 50)
    
    current_dir = Path(".")
    
    # List all directories
    directories = [d for d in current_dir.iterdir() if d.is_dir()]
    
    if not directories:
        print("No directories found in current location.")
        return
    
    print("Found directories:")
    for dir_path in directories:
        print(f"  📁 {dir_path.name}/")
        
        # Count files in each directory
        try:
            files = list(dir_path.glob("*"))
            image_files = [f for f in files if f.suffix.lower() in ['.png', '.jpg', '.jpeg']]
            
            if image_files:
                print(f"    └── {len(image_files)} image files")
                if len(image_files) <= 5:
                    for img_file in image_files:
                        print(f"        └── {img_file.name}")
                else:
                    print(f"        └── {image_files[0].name} ... {image_files[-1].name}")
            else:
                print(f"    └── {len(files)} files (no images)")
                
        except Exception as e:
            print(f"    └── Error reading directory: {e}")
    
    print("\n" + "=" * 50)
    
    # Check for specific DIV2K structure
    expected_dirs = [
        "DIV2K_train_HR",
        "DIV2K_valid_HR", 
        "DIV2K_train_LR_bicubic",
        "DIV2K_valid_LR_bicubic"
    ]
    
    print("Checking for expected DIV2K structure:")
    for dir_name in expected_dirs:
        dir_path = current_dir / dir_name
        if dir_path.exists():
            image_count = len(list(dir_path.glob("*.png"))) + len(list(dir_path.glob("*.jpg")))
            print(f"  ✅ {dir_name}/ - {image_count} images")
        else:
            print(f"  ❌ {dir_name}/ - Not found")
    
    print("\n" + "=" * 50)


def suggest_next_steps():
    """Suggest next steps based on current dataset structure."""
    current_dir = Path(".")
    
    # Check if we have the expected structure
    has_train_hr = (current_dir / "DIV2K_train_HR").exists()
    has_valid_hr = (current_dir / "DIV2K_valid_HR").exists()
    
    if has_train_hr and has_valid_hr:
        print("🎉 Perfect! You have both training and validation data.")
        print("\nYou can now run training with:")
        print("python train.py --hr_train_dir DIV2K_train_HR --hr_val_dir DIV2K_valid_HR")
        
    elif has_train_hr and not has_valid_hr:
        print("📊 You have training data but no validation data.")
        print("\nRun the dataset organization script to create train/validation split:")
        print("python download_dataset.py")
        
    else:
        print("⚠️ No training data found.")
        print("\nPlease download the dataset first:")
        print("python download_dataset.py")


def main():
    """Main function to check dataset structure."""
    print("SRGAN Dataset Structure Checker")
    print("=" * 50)
    
    check_dataset_structure()
    suggest_next_steps()


if __name__ == "__main__":
    main() 