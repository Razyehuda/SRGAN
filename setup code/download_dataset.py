import os
import zipfile
import subprocess
import sys
import shutil
from pathlib import Path

# Set the cache directory as the source for DIV2K_train_HR
CACHE_DIV2K_PATH = Path(r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1")


def install_kagglehub():
    """Install kagglehub package if not already installed."""
    try:
        import kagglehub
        print("Kagglehub package already installed.")
    except ImportError:
        print("Installing kagglehub package...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
        print("Kagglehub package installed successfully.")


def download_div2k_dataset():
    """Download DIV2K dataset from Kaggle using kagglehub."""
    # No need to download again, just print the path
    print(f"Using existing dataset at: {CACHE_DIV2K_PATH}")
    return True


def create_train_val_split():
    """Create train/validation split from training data."""
    print("Creating train/validation split from cache directory...")
    
    # Use the cache directory as the source
    train_dir = CACHE_DIV2K_PATH / "DIV2K_train_HR"
    if not train_dir.exists():
        print(f"Error: {train_dir} not found!")
        return False
    print(f"Found training directory: {train_dir}")
    
    # Get all image files
    image_files = list(train_dir.glob("*.png")) + list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.jpeg"))
    if not image_files:
        print(f"Error: No image files found in {train_dir}")
        return False
    print(f"Found {len(image_files)} training images")
    image_files.sort()
    split_idx = int(0.8 * len(image_files))
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    
    # Create output directories in current working directory
    current_dir = Path(".")
    train_hr_dir = current_dir / "DIV2K_train_HR"
    val_hr_dir = current_dir / "DIV2K_valid_HR"
    train_hr_dir.mkdir(exist_ok=True)
    val_hr_dir.mkdir(exist_ok=True)
    
    # Copy files
    print("Copying training images...")
    for file_path in train_files:
        shutil.copy2(file_path, train_hr_dir / file_path.name)
    print("Copying validation images...")
    for file_path in val_files:
        shutil.copy2(file_path, val_hr_dir / file_path.name)
    print("Train/validation split created successfully!")
    return True


def organize_dataset():
    """Organize the downloaded dataset into the expected structure."""
    # Check current directory for dataset files
    current_dir = Path(".")
    
    # Expected structure after organization:
    # ./
    # ├── DIV2K_train_HR/
    # ├── DIV2K_train_LR_bicubic/
    # │   └── X4/
    # ├── DIV2K_valid_HR/
    # └── DIV2K_valid_LR_bicubic/
    #     └── X4/
    
    print("Organizing dataset structure...")
    
    # Check if directories exist
    hr_train_dir = current_dir / "DIV2K_train_HR"
    hr_valid_dir = current_dir / "DIV2K_valid_HR"
    lr_train_dir = current_dir / "DIV2K_train_LR_bicubic" / "X4"
    lr_valid_dir = current_dir / "DIV2K_valid_LR_bicubic" / "X4"
    
    if not hr_train_dir.exists():
        print("Error: DIV2K_train_HR directory not found!")
        print("Please check if the dataset was downloaded correctly.")
        return False
    
    if not hr_valid_dir.exists():
        print("Error: DIV2K_valid_HR directory not found!")
        print("Please check if the dataset was downloaded correctly.")
        return False
    
    # Create LR directories if they don't exist
    lr_train_dir.mkdir(parents=True, exist_ok=True)
    lr_valid_dir.mkdir(parents=True, exist_ok=True)
    
    print("Dataset structure organized successfully!")
    print(f"Training HR images: {len(list(hr_train_dir.glob('*.png')))}")
    print(f"Validation HR images: {len(list(hr_valid_dir.glob('*.png')))}")
    
    return True


def main():
    """Main function to download and setup DIV2K dataset."""
    print("Setting up DIV2K dataset for SRGAN training...")
    
    # Install kagglehub package
    install_kagglehub()
    
    # Download dataset
    if not download_div2k_dataset():
        return
    
    # Create train/validation split
    if not create_train_val_split():
        return
    
    # Organize dataset
    if not organize_dataset():
        return
    
    print("\nDataset setup completed successfully!")
    print("\nYou can now run the training script with:")
    print("python train.py --hr_train_dir DIV2K_train_HR --hr_val_dir DIV2K_valid_HR")


if __name__ == "__main__":
    main() 