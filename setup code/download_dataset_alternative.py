import os
import subprocess
import sys
import shutil
from pathlib import Path
import urllib.request
import zipfile


def download_div2k_from_official():
    """Download DIV2K dataset from official source."""
    print("Attempting to download DIV2K from official source...")
    
    # Official DIV2K URLs
    urls = {
        "train_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip",
        "valid_hr": "http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip"
    }
    
    data_dir = Path(".")
    
    for name, url in urls.items():
        try:
            print(f"Downloading {name}...")
            zip_path = data_dir / f"{name}.zip"
            
            # Download file
            urllib.request.urlretrieve(url, zip_path)
            
            # Extract
            print(f"Extracting {name}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            # Remove zip file
            zip_path.unlink()
            
            print(f"✓ {name} downloaded and extracted successfully!")
            
        except Exception as e:
            print(f"✗ Failed to download {name}: {e}")
            return False
    
    return True


def create_synthetic_dataset():
    """Create a synthetic dataset for testing purposes."""
    print("Creating synthetic dataset for testing...")
    
    import numpy as np
    from PIL import Image
    
    # Create directories
    train_dir = Path("DIV2K_train_HR")
    valid_dir = Path("DIV2K_valid_HR")
    
    train_dir.mkdir(exist_ok=True)
    valid_dir.mkdir(exist_ok=True)
    
    # Create synthetic images
    num_train = 50
    num_valid = 10
    
    print(f"Creating {num_train} training images...")
    for i in range(num_train):
        # Create a synthetic high-resolution image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some patterns to make it more realistic
        # Add gradients
        for c in range(3):
            img[:, :, c] += np.random.randint(0, 50, (256, 256))
        
        # Add some noise
        noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Save image
        pil_img = Image.fromarray(img)
        pil_img.save(train_dir / f"train_{i:04d}.png")
    
    print(f"Creating {num_valid} validation images...")
    for i in range(num_valid):
        # Create a synthetic high-resolution image
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        # Add some patterns to make it more realistic
        for c in range(3):
            img[:, :, c] += np.random.randint(0, 50, (256, 256))
        
        # Add some noise
        noise = np.random.randint(0, 30, (256, 256, 3), dtype=np.uint8)
        img = np.clip(img + noise, 0, 255).astype(np.uint8)
        
        # Save image
        pil_img = Image.fromarray(img)
        pil_img.save(valid_dir / f"valid_{i:04d}.png")
    
    print("✓ Synthetic dataset created successfully!")
    return True


def try_kaggle_alternative():
    """Try alternative Kaggle dataset sources."""
    print("Trying alternative Kaggle dataset sources...")
    
    try:
        import kagglehub
        
        # Try different DIV2K dataset sources
        alternative_sources = [
            "eugenesiow/div2k-dataset",
            "div2k/div2k-dataset",
            "div2k/div2k-train-hr"
        ]
        
        for source in alternative_sources:
            try:
                print(f"Trying source: {source}")
                path = kagglehub.dataset_download(source, path=".")
                print(f"✓ Successfully downloaded from {source}")
                return True
            except Exception as e:
                print(f"✗ Failed to download from {source}: {e}")
                continue
        
        return False
        
    except ImportError:
        print("kagglehub not available")
        return False


def main():
    """Main function to download or create dataset."""
    print("SRGAN Dataset Setup - Alternative Methods")
    print("=" * 50)
    
    # Try different methods
    methods = [
        ("Official DIV2K", download_div2k_from_official),
        ("Kaggle Alternative", try_kaggle_alternative),
        ("Synthetic Dataset", create_synthetic_dataset)
    ]
    
    for method_name, method_func in methods:
        print(f"\nTrying method: {method_name}")
        print("-" * 30)
        
        if method_func():
            print(f"\n🎉 Success with {method_name}!")
            print("\nYou can now run training with:")
            print("python train.py --hr_train_dir DIV2K_train_HR --hr_val_dir DIV2K_valid_HR")
            return
        else:
            print(f"✗ {method_name} failed")
    
    print("\n❌ All methods failed!")
    print("Please manually download the DIV2K dataset and place it in the current directory.")
    print("Expected structure:")
    print("  DIV2K_train_HR/  (training high-resolution images)")
    print("  DIV2K_valid_HR/  (validation high-resolution images)")


if __name__ == "__main__":
    main() 