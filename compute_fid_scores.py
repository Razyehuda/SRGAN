#!/usr/bin/env python3
"""
Script to compute FID scores for SRGAN models.
This script generates super-resolution images from your trained model
and computes FID scores between the generated images and real HR images.
"""

import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import subprocess
import sys
from tqdm import tqdm
import glob
import shutil
import tempfile

from models_v2 import GeneratorV2
from models.srgan import Generator
from data.dataset import create_data_loaders

# Default paths
DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"

def load_model(model_type, checkpoint_path, device, num_residual_blocks=23, base_channels=64):
    """Load the trained SRGAN generator."""
    if model_type == 'v2':
        # Load V2 model
        generator = GeneratorV2(
            num_residual_blocks=num_residual_blocks,
            num_channels=3,
            base_channels=base_channels
        ).to(device)
    else:
        # Load original model
        generator = Generator(
            num_residual_blocks=16,
            num_channels=3,
            base_channels=64
        ).to(device)
    
    # Allowlist argparse.Namespace for legacy checkpoints
    if model_type == 'original':
        torch.serialization.add_safe_globals([argparse.Namespace])
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Check if checkpoint contains generator state dict
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✓ Loaded generator from checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        # Assume it's just the generator state dict
        generator.load_state_dict(checkpoint)
        print(f"✓ Loaded generator state dict from: {checkpoint_path}")
    
    generator.eval()
    return generator

def preprocess_image(image_path, device, max_size=None):
    """Preprocess input image for the model."""
    # Load image
    img = Image.open(image_path).convert('RGB')
    
    # Resize if max_size is specified (for memory optimization)
    if max_size is not None:
        w, h = img.size
        if w > max_size or h > max_size:
            # Maintain aspect ratio
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])
    
    # Apply transform and add batch dimension
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    return img_tensor, img

def postprocess_image(output_tensor):
    """Convert model output back to PIL image."""
    # Denormalize from [-1, 1] to [0, 1]
    output_tensor = (output_tensor + 1.0) / 2.0
    
    # Clamp to valid range
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    
    # Convert to PIL image
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)
    
    return output_img

def create_lr_from_hr(hr_path, scale_factor=4):
    """Create a low-resolution image from high-resolution for testing."""
    hr_img = Image.open(hr_path)
    w, h = hr_img.size
    
    # Create LR by downsampling
    lr_w, lr_h = w // scale_factor, h // scale_factor
    lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    
    return lr_img

def generate_sr_images(model, hr_dir, output_dir, device, max_size=None, scale_factor=4, model_type='v2'):
    """Generate SR images from HR images by creating LR versions and upscaling."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all HR image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    hr_files = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(image_extensions)]
    
    if not hr_files:
        print(f"No image files found in {hr_dir}")
        return False
    
    print(f"Generating SR images for {len(hr_files)} HR images...")
    
    for filename in tqdm(hr_files, desc="Generating SR images"):
        hr_path = os.path.join(hr_dir, filename)
        
        # Create LR image
        lr_img = create_lr_from_hr(hr_path, scale_factor)
        
        # Save LR image temporarily
        lr_temp_path = os.path.join(output_dir, f"lr_temp_{filename}")
        lr_img.save(lr_temp_path)
        
        # Process with model
        sr_path = os.path.join(output_dir, f"sr_{filename}")
        try:
            # Preprocess
            input_tensor, _ = preprocess_image(lr_temp_path, device, max_size)
            
            # Inference
            with torch.no_grad():
                output_tensor = model(input_tensor)
                output_tensor = output_tensor.clamp(-1.0, 1.0)
            
            # Postprocess
            output_img = postprocess_image(output_tensor)
            
            # Save result
            output_img.save(sr_path)
            
            # Clean up temp file
            os.remove(lr_temp_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            if os.path.exists(lr_temp_path):
                os.remove(lr_temp_path)
    
    print(f"✓ SR images generated in: {output_dir}")
    return True

def install_pytorch_fid():
    """Install pytorch-fid if not already installed."""
    try:
        import pytorch_fid
        print("✓ pytorch-fid is already installed")
        return True
    except ImportError:
        print("Installing pytorch-fid...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-fid"])
            print("✓ pytorch-fid installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install pytorch-fid: {e}")
            return False

def compute_fid_score(sr_dir, hr_dir):
    """Compute FID score between SR and HR images."""
    try:
        # Import pytorch-fid
        from pytorch_fid import fid_score
        
        print(f"Computing FID score...")
        print(f"SR images: {sr_dir}")
        print(f"HR images: {hr_dir}")
        
        # Count images
        sr_images = glob.glob(os.path.join(sr_dir, "*.png")) + glob.glob(os.path.join(sr_dir, "*.jpg"))
        hr_images = glob.glob(os.path.join(hr_dir, "*.png")) + glob.glob(os.path.join(hr_dir, "*.jpg"))
        
        print(f"Found {len(sr_images)} SR images and {len(hr_images)} HR images")
        
        if len(sr_images) == 0 or len(hr_images) == 0:
            print("❌ No images found in one or both directories")
            return None
        
        # Create temporary directories with resized images
        import tempfile
        import shutil
        
        with tempfile.TemporaryDirectory() as temp_sr_dir, tempfile.TemporaryDirectory() as temp_hr_dir:
            print("Resizing images to uniform size for FID computation...")
            
            # Resize SR images
            for i, img_path in enumerate(sr_images):
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((256, 256), Image.BICUBIC)  # Standard size for FID
                temp_path = os.path.join(temp_sr_dir, f"sr_{i:04d}.png")
                img_resized.save(temp_path)
            
            # Resize HR images
            for i, img_path in enumerate(hr_images):
                img = Image.open(img_path).convert('RGB')
                img_resized = img.resize((256, 256), Image.BICUBIC)  # Standard size for FID
                temp_path = os.path.join(temp_hr_dir, f"hr_{i:04d}.png")
                img_resized.save(temp_path)
            
            print(f"Resized {len(sr_images)} SR images and {len(hr_images)} HR images to 256x256")
            
            # Compute FID with resized images
            fid = fid_score.calculate_fid_given_paths(
                [temp_sr_dir, temp_hr_dir], 
                batch_size=50, 
                device='cuda' if torch.cuda.is_available() else 'cpu',
                dims=2048  # Standard Inception v3 feature dimension
            )
            
            return fid
        
    except ImportError:
        print("❌ pytorch-fid not available. Please install it manually:")
        print("pip install pytorch-fid")
        return None
    except Exception as e:
        print(f"❌ Error computing FID: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Compute FID scores for SRGAN models')
    
    # Model arguments
    parser.add_argument('--model_type', type=str, choices=['original', 'v2'], default='v2',
                       help='Model type (original or v2)')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--num_residual_blocks', type=int, default=23,
                       help='Number of RRDB blocks in generator (for v2 model)')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in generator')
    
    # Data arguments
    parser.add_argument('--hr_dir', type=str, default=DEFAULT_VAL_HR,
                       help='Path to HR images directory')
    parser.add_argument('--output_dir', type=str, default='fid_evaluation',
                       help='Output directory for SR images and results')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization)')
    parser.add_argument('--scale_factor', type=int, default=4,
                       help='Scale factor for super-resolution')
    parser.add_argument('--skip_generation', action='store_true',
                       help='Skip SR image generation (use existing SR images)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Install pytorch-fid if needed
    if not install_pytorch_fid():
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    sr_dir = os.path.join(args.output_dir, 'sr_images')
    os.makedirs(sr_dir, exist_ok=True)
    
    # Load model
    print(f"Loading {args.model_type} model from: {args.checkpoint}")
    model = load_model(args.model_type, args.checkpoint, device, args.num_residual_blocks, args.base_channels)
    
    # Generate SR images if not skipped
    if not args.skip_generation:
        print("Generating SR images...")
        success = generate_sr_images(model, args.hr_dir, sr_dir, device, args.max_size, args.scale_factor, args.model_type)
        if not success:
            print("❌ Failed to generate SR images")
            return
    else:
        print("Skipping SR image generation (using existing images)")
        if not os.path.exists(sr_dir) or len(os.listdir(sr_dir)) == 0:
            print("❌ No existing SR images found. Please generate them first or remove --skip_generation flag.")
            return
    
    # Compute FID score
    print("\nComputing FID score...")
    fid_score = compute_fid_score(sr_dir, args.hr_dir)
    
    if fid_score is not None:
        print(f"\n" + "="*50)
        print("FID EVALUATION RESULTS")
        print("="*50)
        print(f"Model Type: {args.model_type}")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"FID Score: {fid_score:.2f}")
        print(f"SR Images: {sr_dir}")
        print(f"HR Images: {args.hr_dir}")
        
        # Save results
        results_file = os.path.join(args.output_dir, 'fid_results.txt')
        with open(results_file, 'w') as f:
            f.write(f"Model Type: {args.model_type}\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"FID Score: {fid_score:.2f}\n")
            f.write(f"SR Images: {sr_dir}\n")
            f.write(f"HR Images: {args.hr_dir}\n")
        
        print(f"✓ Results saved to: {results_file}")
        
        # FID interpretation
        print(f"\nFID Score Interpretation:")
        if fid_score < 20:
            print("  🎉 Excellent! FID < 20 indicates very high quality")
        elif fid_score < 50:
            print("  ✅ Good! FID < 50 indicates good quality")
        elif fid_score < 100:
            print("  ⚠️  Fair! FID < 100 indicates acceptable quality")
        else:
            print("  ❌ Poor! FID >= 100 indicates low quality")
        
        print("="*50)
    else:
        print("❌ Failed to compute FID score")

if __name__ == '__main__':
    main() 