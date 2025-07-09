#!/usr/bin/env python3
"""
Comprehensive evaluation script for SRGAN models.
Computes PSNR, SSIM, and FID scores for both original and v2 models.
Presents results in a table format with mean and standard deviation.
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
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

from models_v2 import GeneratorV2
from models.srgan import Generator
from data.dataset import create_data_loaders
from losses_v2 import calculate_psnr, calculate_ssim

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
        print(f"✓ Loaded {model_type} generator from checkpoint: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        # Assume it's just the generator state dict
        generator.load_state_dict(checkpoint)
        print(f"✓ Loaded {model_type} generator state dict from: {checkpoint_path}")
    
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

def evaluate_model_metrics(model, hr_dir, device, max_size=None, scale_factor=4, model_type='v2'):
    """Evaluate model and compute PSNR, SSIM for each image."""
    print(f"Evaluating {model_type} model...")
    
    # Get all HR image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    hr_files = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(image_extensions)]
    
    if not hr_files:
        print(f"No image files found in {hr_dir}")
        return None
    
    psnr_scores = []
    ssim_scores = []
    
    for filename in tqdm(hr_files, desc=f"Evaluating {model_type} model"):
        hr_path = os.path.join(hr_dir, filename)
        
        # Create LR image
        lr_img = create_lr_from_hr(hr_path, scale_factor)
        
        # Save LR image temporarily
        lr_temp_path = f"temp_lr_{filename}"
        lr_img.save(lr_temp_path)
        
        try:
            # Preprocess
            input_tensor, _ = preprocess_image(lr_temp_path, device, max_size)
            hr_tensor, _ = preprocess_image(hr_path, device, max_size)
            
            # Inference
            with torch.no_grad():
                output_tensor = model(input_tensor)
                output_tensor = output_tensor.clamp(-1.0, 1.0)
            
            # Convert to [0, 1] range for metric calculation
            sr_tensor = (output_tensor + 1.0) / 2.0
            hr_tensor = (hr_tensor + 1.0) / 2.0
            
            # Calculate metrics
            psnr = calculate_psnr(sr_tensor, hr_tensor).item()
            ssim = calculate_ssim(sr_tensor, hr_tensor).item()
            
            psnr_scores.append(psnr)
            ssim_scores.append(ssim)
            
            # Clean up temp file
            os.remove(lr_temp_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            if os.path.exists(lr_temp_path):
                os.remove(lr_temp_path)
    
    return {
        'psnr': psnr_scores,
        'ssim': ssim_scores
    }

def generate_sr_images_for_fid(model, hr_dir, output_dir, device, max_size=None, scale_factor=4, model_type='v2'):
    """Generate SR images for FID computation."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all HR image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    hr_files = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(image_extensions)]
    
    if not hr_files:
        print(f"No image files found in {hr_dir}")
        return False
    
    print(f"Generating SR images for FID computation...")
    
    for filename in tqdm(hr_files, desc=f"Generating SR images for {model_type}"):
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
    
    return True

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

def install_dependencies():
    """Install required dependencies."""
    try:
        import pytorch_fid
        import tabulate
        print("✓ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"Installing missing dependencies: {e}")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-fid", "tabulate", "seaborn"])
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")
            return False

def create_results_table(results):
    """Create a formatted results table."""
    table_data = []
    
    for model_name, metrics in results.items():
        psnr_mean = np.mean(metrics['psnr'])
        psnr_std = np.std(metrics['psnr'])
        ssim_mean = np.mean(metrics['ssim'])
        ssim_std = np.std(metrics['ssim'])
        fid = metrics.get('fid', 'N/A')
        
        table_data.append([
            model_name,
            f"{psnr_mean:.2f} ± {psnr_std:.2f}",
            f"{ssim_mean:.4f} ± {ssim_std:.4f}",
            f"{fid:.2f}" if isinstance(fid, (int, float)) else fid
        ])
    
    headers = ["Model", "PSNR (dB)", "SSIM", "FID"]
    table = tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".2f")
    
    return table

def create_comparison_plots(results, output_dir):
    """Create comparison plots for the metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('SRGAN Models Comparison', fontsize=16, fontweight='bold')
    
    # PSNR comparison
    ax1 = axes[0, 0]
    psnr_data = []
    labels = []
    for model_name, metrics in results.items():
        psnr_data.append(metrics['psnr'])
        labels.append(model_name)
    
    ax1.boxplot(psnr_data, labels=labels)
    ax1.set_title('PSNR Comparison')
    ax1.set_ylabel('PSNR (dB)')
    ax1.grid(True, alpha=0.3)
    
    # SSIM comparison
    ax2 = axes[0, 1]
    ssim_data = []
    for model_name, metrics in results.items():
        ssim_data.append(metrics['ssim'])
    
    ax2.boxplot(ssim_data, labels=labels)
    ax2.set_title('SSIM Comparison')
    ax2.set_ylabel('SSIM')
    ax2.grid(True, alpha=0.3)
    
    # FID comparison
    ax3 = axes[1, 0]
    fid_data = []
    fid_labels = []
    for model_name, metrics in results.items():
        if 'fid' in metrics and isinstance(metrics['fid'], (int, float)):
            fid_data.append(metrics['fid'])
            fid_labels.append(model_name)
    
    if fid_data:
        ax3.bar(fid_labels, fid_data, color=['blue', 'red'])
        ax3.set_title('FID Comparison')
        ax3.set_ylabel('FID Score')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'FID data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('FID Comparison')
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = "Model Comparison Summary\n\n"
    for model_name, metrics in results.items():
        psnr_mean = np.mean(metrics['psnr'])
        psnr_std = np.std(metrics['psnr'])
        ssim_mean = np.mean(metrics['ssim'])
        ssim_std = np.std(metrics['ssim'])
        fid = metrics.get('fid', 'N/A')
        
        summary_text += f"{model_name}:\n"
        summary_text += f"  PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB\n"
        summary_text += f"  SSIM: {ssim_mean:.4f} ± {ssim_std:.4f}\n"
        summary_text += f"  FID: {fid:.2f}\n\n"
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Comparison plots saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive evaluation of SRGAN models')
    
    # Model arguments
    parser.add_argument('--original_checkpoint', type=str, default='checkpoints/best_model.pth',
                       help='Path to original model checkpoint')
    parser.add_argument('--v2_checkpoint', type=str, default='checkpoints_v2/best_model_finetune.pth',
                       help='Path to v2 model checkpoint')
    parser.add_argument('--num_residual_blocks', type=int, default=23,
                       help='Number of RRDB blocks in v2 generator')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in v2 generator')
    
    # Data arguments
    parser.add_argument('--hr_dir', type=str, default=DEFAULT_VAL_HR,
                       help='Path to HR images directory')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization)')
    parser.add_argument('--scale_factor', type=int, default=4,
                       help='Scale factor for super-resolution')
    parser.add_argument('--skip_fid', action='store_true',
                       help='Skip FID computation to save time')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Install dependencies
    if not install_dependencies():
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Results storage
    results = {}
    
    # Evaluate Original Model
    print("\n" + "="*50)
    print("EVALUATING ORIGINAL MODEL")
    print("="*50)
    
    try:
        original_model = load_model('original', args.original_checkpoint, device)
        original_metrics = evaluate_model_metrics(original_model, args.hr_dir, device, args.max_size, args.scale_factor, 'original')
        
        if original_metrics:
            results['Original SRGAN'] = original_metrics
            
            # Generate SR images for FID if not skipped
            if not args.skip_fid:
                original_sr_dir = os.path.join(args.output_dir, 'original_sr_images')
                generate_sr_images_for_fid(original_model, args.hr_dir, original_sr_dir, device, args.max_size, args.scale_factor, 'original')
                
                # Compute FID
                original_fid = compute_fid_score(original_sr_dir, args.hr_dir)
                if original_fid is not None:
                    results['Original SRGAN']['fid'] = original_fid
    except Exception as e:
        print(f"❌ Error evaluating original model: {e}")
    
    # Evaluate V2 Model
    print("\n" + "="*50)
    print("EVALUATING V2 MODEL")
    print("="*50)
    
    try:
        v2_model = load_model('v2', args.v2_checkpoint, device, args.num_residual_blocks, args.base_channels)
        v2_metrics = evaluate_model_metrics(v2_model, args.hr_dir, device, args.max_size, args.scale_factor, 'v2')
        
        if v2_metrics:
            results['SRGAN V2'] = v2_metrics
            
            # Generate SR images for FID if not skipped
            if not args.skip_fid:
                v2_sr_dir = os.path.join(args.output_dir, 'v2_sr_images')
                generate_sr_images_for_fid(v2_model, args.hr_dir, v2_sr_dir, device, args.max_size, args.scale_factor, 'v2')
                
                # Compute FID
                v2_fid = compute_fid_score(v2_sr_dir, args.hr_dir)
                if v2_fid is not None:
                    results['SRGAN V2']['fid'] = v2_fid
    except Exception as e:
        print(f"❌ Error evaluating v2 model: {e}")
    
    # Create results table
    if results:
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        table = create_results_table(results)
        print(table)
        
        # Save results
        results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_file, 'w') as f:
            f.write("SRGAN Models Evaluation Results\n")
            f.write("="*50 + "\n\n")
            f.write(table)
            f.write("\n\nDetailed Results:\n")
            for model_name, metrics in results.items():
                f.write(f"\n{model_name}:\n")
                f.write(f"  PSNR: {np.mean(metrics['psnr']):.2f} ± {np.std(metrics['psnr']):.2f} dB\n")
                f.write(f"  SSIM: {np.mean(metrics['ssim']):.4f} ± {np.std(metrics['ssim']):.4f}\n")
                if 'fid' in metrics:
                    f.write(f"  FID: {metrics['fid']:.2f}\n")
        
        print(f"\n✓ Results saved to: {results_file}")
        
        # Create comparison plots
        create_comparison_plots(results, args.output_dir)
        
    else:
        print("❌ No results to display")

if __name__ == '__main__':
    main() 