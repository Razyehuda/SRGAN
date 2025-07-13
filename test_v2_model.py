# test_v2_model.py

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
import gc
from tqdm import tqdm
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from models_v2 import GeneratorV2
from data.dataset import create_data_loaders
from losses_v2 import calculate_psnr, calculate_ssim

# Default validation HR directory from training
#DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
DEFAULT_VAL_HR = "/DIV2K_valid_HR/DIV2K_valid_HR"
def load_v2_model(checkpoint_path, device, num_residual_blocks=23, base_channels=64):
    """Load the trained SRGAN generator from train_v2 checkpoint."""
    # Create generator with same parameters as training
    generator = GeneratorV2(
        num_residual_blocks=num_residual_blocks,
        num_channels=3,
        base_channels=base_channels
    ).to(device)
    
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

def inference_single_image(model, image_path, output_path, device, max_size=None):
    """Perform inference on a single image."""
    print(f"Processing: {image_path}")
    
    try:
        # Preprocess
        input_tensor, original_img = preprocess_image(image_path, device, max_size)
        
        # Clear GPU cache before inference
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(-1.0, 1.0)  # Same as training
        
        # Postprocess
        output_img = postprocess_image(output_tensor)
        
        # Save result
        output_img.save(output_path)
        print(f"✓ Saved result to: {output_path}")
        
        # Clear memory
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return output_img
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Memory error processing {image_path}: {e}")
            print("Try reducing max_size or using CPU")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return None
        else:
            raise e

def inference_batch(model, input_dir, output_dir, device, max_size=None, batch_size=1):
    """Perform inference on all images in a directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process images
    for filename in tqdm(image_files, desc="Processing images"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"sr_{filename}")
        
        try:
            result = inference_single_image(model, input_path, output_path, device, max_size)
            if result is None:
                print(f"Skipped {filename} due to memory error")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
        
        # Clear memory between images
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

def evaluate_on_validation_set(model, val_loader, device):
    """Evaluate model on validation set and calculate PSNR/SSIM."""
    model.eval()
    total_psnr = 0.0
    total_ssim = 0.0
    num_samples = 0
    
    print("Evaluating on validation set...")
    with torch.no_grad():
        for lr_imgs, hr_imgs in tqdm(val_loader, desc="Validation"):
            lr_imgs = lr_imgs.to(device)
            hr_imgs = hr_imgs.to(device)
            
            sr_imgs = model(lr_imgs).clamp(-1.0, 1.0)
            
            # Convert to [0, 1] range for metric calculation
            sr_imgs = (sr_imgs + 1.0) / 2.0
            hr_imgs = (hr_imgs + 1.0) / 2.0
            
            for i in range(sr_imgs.size(0)):
                total_psnr += calculate_psnr(sr_imgs[i:i+1], hr_imgs[i:i+1]).item()
                total_ssim += calculate_ssim(sr_imgs[i:i+1], hr_imgs[i:i+1]).item()
                num_samples += 1
    
    avg_psnr = total_psnr / num_samples
    avg_ssim = total_ssim / num_samples
    
    print(f"Validation Results:")
    print(f"  PSNR: {avg_psnr:.2f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    
    return avg_psnr, avg_ssim

def compare_images(original_path, lr_path, sr_path, output_path, show_display=True):
    """Create a side-by-side comparison of original HR, LR input, and SR output images."""
    original = Image.open(original_path)
    lr = Image.open(lr_path)
    sr = Image.open(sr_path)
    
    # Create comparison image with three columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(original)
    axes[0].set_title('Original (High Resolution)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(lr)
    axes[1].set_title('Low Resolution Input (4x downsampled)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(sr)
    axes[2].set_title('SRGAN Output (Super-Resolution)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the comparison
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Three-way comparison saved to: {output_path}")
    
    # Display the comparison if requested
    if show_display:
        plt.show()
    
    plt.close()

def display_comparison_interactive(hr_path, lr_path, sr_path):
    """Display a three-way comparison interactively with better formatting."""
    original = Image.open(hr_path)
    lr = Image.open(lr_path)
    sr = Image.open(sr_path)
    
    # Create a larger figure for better visibility
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    
    # Display images
    axes[0].imshow(original)
    axes[0].set_title('Original (High Resolution)', fontsize=14, fontweight='bold', pad=20)
    axes[0].axis('off')
    
    axes[1].imshow(lr)
    axes[1].set_title('Low Resolution Input\n(4x downsampled)', fontsize=14, fontweight='bold', pad=20)
    axes[1].axis('off')
    
    axes[2].imshow(sr)
    axes[2].set_title('SRGAN Output\n(Super-Resolution)', fontsize=14, fontweight='bold', pad=20)
    axes[2].axis('off')
    
    # Add a main title
    fig.suptitle('SRGAN Super-Resolution Comparison', fontsize=16, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Show the comparison
    plt.show()
    
    return fig

def create_lr_from_hr(hr_path, scale_factor=4):
    """Create a low-resolution image from high-resolution for testing."""
    hr_img = Image.open(hr_path)
    w, h = hr_img.size
    
    # Create LR by downsampling
    lr_w, lr_h = w // scale_factor, h // scale_factor
    lr_img = hr_img.resize((lr_w, lr_h), Image.BICUBIC)
    
    return lr_img

def test_with_hr_images(model, hr_dir, output_dir, device, max_size=None, scale_factor=4, show_display=True):
    """Test model by creating LR images from HR and comparing results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all HR image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    hr_files = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(image_extensions)]
    
    if not hr_files:
        print(f"No image files found in {hr_dir}")
        return
    
    print(f"Testing with {len(hr_files)} HR images")
    
    for filename in tqdm(hr_files, desc="Testing with HR images"):
        hr_path = os.path.join(hr_dir, filename)
        
        # Create LR image
        lr_img = create_lr_from_hr(hr_path, scale_factor)
        
        # Save LR image temporarily
        lr_temp_path = os.path.join(output_dir, f"lr_temp_{filename}")
        lr_img.save(lr_temp_path)
        
        # Process with model
        sr_path = os.path.join(output_dir, f"sr_{filename}")
        try:
            inference_single_image(model, lr_temp_path, sr_path, device, max_size)
            
            # Create three-way comparison: HR, LR, SR
            comparison_path = os.path.join(output_dir, f"comparison_{filename}")
            compare_images(hr_path, lr_temp_path, sr_path, comparison_path, show_display=show_display)
            
            # Also display interactively if requested
            if show_display:
                print(f"\nDisplaying comparison for {filename}...")
                display_comparison_interactive(hr_path, lr_temp_path, sr_path)
                input("Press Enter to continue to next image...")
            
            # Clean up temp file
            os.remove(lr_temp_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            if os.path.exists(lr_temp_path):
                os.remove(lr_temp_path)

def main():
    parser = argparse.ArgumentParser(description='SRGAN V2 Model Testing')
    
    # Model loading arguments
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (pretrain or finetune)')
    parser.add_argument('--num_residual_blocks', type=int, default=23,
                       help='Number of RRDB blocks in generator')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in generator')
    
    # Testing mode arguments
    parser.add_argument('--mode', type=str, choices=['single', 'batch', 'validation', 'hr_test'],
                       required=True, help='Testing mode')
    parser.add_argument('--input', type=str, default=None,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output image or directory')
    
    # Validation set arguments
    parser.add_argument('--hr_val_dir', type=str, default=DEFAULT_VAL_HR,
                       help='Path to validation HR images')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for validation')
    parser.add_argument('--patch_size', type=int, default=128,
                       help='HR patch size for validation')
    parser.add_argument('--scale_factor', type=int, default=4,
                       help='Scale factor for super-resolution')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of workers for data loading')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization)')
    parser.add_argument('--compare', action='store_true',
                       help='Create side-by-side comparison with original')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable interactive display (only save images)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Memory optimization tips
    if device.type == 'cuda':
        print("Memory optimization tips:")
        if args.max_size is None:
            print("  - Use --max_size 512 to limit image size")
        print("  - Use --device cpu if GPU memory is insufficient")
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = load_v2_model(args.checkpoint, device, args.num_residual_blocks, args.base_channels)
    
    # Run appropriate test mode
    if args.mode == 'single':
        if not args.input:
            print("Error: --input is required for single image mode")
            return
        
        if args.compare:
            # For single image, we'll treat the input as HR and create LR from it
            hr_path = args.input
            lr_img = create_lr_from_hr(hr_path, args.scale_factor)
            
            # Save LR temporarily
            lr_temp_path = args.output.replace('.', '_lr_temp.')
            lr_img.save(lr_temp_path)
            
            # Create SR output
            sr_path = args.output
            inference_single_image(model, lr_temp_path, sr_path, device, args.max_size)
            
            # Create three-way comparison
            compare_path = args.output.replace('.', '_comparison.')
            show_display = not args.no_display
            compare_images(hr_path, lr_temp_path, sr_path, compare_path, show_display=show_display)
            
            # Also display interactively if requested
            if show_display:
                print(f"\nDisplaying comparison for {os.path.basename(args.input)}...")
                display_comparison_interactive(hr_path, lr_temp_path, sr_path)
            
            # Clean up temp file
            os.remove(lr_temp_path)
        else:
            inference_single_image(model, args.input, args.output, device, args.max_size)
    
    elif args.mode == 'batch':
        if not args.input:
            print("Error: --input is required for batch mode")
            return
        
        inference_batch(model, args.input, args.output, device, args.max_size)
        
        if args.compare:
            print("Comparison mode not available for batch processing")
    
    elif args.mode == 'validation':
        # Create validation data loader
        _, val_loader = create_data_loaders(
            hr_train_dir=None, hr_val_dir=args.hr_val_dir,
            batch_size=args.batch_size, patch_size=args.patch_size,
            scale_factor=args.scale_factor, num_workers=args.num_workers
        )
        
        # Evaluate on validation set
        psnr, ssim = evaluate_on_validation_set(model, val_loader, device)
        
        # Save results
        results = {
            'checkpoint': args.checkpoint,
            'psnr': psnr,
            'ssim': ssim
        }
        
        results_file = os.path.join(args.output, 'validation_results.txt')
        os.makedirs(args.output, exist_ok=True)
        with open(results_file, 'w') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"PSNR: {psnr:.2f} dB\n")
            f.write(f"SSIM: {ssim:.4f}\n")
        
        print(f"✓ Results saved to: {results_file}")
    
    elif args.mode == 'hr_test':
        if not args.input:
            print("Error: --input is required for hr_test mode")
            return
        
        show_display = not args.no_display
        test_with_hr_images(model, args.input, args.output, device, args.max_size, args.scale_factor, show_display)
    
    else:
        print(f"Unknown mode: {args.mode}")

if __name__ == '__main__':
    main() 
