# test_both_models.py

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

from models.srgan import Generator
from models_v2 import GeneratorV2
from data.dataset import create_data_loaders
from utils.losses import calculate_psnr, calculate_ssim

# Default validation HR directory from training
DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"

def load_original_model(checkpoint_path, device):
    """Load the original SRGAN generator."""
    generator = Generator().to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✓ Loaded original generator from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        generator.load_state_dict(checkpoint)
        print(f"✓ Loaded original generator state dict from: {checkpoint_path}")
    
    generator.eval()
    return generator

def load_v2_model(checkpoint_path, device, num_residual_blocks=23, base_channels=64):
    """Load the v2 SRGAN generator."""
    generator = GeneratorV2(
        num_residual_blocks=num_residual_blocks,
        num_channels=3,
        base_channels=base_channels
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"✓ Loaded v2 generator from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'Unknown')}")
    else:
        generator.load_state_dict(checkpoint)
        print(f"✓ Loaded v2 generator state dict from: {checkpoint_path}")
    
    generator.eval()
    return generator

def preprocess_image(image_path, device, max_size=None):
    """Preprocess input image for the model."""
    img = Image.open(image_path).convert('RGB')
    
    if max_size is not None:
        w, h = img.size
        if w > max_size or h > max_size:
            if w > h:
                new_w = max_size
                new_h = int(h * max_size / w)
            else:
                new_h = max_size
                new_w = int(w * max_size / h)
            img = img.resize((new_w, new_h), Image.BICUBIC)
            print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    return img_tensor, img

def postprocess_image(output_tensor):
    """Convert model output back to PIL image."""
    output_tensor = (output_tensor + 1.0) / 2.0
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)
    return output_img

def create_lr_from_hr(hr_path, scale_factor=4, crop_size=256):
    """Create a low-resolution image from high-resolution for testing with random crop."""
    hr_img = Image.open(hr_path)
    w, h = hr_img.size
    
    # Ensure we can crop the specified size
    if w < crop_size or h < crop_size:
        # If image is too small, resize it to be at least crop_size
        min_size = max(crop_size, min(w, h))
        if w < min_size:
            hr_img = hr_img.resize((min_size, h), Image.BICUBIC)
            w = min_size
        if h < min_size:
            hr_img = hr_img.resize((w, min_size), Image.BICUBIC)
            h = min_size
    
    # Calculate random crop coordinates
    max_x = w - crop_size
    max_y = h - crop_size
    
    if max_x <= 0 or max_y <= 0:
        # If image is exactly crop_size, use (0, 0)
        crop_x, crop_y = 0, 0
    else:
        # Use fixed seed for reproducible crops
        np.random.seed(42)  # Fixed seed for consistent crops
        crop_x = np.random.randint(0, max_x + 1)
        crop_y = np.random.randint(0, max_y + 1)
    
    # Crop the HR image
    hr_crop = hr_img.crop((crop_x, crop_y, crop_x + crop_size, crop_y + crop_size))
    
    # Create LR by downsampling the crop
    lr_w, lr_h = crop_size // scale_factor, crop_size // scale_factor
    lr_img = hr_crop.resize((lr_w, lr_h), Image.BICUBIC)
    
    return lr_img, hr_crop

def inference_single_image(model, image_path, device, max_size=None):
    """Perform inference on a single image and return the output image."""
    try:
        input_tensor, original_img = preprocess_image(image_path, device, max_size)
        
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            output_tensor = model(input_tensor)
            output_tensor = output_tensor.clamp(-1.0, 1.0)
        
        output_img = postprocess_image(output_tensor)
        
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return output_img
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"❌ Memory error: {e}")
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return None
        else:
            raise e

def create_four_way_comparison(hr_path, lr_path, sr1_path, sr2_path, output_path, show_display=True):
    """Create a four-way comparison: LR, Model 1 Output, Model 2 Output, Ground Truth HR."""
    lr = Image.open(lr_path)
    sr1 = Image.open(sr1_path)
    sr2 = Image.open(sr2_path)
    hr = Image.open(hr_path)
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    axes[0].imshow(lr)
    axes[0].set_title('Low Resolution Input\n(4x downsampled)', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(sr1)
    axes[1].set_title('Model 1 Output\n(Original SRGAN)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(sr2)
    axes[2].set_title('Model 2 Output\n(SRGAN v2)', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(hr)
    axes[3].set_title('Ground Truth\n(High Resolution)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    fig.suptitle('SRGAN Model Comparison', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Four-way comparison saved to: {output_path}")
    
    if show_display:
        plt.show()
    
    plt.close()

def test_both_models_on_validation(model1, model2, hr_dir, output_dir, device, max_size=None, 
                                  scale_factor=4, crop_size=256, num_samples=10, show_display=True):
    """Test both models on validation images and create four-way comparisons."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Get validation image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    hr_files = [f for f in os.listdir(hr_dir) 
                if f.lower().endswith(image_extensions)]
    
    if not hr_files:
        print(f"No image files found in {hr_dir}")
        return
    
    # Limit to num_samples
    hr_files = hr_files[:num_samples]
    print(f"Testing with {len(hr_files)} validation images (crop size: {crop_size}x{crop_size})")
    
    for i, filename in enumerate(tqdm(hr_files, desc="Processing validation images")):
        hr_path = os.path.join(hr_dir, filename)
        
        # Create LR image and HR crop
        lr_img, hr_crop = create_lr_from_hr(hr_path, scale_factor, crop_size)
        
        # Save LR image temporarily
        lr_temp_path = os.path.join(output_dir, f"lr_temp_{filename}")
        lr_img.save(lr_temp_path)
        
        # Save HR crop temporarily
        hr_crop_path = os.path.join(output_dir, f"hr_crop_{filename}")
        hr_crop.save(hr_crop_path)
        
        # Process with both models
        sr1_path = os.path.join(output_dir, f"sr1_{filename}")
        sr2_path = os.path.join(output_dir, f"sr2_{filename}")
        
        try:
            # Model 1 inference
            sr1_img = inference_single_image(model1, lr_temp_path, device, max_size)
            if sr1_img:
                sr1_img.save(sr1_path)
            
            # Model 2 inference
            sr2_img = inference_single_image(model2, lr_temp_path, device, max_size)
            if sr2_img:
                sr2_img.save(sr2_path)
            
            # Create four-way comparison
            if sr1_img and sr2_img:
                comparison_path = os.path.join(output_dir, f"comparison_{filename}")
                create_four_way_comparison(hr_crop_path, lr_temp_path, sr1_path, sr2_path, 
                                         comparison_path, show_display=show_display)
                
                if show_display:
                    print(f"\nDisplaying comparison {i+1}/{len(hr_files)} for {filename}...")
                    input("Press Enter to continue to next image...")
            
            # Clean up temp files
            os.remove(lr_temp_path)
            os.remove(hr_crop_path)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            if os.path.exists(lr_temp_path):
                os.remove(lr_temp_path)

def main():
    parser = argparse.ArgumentParser(description='Compare Original SRGAN vs SRGAN v2 Models')
    
    # Model loading arguments
    parser.add_argument('--checkpoint1', type=str, required=True,
                       help='Path to original SRGAN checkpoint')
    parser.add_argument('--checkpoint2', type=str, required=True,
                       help='Path to SRGAN v2 checkpoint')
    parser.add_argument('--num_residual_blocks', type=int, default=23,
                       help='Number of RRDB blocks in v2 generator')
    parser.add_argument('--base_channels', type=int, default=64,
                       help='Base number of channels in v2 generator')
    
    # Testing arguments
    parser.add_argument('--hr_dir', type=str, default=DEFAULT_VAL_HR,
                       help='Path to HR images directory')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of validation samples to test')
    
    # Processing arguments
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization)')
    parser.add_argument('--scale_factor', type=int, default=4,
                       help='Scale factor for super-resolution')
    parser.add_argument('--crop_size', type=int, default=256,
                       help='Size of random crop (will be 16x smaller in pixels)')
    parser.add_argument('--no_display', action='store_true',
                       help='Disable interactive display (only save images)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print("Memory optimization tips:")
        if args.max_size is None:
            print("  - Use --max_size 512 to limit image size")
        print("  - Use --device cpu if GPU memory is insufficient")
    
    # Load both models
    print(f"Loading original SRGAN from: {args.checkpoint1}")
    model1 = load_original_model(args.checkpoint1, device)
    
    print(f"Loading SRGAN v2 from: {args.checkpoint2}")
    model2 = load_v2_model(args.checkpoint2, device, args.num_residual_blocks, args.base_channels)
    
    # Test both models
    show_display = not args.no_display
    test_both_models_on_validation(
        model1, model2, args.hr_dir, args.output_dir, device, 
        args.max_size, args.scale_factor, args.crop_size, args.num_samples, show_display
    )
    
    print(f"\n✓ Testing completed! Results saved in: {args.output_dir}")

if __name__ == '__main__':
    main() 