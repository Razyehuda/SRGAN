import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from models.srgan import Generator
import gc

# Default validation HR directory from training
DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"

def load_model(checkpoint_path, device):
    """Load the trained SRGAN generator from checkpoint."""
    generator = Generator(
        num_residual_blocks=16,
        num_channels=3,
        base_channels=64
    ).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded generator from checkpoint: {checkpoint_path}")
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        generator_state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items() 
                              if k.startswith('generator.')}
        generator.load_state_dict(generator_state_dict)
        print(f"Loaded generator from full model checkpoint: {checkpoint_path}")
    else:
        generator.load_state_dict(checkpoint)
        print(f"Loaded generator state dict from: {checkpoint_path}")
    
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

def create_comparison(model, hr_path, output_path, device, max_size=None):
    """Create comparison for a single image."""
    print(f"Processing: {os.path.basename(hr_path)}")
    
    try:
        # Load original HR image
        original_img = Image.open(hr_path).convert('RGB')
        
        # Resize HR if needed for memory optimization
        if max_size is not None:
            w, h = original_img.size
            if w > max_size or h > max_size:
                if w > h:
                    new_w = max_size
                    new_h = int(h * max_size / w)
                else:
                    new_h = max_size
                    new_w = int(w * max_size / h)
                original_img = original_img.resize((new_w, new_h), Image.BICUBIC)
                print(f"Resized HR from {w}x{h} to {new_w}x{new_h}")
        
        # Create LR image by downsampling HR (exactly like training dataset)
        lr_w = original_img.width // 4
        lr_h = original_img.height // 4
        lr_img = original_img.resize((lr_w, lr_h), Image.BICUBIC)
        print(f"Created LR image: {lr_w}x{lr_h} from HR: {original_img.width}x{original_img.height}")
        
        # Preprocess LR image for the model (same as training)
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
        ])
        
        input_tensor = transform(lr_img).unsqueeze(0).to(device)
        
        # Clear GPU cache
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        # Run inference
        with torch.no_grad():
            output_tensor = model(input_tensor)
        
        # Postprocess SR output (denormalize from [-1, 1] to [0, 1])
        output_tensor = (output_tensor + 1.0) / 2.0
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        output_tensor = output_tensor.squeeze(0).cpu()
        sr_img = transforms.ToPILImage()(output_tensor)
        
        # Create bicubic upscaled version from LR
        bicubic_img = lr_img.resize(original_img.size, Image.BICUBIC)
        
        # Create comparison
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(original_img)
        axes[0].set_title('Original HR')
        axes[0].axis('off')
        
        axes[1].imshow(lr_img)
        axes[1].set_title('LR Input (4x downsampled)')
        axes[1].axis('off')
        
        axes[2].imshow(bicubic_img)
        axes[2].set_title('Bicubic Upscaled')
        axes[2].axis('off')
        
        axes[3].imshow(sr_img)
        axes[3].set_title('SRGAN Output')
        axes[3].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison saved to: {output_path}")
        
        # Clear memory
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"Error processing {hr_path}: {e}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        return False

def main():
    parser = argparse.ArgumentParser(description='Create validation set comparisons')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_100.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--val_dir', type=str, default=DEFAULT_VAL_HR,
                       help='Path to validation HR images directory')
    parser.add_argument('--output_dir', type=str, default='validation_comparisons',
                       help='Output directory for comparisons')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization, default: no limit)')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of validation images to process')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Memory optimization tips
    if device.type == 'cuda' and args.max_size is None:
        print("Note: No max_size set - using full resolution images")
        print("If you run into memory issues, use --max_size 1024 or --device cpu")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Get validation images
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(args.val_dir) 
                   if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No image files found in {args.val_dir}")
        return
    
    # Limit number of samples
    image_files = image_files[:args.num_samples]
    print(f"Processing {len(image_files)} validation images")
    
    # Process each image
    successful = 0
    for i, filename in enumerate(image_files):
        hr_path = os.path.join(args.val_dir, filename)
        output_path = os.path.join(args.output_dir, f"comparison_{filename}")
        
        if create_comparison(model, hr_path, output_path, device, args.max_size):
            successful += 1
        
        # Progress update
        print(f"Progress: {i+1}/{len(image_files)} ({successful} successful)")
    
    print(f"\nCompleted! {successful}/{len(image_files)} comparisons created in {args.output_dir}")

if __name__ == '__main__':
    main() 