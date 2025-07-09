import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
from models.srgan import Generator
import matplotlib.pyplot as plt
import gc

# Default validation HR directory from training
DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"

def load_model(checkpoint_path, device):
    """Load the trained SRGAN generator from checkpoint."""
    # Create generator with same parameters as training
    generator = Generator(
        num_residual_blocks=16,
        num_channels=3,
        base_channels=64
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Check if checkpoint contains generator state dict
    if 'generator_state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['generator_state_dict'])
        print(f"Loaded generator from checkpoint: {checkpoint_path}")
    elif 'state_dict' in checkpoint:
        # If it's a full model checkpoint, extract generator
        state_dict = checkpoint['state_dict']
        # Filter only generator parameters
        generator_state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items() 
                              if k.startswith('generator.')}
        generator.load_state_dict(generator_state_dict)
        print(f"Loaded generator from full model checkpoint: {checkpoint_path}")
    else:
        # Assume it's just the generator state dict
        generator.load_state_dict(checkpoint)
        print(f"Loaded generator state dict from: {checkpoint_path}")
    
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
        
        # Postprocess
        output_img = postprocess_image(output_tensor)
        
        # Save result
        output_img.save(output_path)
        print(f"Saved result to: {output_path}")
        
        # Clear memory (but keep output_img for return)
        del input_tensor, output_tensor
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        return output_img
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Memory error processing {image_path}: {e}")
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
    
    # Process images in batches to manage memory
    for i in range(0, len(image_files), batch_size):
        batch_files = image_files[i:i + batch_size]
        
        for filename in batch_files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"sr_{filename}")
            
            try:
                result = inference_single_image(model, input_path, output_path, device, max_size)
                if result is None:
                    print(f"Skipped {filename} due to memory error")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
        
        # Clear memory between batches
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()


def compare_images(original_path, sr_path, output_path):
    """Create a side-by-side comparison of original and super-resolved images."""
    original = Image.open(original_path)
    sr = Image.open(sr_path)
    
    # Resize original to match SR size for comparison
    original_resized = original.resize(sr.size, Image.BICUBIC)
    
    # Create comparison image
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original)
    axes[0].set_title('Original (Low Resolution)')
    axes[0].axis('off')
    
    axes[1].imshow(original_resized)
    axes[1].set_title('Original (Upscaled with Bicubic)')
    axes[1].axis('off')
    
    axes[2].imshow(sr)
    axes[2].set_title('SRGAN Output')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='SRGAN Inference')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_epoch_100.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--input', type=str, default=DEFAULT_VAL_HR,
                       help='Path to input image or directory (default: validation HR images)')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output image or directory')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda, cpu, or auto)')
    parser.add_argument('--compare', action='store_true',
                       help='Create side-by-side comparison with original')
    parser.add_argument('--max_size', type=int, default=None,
                       help='Maximum image size (for memory optimization)')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for processing multiple images')
    
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
        print("  - Use --batch_size 1 for single image processing")
        print("  - Use --device cpu if GPU memory is insufficient")
    
    # Load model
    model = load_model(args.checkpoint, device)
    
    # Check if input is file or directory
    if os.path.isfile(args.input):
        # Single image inference
        if args.compare:
            # Create comparison
            sr_path = args.output
            inference_single_image(model, args.input, sr_path, device, args.max_size)
            compare_path = args.output.replace('.', '_comparison.')
            compare_images(args.input, sr_path, compare_path)
        else:
            inference_single_image(model, args.input, args.output, device, args.max_size)
    
    elif os.path.isdir(args.input):
        # Batch inference
        inference_batch(model, args.input, args.output, device, args.max_size, args.batch_size)
        
        if args.compare:
            print("Comparison mode not available for batch processing")
    
    else:
        print(f"Input path does not exist: {args.input}")


if __name__ == '__main__':
    main() 