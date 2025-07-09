import torch
from models.srgan import Generator
from torchvision import transforms
from PIL import Image
import os

def test_inference():
    """Test inference with the dummy image."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint_path = 'checkpoints/checkpoint_epoch_100.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    # Create generator
    generator = Generator(
        num_residual_blocks=16,
        num_channels=3,
        base_channels=64
    ).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Try different checkpoint formats
        if 'generator_state_dict' in checkpoint:
            generator.load_state_dict(checkpoint['generator_state_dict'])
            print("Loaded generator from 'generator_state_dict'")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            generator_state_dict = {k.replace('generator.', ''): v for k, v in state_dict.items() 
                                  if k.startswith('generator.')}
            generator.load_state_dict(generator_state_dict)
            print("Loaded generator from 'state_dict'")
        else:
            generator.load_state_dict(checkpoint)
            print("Loaded generator from direct state dict")
            
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    generator.eval()
    
    # Test with dummy image
    dummy_path = 'dummy_data/dummy.png'
    
    if not os.path.exists(dummy_path):
        print(f"Dummy image not found: {dummy_path}")
        return
    
    # Load and preprocess image
    img = Image.open(dummy_path).convert('RGB')
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        output_tensor = generator(input_tensor)
    
    print(f"Output shape: {output_tensor.shape}")
    
    # Postprocess
    output_tensor = (output_tensor + 1.0) / 2.0  # Denormalize from [-1, 1] to [0, 1]
    output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    
    # Convert to PIL image
    output_tensor = output_tensor.squeeze(0).cpu()
    output_img = transforms.ToPILImage()(output_tensor)
    
    # Save result
    output_path = 'test_inference_output.png'
    output_img.save(output_path)
    
    print(f"Test successful! Output saved to: {output_path}")
    print(f"Original size: {img.size}")
    print(f"Output size: {output_img.size}")

if __name__ == '__main__':
    test_inference() 