import torch
import time
import numpy as np

def check_cuda():
    """Check CUDA availability and performance."""
    
    print("="*50)
    print("CUDA DIAGNOSTICS")
    print("="*50)
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        
        # Set device
        device = torch.device('cuda')
        print(f"Using device: {device}")
        
        # Test GPU performance
        print("\nTesting GPU performance...")
        
        # Create test tensors
        size = (3, 64, 64)
        x_cpu = torch.randn(size)
        x_gpu = x_cpu.to(device)
        
        # Test CPU vs GPU speed
        start_time = time.time()
        for _ in range(100):
            _ = torch.nn.functional.conv2d(x_cpu, torch.randn(64, 3, 3, 3), padding=1)
        cpu_time = time.time() - start_time
        
        start_time = time.time()
        for _ in range(100):
            _ = torch.nn.functional.conv2d(x_gpu, torch.randn(64, 3, 3, 3, device=device), padding=1)
        gpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.1f}x")
        
        # Check memory usage
        print(f"\nGPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
        print(f"GPU memory cached: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")
        
    else:
        print("CUDA is not available. Training will use CPU.")
        print("This will be significantly slower!")
        
        # Test CPU performance
        print("\nTesting CPU performance...")
        size = (3, 64, 64)
        x_cpu = torch.randn(size)
        
        start_time = time.time()
        for _ in range(10):  # Reduced iterations for CPU
            _ = torch.nn.functional.conv2d(x_cpu, torch.randn(64, 3, 3, 3), padding=1)
        cpu_time = time.time() - start_time
        
        print(f"CPU time: {cpu_time:.3f}s")
    
    print("="*50)
    
    return torch.cuda.is_available()

def check_training_config():
    """Check the training configuration for potential performance issues."""
    
    print("\n" + "="*50)
    print("TRAINING CONFIGURATION ANALYSIS")
    print("="*50)
    
    # Default configuration values from train.py
    config = {
        'batch_size': 16,
        'patch_size': 96,
        'num_epochs': 100,
        'num_residual_blocks': 16,
        'base_channels': 64,
        'scale_factor': 4,
        'g_steps': 2,
        'pretrain_epochs': 5
    }
    
    print("Current default configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Calculate model complexity
    # Rough estimate of parameters
    g_params = config['num_residual_blocks'] * config['base_channels'] * 64 * 3 * 3 * 2  # Residual blocks
    g_params += config['base_channels'] * 64 * 3 * 3  # Initial conv
    g_params += 64 * 3 * 3 * 3  # Final conv
    
    d_params = config['base_channels'] * 64 * 3 * 3 * 4  # Discriminator layers
    
    total_params = g_params + d_params
    
    print(f"\nEstimated model parameters:")
    print(f"  Generator: ~{g_params/1e6:.1f}M")
    print(f"  Discriminator: ~{d_params/1e6:.1f}M")
    print(f"  Total: ~{total_params/1e6:.1f}M")
    
    # Performance recommendations
    print(f"\nPerformance recommendations:")
    
    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available - training will be very slow!")
        print("  💡 Consider reducing batch_size to 4-8")
        print("  💡 Consider reducing num_residual_blocks to 8")
        print("  💡 Consider reducing num_epochs for testing")
    
    if config['batch_size'] > 8 and not torch.cuda.is_available():
        print("  ⚠️  Large batch size without GPU will be very slow!")
    
    if config['num_residual_blocks'] > 8:
        print("  💡 Consider reducing num_residual_blocks for faster training")
    
    print("="*50)

if __name__ == "__main__":
    cuda_available = check_cuda()
    check_training_config()
    
    if not cuda_available:
        print("\n🚨 WARNING: Training without CUDA will be extremely slow!")
        print("Consider:")
        print("1. Installing CUDA and PyTorch with CUDA support")
        print("2. Using Google Colab or other cloud GPU services")
        print("3. Reducing model complexity for CPU training") 