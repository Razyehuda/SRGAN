import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import random
import cv2
from torchvision import transforms


class DIV2KDataset(Dataset):
    """DIV2K dataset loader for SRGAN training."""
    
    def __init__(self, hr_dir, lr_dir=None, patch_size=96, scale_factor=4, 
                 transform=None, is_training=True):
        """
        Args:
            hr_dir: Directory with high-resolution images
            lr_dir: Directory with low-resolution images (if None, will downsample HR)
            patch_size: Size of training patches
            scale_factor: Upscaling factor
            transform: Optional transform to be applied
            is_training: Whether this is for training or validation
        """
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.transform = transform
        self.is_training = is_training
        
        # Get list of image files
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.hr_files:
            raise ValueError(f"No image files found in HR directory: {hr_dir}")
        
        if self.lr_dir:
            self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not self.lr_files:
                raise ValueError(f"No image files found in LR directory: {lr_dir}")
        else:
            self.lr_files = self.hr_files
            
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.lr_dir:
            # Load pre-generated LR image
            lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
            lr_img = Image.open(lr_path).convert('RGB')
        else:
            # Generate LR image by downsampling
            lr_img = hr_img.resize((hr_img.width // self.scale_factor, 
                                   hr_img.height // self.scale_factor), 
                                  Image.BICUBIC)
        
        if self.is_training:
            # Random crop for training
            hr_img, lr_img = self._random_crop(hr_img, lr_img)
            
            # Random horizontal flip
            if random.random() > 0.5:
                hr_img = hr_img.transpose(Image.FLIP_LEFT_RIGHT)
                lr_img = lr_img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.choice([90, 180, 270])
                hr_img = hr_img.rotate(angle)
                lr_img = lr_img.rotate(angle)
        
        # Convert to tensors
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        else:
            hr_img = transforms.ToTensor()(hr_img)
            lr_img = transforms.ToTensor()(lr_img)
        
        return lr_img, hr_img
    
    def _random_crop(self, hr_img, lr_img):
        """Random crop both HR and LR images."""
        w, h = hr_img.size
        lr_w, lr_h = lr_img.size
        
        # Calculate crop size
        crop_size = self.patch_size
        lr_crop_size = crop_size // self.scale_factor
        
        # Ensure we have enough space for cropping
        if w < crop_size or h < crop_size:
            raise ValueError(f"Image size ({w}x{h}) is smaller than crop size ({crop_size})")
        
        if lr_w < lr_crop_size or lr_h < lr_crop_size:
            raise ValueError(f"LR image size ({lr_w}x{lr_h}) is smaller than LR crop size ({lr_crop_size})")
        
        # Random crop coordinates for HR image
        x = random.randint(0, w - crop_size)
        y = random.randint(0, h - crop_size)
        
        # Calculate corresponding LR crop coordinates
        # The LR crop should correspond to the same spatial region as the HR crop
        lr_x = x // self.scale_factor
        lr_y = y // self.scale_factor
        
        # Ensure LR crop coordinates are within bounds
        lr_x = min(lr_x, lr_w - lr_crop_size)
        lr_y = min(lr_y, lr_h - lr_crop_size)
        
        # Crop images
        hr_crop = hr_img.crop((x, y, x + crop_size, y + crop_size))
        lr_crop = lr_img.crop((lr_x, lr_y, lr_x + lr_crop_size, lr_y + lr_crop_size))
        
        return hr_crop, lr_crop


class DIV2KTestDataset(Dataset):
    """DIV2K test dataset loader."""
    
    def __init__(self, hr_dir, lr_dir=None, scale_factor=4, transform=None):
        self.hr_dir = hr_dir
        self.lr_dir = lr_dir
        self.scale_factor = scale_factor
        self.transform = transform
        
        self.hr_files = sorted([f for f in os.listdir(hr_dir) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if not self.hr_files:
            raise ValueError(f"No image files found in HR directory: {hr_dir}")
        
        if self.lr_dir:
            self.lr_files = sorted([f for f in os.listdir(lr_dir) 
                                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            if not self.lr_files:
                raise ValueError(f"No image files found in LR directory: {lr_dir}")
        else:
            self.lr_files = self.hr_files
    
    def __len__(self):
        return len(self.hr_files)
    
    def __getitem__(self, idx):
        # Load HR image
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])
        hr_img = Image.open(hr_path).convert('RGB')
        
        if self.lr_dir:
            # Load pre-generated LR image
            lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
            lr_img = Image.open(lr_path).convert('RGB')
        else:
            # Generate LR image by downsampling
            lr_img = hr_img.resize((hr_img.width // self.scale_factor, 
                                   hr_img.height // self.scale_factor), 
                                  Image.BICUBIC)
        
        # Convert to tensors
        if self.transform:
            hr_img = self.transform(hr_img)
            lr_img = self.transform(lr_img)
        else:
            hr_img = transforms.ToTensor()(hr_img)
            lr_img = transforms.ToTensor()(lr_img)
        
        return lr_img, hr_img


def create_data_loaders(hr_train_dir, hr_val_dir, lr_train_dir=None, lr_val_dir=None,
                       batch_size=16, patch_size=96, scale_factor=4, num_workers=4):
    """Create training and validation data loaders."""
    
    # Transforms - Use [-1, 1] normalization for GAN training
    # This matches the generator's tanh output
    train_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0, 1]
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Converts to [-1, 1]
    ])
    
    # Create training dataset (with cropping)
    train_dataset = DIV2KDataset(
        hr_dir=hr_train_dir,
        lr_dir=lr_train_dir,
        patch_size=patch_size,
        scale_factor=scale_factor,
        transform=train_transform,
        is_training=True
    )
    
    # Create validation dataset (without cropping, full images)
    val_dataset = DIV2KTestDataset(
        hr_dir=hr_val_dir,
        lr_dir=lr_val_dir,
        scale_factor=scale_factor,
        transform=val_transform
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,  # Keep batch size 1 for validation due to variable image sizes
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 