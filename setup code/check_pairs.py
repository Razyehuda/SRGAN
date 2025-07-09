import os
import sys
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from data.dataset import DIV2KDataset

# Set these paths to your actual HR and LR directories
HR_DIR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
LR_DIR = None  # Set to None to generate LR images on the fly

# Use only ToTensor() for visualization
simple_transform = transforms.Compose([
    transforms.ToTensor()
])

# Instantiate the dataset
# You can also check the training set by changing HR_DIR

dataset = DIV2KDataset(hr_dir=HR_DIR, lr_dir=LR_DIR, is_training=False, transform=simple_transform)

print(f"Number of samples: {len(dataset)}")

# Check a few random pairs
import random
indices = random.sample(range(len(dataset)), min(5, len(dataset)))

for idx in indices:
    lr_img, hr_img = dataset[idx]
    print(f"Sample {idx}: LR shape: {lr_img.shape}, HR shape: {hr_img.shape}")
    lr_img_vis = lr_img.clamp(0, 1).permute(1, 2, 0).numpy()
    hr_img_vis = hr_img.clamp(0, 1).permute(1, 2, 0).numpy()
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(lr_img_vis)
    plt.title('LR Image')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(hr_img_vis)
    plt.title('HR Image')
    plt.axis('off')
    plt.suptitle(f'Sample {idx}')
    plt.show() 