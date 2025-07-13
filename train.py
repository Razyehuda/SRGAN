import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import time
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import csv

from models.srgan import SRGAN, Generator, Discriminator
from data.dataset import create_data_loaders
from utils.losses import SRGANLoss, DiscriminatorLoss, calculate_psnr, calculate_ssim

# Default paths for the DIV2K dataset in the KaggleHub cache
#DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
#DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
DEFAULT_TRAIN_HR = "/DIV2K_train_HR/DIV2K_train_HR"
DEFAULT_VAL_HR = "/DIV2K_valid_HR/DIV2K_valid_HR"

class SRGANTrainer:
    """SRGAN trainer class."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create models
        self.generator = Generator(
            num_residual_blocks=config.num_residual_blocks,
            num_channels=config.num_channels,
            base_channels=config.base_channels
        ).to(self.device)
        
        self.discriminator = Discriminator(
            num_channels=config.num_channels,
            base_channels=config.base_channels
        ).to(self.device)
        
        # Create optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.g_lr,
            betas=(0.9, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.d_lr,
            betas=(0.9, 0.999)
        )
        
        # Create learning rate schedulers
        # Use CosineAnnealingLR for better convergence
        self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, 
            T_max=config.num_epochs,
            eta_min=config.g_lr * config.g_lr_min_ratio
        )
        
        self.d_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, 
            T_max=config.num_epochs,
            eta_min=config.d_lr * config.d_lr_min_ratio
        )
        
        # Create loss functions
        self.g_loss_fn = SRGANLoss(
            content_weight=config.content_weight,
            perceptual_weight=config.perceptual_weight,
            adversarial_weight=config.adversarial_weight
        ).to(self.device)
        
        self.d_loss_fn = DiscriminatorLoss().to(self.device)
        
        # Add a separate loss function just for pre-training
        self.pretrain_loss_fn = nn.L1Loss().to(self.device)
        
        # Create data loaders
        self.train_loader, self.val_loader = create_data_loaders(
            hr_train_dir=config.hr_train_dir,
            hr_val_dir=config.hr_val_dir,
            lr_train_dir=config.lr_train_dir,
            lr_val_dir=config.lr_val_dir,
            batch_size=config.batch_size,
            patch_size=config.patch_size,
            scale_factor=config.scale_factor,
            num_workers=config.num_workers
        )
        
        # Create tensorboard writer
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.current_epoch = 0
        self.best_psnr = 0.0
        
        # Loss tracking lists for plotting
        self.epoch_losses = {
            'g_loss': [],
            'd_loss': [],
            'content_loss': [],
            'perceptual_loss': [],
            'adversarial_loss': [],
            'psnr': [],
            'ssim': []
        }
        
        # Create output directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)
        
    def train_epoch(self, pretrain=False):
        """Train for one epoch. If pretrain=True, only train generator with content/perceptual loss."""
        self.generator.train()
        self.discriminator.train()
        
        total_g_loss = 0.0
        total_d_loss = 0.0
        total_content_loss = 0.0
        total_perceptual_loss = 0.0
        total_adversarial_loss = 0.0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}')
        
        for batch_idx, (lr_imgs, hr_imgs) in enumerate(progress_bar):
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            if pretrain:
                # Pretrain generator only (no adversarial loss, no discriminator update)
                self.g_optimizer.zero_grad()
                fake_imgs = self.generator(lr_imgs)
                
                # Clamp the output before calculating loss
                fake_imgs_clamped = fake_imgs.clamp(-1.0, 1.0)
                
                # Use ONLY a pixel-wise loss for pre-training to maximize PSNR
                g_loss = self.pretrain_loss_fn(fake_imgs_clamped, hr_imgs)
                
                g_loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                self.g_optimizer.step()
                
                # Set other losses to 0 for logging purposes
                d_loss = torch.tensor(0.0)
                content_loss = g_loss  # For logging, content loss is our g_loss here
                perceptual_loss = torch.tensor(0.0)
                adversarial_loss = torch.tensor(0.0)
            else:
                # Train discriminator
                self.d_optimizer.zero_grad()
                fake_imgs = self.generator(lr_imgs)
                # Clamp the outputs before passing to the discriminator
                fake_imgs_clamped = fake_imgs.clamp(-1.0, 1.0)
                real_outputs = self.discriminator(hr_imgs)
                fake_outputs = self.discriminator(fake_imgs_clamped.detach())
                d_loss = self.d_loss_fn(real_outputs, fake_outputs)
                d_loss.backward()
                # Add gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
                self.d_optimizer.step()
                
                # Train generator
                g_loss = content_loss = perceptual_loss = adversarial_loss = 0.0
                for _ in range(self.config.g_steps):
                    self.g_optimizer.zero_grad()
                    # Generate fake images (reuse if it's the first iteration)
                    if _ == 0:
                        fake_imgs = self.generator(lr_imgs)
                        fake_imgs_clamped = fake_imgs.clamp(-1.0, 1.0)
                    else:
                        # For subsequent iterations, regenerate to get gradients
                        fake_imgs = self.generator(lr_imgs)
                        fake_imgs_clamped = fake_imgs.clamp(-1.0, 1.0)
                    
                    fake_outputs = self.discriminator(fake_imgs_clamped)
                    g_loss_, content_loss_, perceptual_loss_, adversarial_loss_ = self.g_loss_fn(
                        fake_imgs, hr_imgs, fake_outputs
                    )
                    g_loss_.backward()
                    # Add gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
                    self.g_optimizer.step()
                    g_loss += g_loss_.item()
                    content_loss += content_loss_.item()
                    perceptual_loss += perceptual_loss_.item()
                    adversarial_loss += adversarial_loss_.item()
                
                g_loss /= self.config.g_steps
                content_loss /= self.config.g_steps
                perceptual_loss /= self.config.g_steps
                adversarial_loss /= self.config.g_steps
            # Update statistics
            total_g_loss += g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss
            total_d_loss += d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss
            total_content_loss += content_loss.item() if isinstance(content_loss, torch.Tensor) else content_loss
            total_perceptual_loss += perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss
            total_adversarial_loss += adversarial_loss.item() if isinstance(adversarial_loss, torch.Tensor) else adversarial_loss
            # Update progress bar
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss:.4f}' if not isinstance(g_loss, torch.Tensor) else f'{g_loss.item():.4f}',
                'D_Loss': f'{d_loss:.4f}' if not isinstance(d_loss, torch.Tensor) else f'{d_loss.item():.4f}',
                'Content': f'{content_loss:.4f}' if not isinstance(content_loss, torch.Tensor) else f'{content_loss.item():.4f}',
                'Perceptual': f'{perceptual_loss:.4f}' if not isinstance(perceptual_loss, torch.Tensor) else f'{perceptual_loss.item():.4f}',
                'Adversarial': f'{adversarial_loss:.4f}' if not isinstance(adversarial_loss, torch.Tensor) else f'{adversarial_loss.item():.4f}'
            })
            # Log to tensorboard
            if batch_idx % self.config.log_interval == 0:
                step = self.current_epoch * len(self.train_loader) + batch_idx
                self.writer.add_scalar('Loss/Generator', g_loss.item() if isinstance(g_loss, torch.Tensor) else g_loss, step)
                self.writer.add_scalar('Loss/Discriminator', d_loss.item() if isinstance(d_loss, torch.Tensor) else d_loss, step)
                self.writer.add_scalar('Loss/Content', content_loss.item() if isinstance(content_loss, torch.Tensor) else content_loss, step)
                self.writer.add_scalar('Loss/Perceptual', perceptual_loss.item() if isinstance(perceptual_loss, torch.Tensor) else perceptual_loss, step)
                self.writer.add_scalar('Loss/Adversarial', adversarial_loss.item() if isinstance(adversarial_loss, torch.Tensor) else adversarial_loss, step)
        # Calculate average losses
        avg_g_loss = total_g_loss / len(self.train_loader)
        avg_d_loss = total_d_loss / len(self.train_loader)
        avg_content_loss = total_content_loss / len(self.train_loader)
        avg_perceptual_loss = total_perceptual_loss / len(self.train_loader)
        avg_adversarial_loss = total_adversarial_loss / len(self.train_loader)
        return {
            'g_loss': avg_g_loss,
            'd_loss': avg_d_loss,
            'content_loss': avg_content_loss,
            'perceptual_loss': avg_perceptual_loss,
            'adversarial_loss': avg_adversarial_loss
        }
    
    def validate(self):
        """Validate the model."""
        self.generator.eval()
        
        total_psnr = 0.0
        total_ssim = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(self.val_loader, desc='Validation'):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)
                
                # Generate super-resolved images
                sr_imgs = self.generator(lr_imgs)
                
                # Clamp the raw output to the valid [-1, 1] range first
                sr_imgs = sr_imgs.clamp(-1.0, 1.0)
                
                # Denormalize from [-1, 1] to [0, 1] for metric calculation
                sr_imgs = (sr_imgs + 1.0) / 2.0
                hr_imgs = (hr_imgs + 1.0) / 2.0
                
                # Calculate metrics
                for i in range(sr_imgs.size(0)):
                    sr_img = sr_imgs[i:i+1]
                    hr_img = hr_imgs[i:i+1]
                    
                    psnr = calculate_psnr(sr_img, hr_img)
                    ssim = calculate_ssim(sr_img, hr_img)
                    
                    total_psnr += psnr.item()
                    total_ssim += ssim.item()
                    num_samples += 1
        
        avg_psnr = total_psnr / num_samples
        avg_ssim = total_ssim / num_samples
        
        return avg_psnr, avg_ssim
    
    def save_samples(self, epoch):
        """Save sample images."""
        self.generator.eval()
        
        with torch.no_grad():
            # Get a batch from validation set
            lr_imgs, hr_imgs = next(iter(self.val_loader))
            lr_imgs = lr_imgs.to(self.device)
            hr_imgs = hr_imgs.to(self.device)
            
            # Generate super-resolved images
            sr_imgs = self.generator(lr_imgs)
            
            # Clamp the raw output to the valid [-1, 1] range first
            sr_imgs = sr_imgs.clamp(-1.0, 1.0)
            
            # Denormalize from [-1, 1] to [0, 1] for visualization
            lr_imgs = (lr_imgs + 1.0) / 2.0
            hr_imgs = (hr_imgs + 1.0) / 2.0
            sr_imgs = (sr_imgs + 1.0) / 2.0
            
            # Convert to numpy for visualization
            lr_imgs_np = lr_imgs.cpu().numpy()
            hr_imgs_np = hr_imgs.cpu().numpy()
            sr_imgs_np = sr_imgs.cpu().numpy()
            
            num_samples = min(4, lr_imgs_np.shape[0])
            
            # Create figure with proper sizing to show actual resolution differences
            fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
            if num_samples == 1:
                axes = np.expand_dims(axes, axis=0)
            
            for i in range(num_samples):
                # Low resolution - show at actual size
                lr_img = np.transpose(lr_imgs_np[i], (1, 2, 0))
                axes[i, 0].imshow(lr_img, interpolation='nearest')
                axes[i, 0].set_title(f'Low Resolution ({lr_img.shape[1]}x{lr_img.shape[0]})')
                axes[i, 0].axis('off')
                
                # Super resolved
                sr_img = np.transpose(sr_imgs_np[i], (1, 2, 0))
                axes[i, 1].imshow(sr_img, interpolation='nearest')
                axes[i, 1].set_title(f'Super Resolved ({sr_img.shape[1]}x{sr_img.shape[0]})')
                axes[i, 1].axis('off')
                
                # High resolution
                hr_img = np.transpose(hr_imgs_np[i], (1, 2, 0))
                axes[i, 2].imshow(hr_img, interpolation='nearest')
                axes[i, 2].set_title(f'High Resolution ({hr_img.shape[1]}x{hr_img.shape[0]})')
                axes[i, 2].axis('off')
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.sample_dir, f'samples_epoch_{epoch}.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    def save_checkpoint(self, epoch, psnr, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'psnr': psnr,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Check if checkpoint contains all required keys
            required_keys = ['generator_state_dict', 'discriminator_state_dict', 
                           'g_optimizer_state_dict', 'd_optimizer_state_dict', 'epoch', 'psnr']
            missing_keys = [key for key in required_keys if key not in checkpoint]
            if missing_keys:
                raise ValueError(f"Checkpoint missing required keys: {missing_keys}")
            
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            
            # Load scheduler states if available
            if 'g_scheduler_state_dict' in checkpoint:
                self.g_scheduler.load_state_dict(checkpoint['g_scheduler_state_dict'])
            if 'd_scheduler_state_dict' in checkpoint:
                self.d_scheduler.load_state_dict(checkpoint['d_scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.best_psnr = checkpoint['psnr']
            
            print(f"✓ Loaded checkpoint from epoch {self.current_epoch} with PSNR: {self.best_psnr:.2f}")
            
        except Exception as e:
            print(f"✗ Failed to load checkpoint from {checkpoint_path}: {e}")
            print("Starting training from scratch...")
            self.current_epoch = 0
            self.best_psnr = 0.0
    
    def train(self):
        """Main training loop."""
        print(f"Starting training on device: {self.device}")
        print(f"Number of training samples: {len(self.train_loader.dataset)}")
        print(f"Number of validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            
            # Pretrain generator for pretrain_epochs
            if epoch < self.config.pretrain_epochs:
                print(f"[Pretraining Generator] Epoch {epoch + 1}/{self.config.pretrain_epochs}")
                train_losses = self.train_epoch(pretrain=True)
            else:
                train_losses = self.train_epoch(pretrain=False)
            
            # Validate
            psnr, ssim = self.validate()
            
            # Store losses for plotting
            self.epoch_losses['g_loss'].append(train_losses['g_loss'])
            self.epoch_losses['d_loss'].append(train_losses['d_loss'])
            self.epoch_losses['content_loss'].append(train_losses['content_loss'])
            self.epoch_losses['perceptual_loss'].append(train_losses['perceptual_loss'])
            self.epoch_losses['adversarial_loss'].append(train_losses['adversarial_loss'])
            self.epoch_losses['psnr'].append(psnr)
            self.epoch_losses['ssim'].append(ssim)
            
            # Log metrics
            self.writer.add_scalar('Metrics/PSNR', psnr, epoch)
            self.writer.add_scalar('Metrics/SSIM', ssim, epoch)
            
            # Log learning rates
            current_g_lr = self.g_optimizer.param_groups[0]['lr']
            current_d_lr = self.d_optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate/Generator', current_g_lr, epoch)
            self.writer.add_scalar('Learning_Rate/Discriminator', current_d_lr, epoch)
            
            print(f"Train - G_Loss: {train_losses['g_loss']:.4f}, D_Loss: {train_losses['d_loss']:.4f}")
            print(f"Validation - PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")
            print(f"Learning Rates - Generator: {current_g_lr:.2e}, Discriminator: {current_d_lr:.2e}")
            
            # Save samples
            if (epoch + 1) % self.config.sample_interval == 0:
                self.save_samples(epoch + 1)
            
            # Save checkpoint
            is_best = psnr > self.best_psnr
            if is_best:
                self.best_psnr = psnr
            
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, psnr, is_best)
            
            # Step learning rate schedulers
            self.g_scheduler.step()
            self.d_scheduler.step()
            
            self.current_epoch = epoch + 1
        
        print(f"\nTraining completed! Best PSNR: {self.best_psnr:.2f}")
        self.writer.close()
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training curves at the end of training."""
        epochs = list(range(1, len(self.epoch_losses['g_loss']) + 1))
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SRGAN Training Progress', fontsize=16)
        
        # Track learning rates for plotting
        g_lrs = []
        d_lrs = []
        for epoch in range(len(self.epoch_losses['g_loss'])):
            # Simulate the learning rate schedule to get historical values
            # This is a simplified approach - in practice you'd want to store actual LR values
            if epoch < len(self.epoch_losses['g_loss']):
                g_lrs.append(self.config.g_lr * (0.5 * (1 + np.cos(np.pi * epoch / self.config.num_epochs))))
                d_lrs.append(self.config.d_lr * (0.5 * (1 + np.cos(np.pi * epoch / self.config.num_epochs))))
        
        # Plot losses
        loss_plots = [
            ('g_loss', 'Generator Loss', 'blue', 0, 0),
            ('d_loss', 'Discriminator Loss', 'red', 0, 1),
            ('content_loss', 'Content Loss', 'green', 0, 2),
            ('perceptual_loss', 'Perceptual Loss', 'orange', 1, 0),
            ('adversarial_loss', 'Adversarial Loss', 'purple', 1, 1),
        ]
        
        for loss_key, title, color, row, col in loss_plots:
            if self.epoch_losses[loss_key]:
                ax = axes[row, col]
                ax.plot(epochs, self.epoch_losses[loss_key], color=color, linewidth=2, marker='o', markersize=4)
                ax.set_title(title)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss Value')
                ax.grid(True, alpha=0.3)
                
                # Add moving average
                if len(self.epoch_losses[loss_key]) > 5:
                    window = min(5, len(self.epoch_losses[loss_key]) // 2)
                    moving_avg = np.convolve(self.epoch_losses[loss_key], np.ones(window)/window, mode='valid')
                    moving_epochs = epochs[window-1:]
                    ax.plot(moving_epochs, moving_avg, color=color, alpha=0.7, linestyle='--', linewidth=1)
        
        # Plot metrics
        if self.epoch_losses['psnr']:
            ax = axes[1, 2]
            ax.plot(epochs, self.epoch_losses['psnr'], color='blue', linewidth=2, marker='o', markersize=4, label='PSNR')
            ax.set_title('PSNR and SSIM')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('PSNR (dB)', color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, alpha=0.3)
            
            if self.epoch_losses['ssim']:
                ax2 = ax.twinx()
                ax2.plot(epochs, self.epoch_losses['ssim'], color='red', linewidth=2, marker='s', markersize=4, label='SSIM')
                ax2.set_ylabel('SSIM', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
        
        # Plot learning rates
        if g_lrs and d_lrs:
            # Create a new subplot for learning rates
            ax_lr = plt.subplot(2, 3, 6)
            ax_lr.plot(epochs, g_lrs, color='blue', linewidth=2, marker='o', markersize=4, label='Generator LR')
            ax_lr.plot(epochs, d_lrs, color='red', linewidth=2, marker='s', markersize=4, label='Discriminator LR')
            ax_lr.set_title('Learning Rates')
            ax_lr.set_xlabel('Epoch')
            ax_lr.set_ylabel('Learning Rate')
            ax_lr.legend()
            ax_lr.grid(True, alpha=0.3)
            ax_lr.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.sample_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save loss data to CSV
        self.save_loss_data()
        
        # Create detailed loss component plot
        self.plot_loss_components(epochs)
    
    def save_loss_data(self):
        """Save training loss data to CSV file."""
        csv_path = os.path.join(self.config.sample_dir, 'training_losses.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            fieldnames = ['epoch', 'g_loss', 'd_loss', 'content_loss', 'perceptual_loss', 'adversarial_loss', 'psnr', 'ssim', 'g_lr', 'd_lr']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for i in range(len(self.epoch_losses['g_loss'])):
                # Calculate learning rates for this epoch
                g_lr = self.config.g_lr * (0.5 * (1 + np.cos(np.pi * i / self.config.num_epochs)))
                d_lr = self.config.d_lr * (0.5 * (1 + np.cos(np.pi * i / self.config.num_epochs)))
                
                writer.writerow({
                    'epoch': i + 1,
                    'g_loss': self.epoch_losses['g_loss'][i],
                    'd_loss': self.epoch_losses['d_loss'][i],
                    'content_loss': self.epoch_losses['content_loss'][i],
                    'perceptual_loss': self.epoch_losses['perceptual_loss'][i],
                    'adversarial_loss': self.epoch_losses['adversarial_loss'][i],
                    'psnr': self.epoch_losses['psnr'][i],
                    'ssim': self.epoch_losses['ssim'][i],
                    'g_lr': g_lr,
                    'd_lr': d_lr
                })
        
        print(f"Loss data saved to {csv_path}")
    
    def plot_loss_components(self, epochs):
        """Plot generator loss components separately."""
        if not (self.epoch_losses['content_loss'] and self.epoch_losses['perceptual_loss'] and self.epoch_losses['adversarial_loss']):
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot all loss components
        components = ['content_loss', 'perceptual_loss', 'adversarial_loss']
        colors = ['green', 'orange', 'purple']
        labels = ['Content Loss', 'Perceptual Loss', 'Adversarial Loss']
        
        for component, color, label in zip(components, colors, labels):
            if self.epoch_losses[component]:
                ax1.plot(epochs, self.epoch_losses[component], color=color, linewidth=2, marker='o', markersize=4, label=label)
        
        ax1.set_title('Generator Loss Components')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot loss ratios
        content_values = np.array(self.epoch_losses['content_loss'])
        perceptual_values = np.array(self.epoch_losses['perceptual_loss'])
        adversarial_values = np.array(self.epoch_losses['adversarial_loss'])
        
        # Calculate ratios
        total_loss = content_values + perceptual_values + adversarial_values
        # Avoid division by zero
        total_loss = np.where(total_loss == 0, 1e-8, total_loss)
        content_ratio = content_values / total_loss
        perceptual_ratio = perceptual_values / total_loss
        adversarial_ratio = adversarial_values / total_loss
        
        ax2.plot(epochs, content_ratio, color='green', linewidth=2, marker='o', markersize=4, label='Content Ratio')
        ax2.plot(epochs, perceptual_ratio, color='orange', linewidth=2, marker='o', markersize=4, label='Perceptual Ratio')
        ax2.plot(epochs, adversarial_ratio, color='purple', linewidth=2, marker='o', markersize=4, label='Adversarial Ratio')
        
        ax2.set_title('Loss Component Ratios')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Ratio')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.sample_dir, 'loss_components.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        
        if self.epoch_losses['psnr']:
            print(f"Final PSNR: {self.epoch_losses['psnr'][-1]:.2f} dB")
            print(f"Best PSNR: {max(self.epoch_losses['psnr']):.2f} dB")
        else:
            print("No PSNR data available")
            
        if self.epoch_losses['ssim']:
            print(f"Final SSIM: {self.epoch_losses['ssim'][-1]:.4f}")
            print(f"Best SSIM: {max(self.epoch_losses['ssim']):.4f}")
        else:
            print("No SSIM data available")
            
        if self.epoch_losses['g_loss']:
            print(f"Final Generator Loss: {self.epoch_losses['g_loss'][-1]:.4f}")
        else:
            print("No generator loss data available")
            
        if self.epoch_losses['d_loss']:
            print(f"Final Discriminator Loss: {self.epoch_losses['d_loss'][-1]:.4f}")
        else:
            print("No discriminator loss data available")
        
        # Loss component analysis
        if (self.epoch_losses['content_loss'] and self.epoch_losses['perceptual_loss'] and 
            self.epoch_losses['adversarial_loss']):
            final_content = self.epoch_losses['content_loss'][-1]
            final_perceptual = self.epoch_losses['perceptual_loss'][-1]
            final_adversarial = self.epoch_losses['adversarial_loss'][-1]
            total_final = final_content + final_perceptual + final_adversarial
            
            if total_final > 0:
                print(f"\nFinal Loss Components:")
                print(f"  Content: {final_content:.4f} ({final_content/total_final*100:.1f}%)")
                print(f"  Perceptual: {final_perceptual:.4f} ({final_perceptual/total_final*100:.1f}%)")
                print(f"  Adversarial: {final_adversarial:.4f} ({final_adversarial/total_final*100:.1f}%)")
            else:
                print("\nFinal Loss Components: All zero")
        else:
            print("\nLoss component data not available")
            
        print("="*50)


def main():
    parser = argparse.ArgumentParser(description='SRGAN Training')
    
    # Data paths
    parser.add_argument('--hr_train_dir', type=str, default=DEFAULT_TRAIN_HR, help='High-resolution training images directory')
    parser.add_argument('--hr_val_dir', type=str, default=DEFAULT_VAL_HR, help='High-resolution validation images directory')
    parser.add_argument('--lr_train_dir', type=str, default=None, help='Low-resolution training images directory')
    parser.add_argument('--lr_val_dir', type=str, default=None, help='Low-resolution validation images directory')
    
    # Model parameters
    parser.add_argument('--num_residual_blocks', type=int, default=16, help='Number of residual blocks')
    parser.add_argument('--num_channels', type=int, default=3, help='Number of image channels')
    parser.add_argument('--base_channels', type=int, default=64, help='Base number of channels')
    parser.add_argument('--scale_factor', type=int, default=4, help='Upscaling factor')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=96, help='Training patch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--g_lr', type=float, default=1e-4, help='Generator learning rate')
    parser.add_argument('--d_lr', type=float, default=1e-5, help='Discriminator learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    
    # Learning rate scheduling
    parser.add_argument('--g_lr_min_ratio', type=float, default=0.01, help='Minimum generator LR as ratio of initial LR (for CosineAnnealingLR)')
    parser.add_argument('--d_lr_min_ratio', type=float, default=0.01, help='Minimum discriminator LR as ratio of initial LR (for CosineAnnealingLR)')
    
    # Loss weights
    parser.add_argument('--content_weight', type=float, default=1.0, help='Content loss weight')
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--adversarial_weight', type=float, default=3e-5, help='Adversarial loss weight')
    
    # Logging and saving
    parser.add_argument('--log_dir', type=str, default='logs', help='Tensorboard log directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--sample_dir', type=str, default='samples', help='Sample images directory')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    parser.add_argument('--save_interval', type=int, default=10, help='Checkpoint saving interval')
    parser.add_argument('--sample_interval', type=int, default=5, help='Sample saving interval')
    
    # Resume training
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    # New parameter
    parser.add_argument('--g_steps', type=int, default=2, help='Number of generator updates per discriminator update')
    
    # Pretraining parameter
    parser.add_argument('--pretrain_epochs', type=int, default=5, help='Number of epochs to pretrain the generator (no adversarial loss)')
    
    config = parser.parse_args()
    
    # Create trainer
    trainer = SRGANTrainer(config)
    
    # Resume from checkpoint if specified
    if config.resume:
        trainer.load_checkpoint(config.resume)
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main() 
