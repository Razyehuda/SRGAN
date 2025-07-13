# train_v2.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from models_v2 import GeneratorV2, DiscriminatorV2
from data.dataset import create_data_loaders
from losses_v2 import SRGANLossV2, DiscriminatorLossV2, calculate_psnr, calculate_ssim

# Default paths
#DEFAULT_TRAIN_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_train_HR/DIV2K_train_HR"
#DEFAULT_VAL_HR = r"C:/Users/Razye/.cache/kagglehub/datasets/joe1995/div2k-dataset/versions/1/DIV2K_valid_HR/DIV2K_valid_HR"
DEFAULT_TRAIN_HR = "/DIV2K_train_HR/DIV2K_train_HR"
DEFAULT_VAL_HR = "/DIV2K_valid_HR/DIV2K_valid_HR"
class SRGANTrainerV2:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # --- Create models ---
        self.generator = GeneratorV2(
            num_residual_blocks=config.num_residual_blocks,
            base_channels=config.base_channels
        ).to(self.device)
        
        # --- Create optimizers ---
        self.g_optimizer = optim.Adam(self.generator.parameters(), lr=config.g_lr, betas=(0.9, 0.999))
        self.g_scheduler = optim.lr_scheduler.StepLR(self.g_optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
        
        # --- Initialize components based on mode ---
        if self.config.mode == 'pretrain':
            print("--- Running in PRETRAIN mode ---")
            self.g_loss_fn = nn.L1Loss().to(self.device)
        elif self.config.mode == 'finetune':
            print("--- Running in FINETUNE mode ---")
            self.discriminator = DiscriminatorV2(base_channels=config.base_channels).to(self.device)
            self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.d_lr, betas=(0.9, 0.999))
            self.d_scheduler = optim.lr_scheduler.StepLR(self.d_optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)
            
            self.g_loss_fn = SRGANLossV2(
                content_weight=config.content_weight,
                perceptual_weight=config.perceptual_weight,
                adversarial_weight=config.adversarial_weight
            ).to(self.device)
            self.d_loss_fn = DiscriminatorLossV2().to(self.device)
        else:
            raise ValueError("Invalid mode specified. Choose 'pretrain' or 'finetune'.")
            
        # --- Data loaders ---
        self.train_loader, self.val_loader = create_data_loaders(
            hr_train_dir=config.hr_train_dir, hr_val_dir=config.hr_val_dir,
            batch_size=config.batch_size, patch_size=config.patch_size,
            scale_factor=config.scale_factor, num_workers=config.num_workers
        )
        
        # --- Logging and state ---
        self.writer = SummaryWriter(log_dir=config.log_dir)
        self.current_epoch = 0
        self.best_psnr = 0.0
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.sample_dir, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch, logic depends on the mode."""
        if self.config.mode == 'pretrain':
            return self.pretrain_generator_epoch()
        else:
            return self.finetune_gan_epoch()

    def pretrain_generator_epoch(self):
        self.generator.train()
        total_g_loss = 0.0
        progress_bar = tqdm(self.train_loader, desc=f'Pretrain Epoch {self.current_epoch + 1}')
        
        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
            
            self.g_optimizer.zero_grad()
            sr_imgs = self.generator(lr_imgs).clamp(-1.0, 1.0)
            
            g_loss = self.g_loss_fn(sr_imgs, hr_imgs)
            g_loss.backward()
            self.g_optimizer.step()
            
            total_g_loss += g_loss.item()
            progress_bar.set_postfix({'G_L1_Loss': f'{g_loss.item():.4f}'})

        avg_g_loss = total_g_loss / len(self.train_loader)
        self.writer.add_scalar('Pretrain/Loss_G', avg_g_loss, self.current_epoch)
        return {'g_loss': avg_g_loss, 'd_loss': 0, 'content_loss': avg_g_loss, 'perceptual_loss': 0, 'adversarial_loss': 0}

    def finetune_gan_epoch(self):
        self.generator.train()
        self.discriminator.train()
        
        total_losses = {'g': 0, 'd': 0, 'content': 0, 'perceptual': 0, 'adv': 0}
        progress_bar = tqdm(self.train_loader, desc=f'Finetune Epoch {self.current_epoch + 1}')

        for lr_imgs, hr_imgs in progress_bar:
            lr_imgs, hr_imgs = lr_imgs.to(self.device), hr_imgs.to(self.device)
            
            # --- Train Discriminator ---
            self.d_optimizer.zero_grad()
            sr_imgs = self.generator(lr_imgs).detach()
            sr_imgs_clamped = sr_imgs.clamp(-1.0, 1.0)
            
            real_outputs = self.discriminator(hr_imgs)
            fake_outputs = self.discriminator(sr_imgs_clamped)
            
            d_loss = self.d_loss_fn(real_outputs, fake_outputs)
            d_loss.backward()
            self.d_optimizer.step()
            
            # --- Train Generator ---
            self.g_optimizer.zero_grad()
            sr_imgs = self.generator(lr_imgs) # Re-evaluate with gradients
            sr_imgs_clamped = sr_imgs.clamp(-1.0, 1.0)
            
            real_outputs_for_g = self.discriminator(hr_imgs).detach()
            fake_outputs_for_g = self.discriminator(sr_imgs_clamped)
            
            g_loss, content_l, perceptual_l, adv_l = self.g_loss_fn(
                sr_imgs, hr_imgs, real_outputs_for_g, fake_outputs_for_g
            )
            g_loss.backward()
            self.g_optimizer.step()
            
            # Update totals and progress bar
            total_losses['g'] += g_loss.item()
            total_losses['d'] += d_loss.item()
            total_losses['content'] += content_l.item()
            total_losses['perceptual'] += perceptual_l.item()
            total_losses['adv'] += adv_l.item()
            progress_bar.set_postfix({
                'G_Loss': f'{g_loss.item():.4f}', 'D_Loss': f'{d_loss.item():.4f}',
                'Content': f'{content_l.item():.4f}', 'Adv': f'{adv_l.item():.4f}'
            })

        num_batches = len(self.train_loader)
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        self.writer.add_scalar('Finetune/Loss_G', avg_losses['g'], self.current_epoch)
        self.writer.add_scalar('Finetune/Loss_D', avg_losses['d'], self.current_epoch)
        return {
            'g_loss': avg_losses['g'], 'd_loss': avg_losses['d'],
            'content_loss': avg_losses['content'], 'perceptual_loss': avg_losses['perceptual'],
            'adversarial_loss': avg_losses['adv']
        }
        
    def validate(self):
        # This method can be reused from your original trainer without changes
        # It only uses the generator for evaluation.
        self.generator.eval()
        total_psnr = 0.0
        total_ssim = 0.0
        with torch.no_grad():
            for lr_imgs, hr_imgs in tqdm(self.val_loader, desc='Validation'):
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)  # Move hr_imgs to device
                sr_imgs = self.generator(lr_imgs).clamp(-1.0, 1.0)
                
                sr_imgs = (sr_imgs + 1.0) / 2.0
                hr_imgs = (hr_imgs + 1.0) / 2.0
                
                for i in range(sr_imgs.size(0)):
                    total_psnr += calculate_psnr(sr_imgs[i:i+1], hr_imgs[i:i+1]).item()
                    total_ssim += calculate_ssim(sr_imgs[i:i+1], hr_imgs[i:i+1]).item()
        
        avg_psnr = total_psnr / len(self.val_loader.dataset)
        avg_ssim = total_ssim / len(self.val_loader.dataset)
        
        self.writer.add_scalar('Metrics/PSNR', avg_psnr, self.current_epoch)
        self.writer.add_scalar('Metrics/SSIM', avg_ssim, self.current_epoch)
        return avg_psnr, avg_ssim

    def save_checkpoint(self, epoch, is_best=False):
        state = {'epoch': epoch, 'generator_state_dict': self.generator.state_dict()}
        if self.config.mode == 'finetune':
            state['discriminator_state_dict'] = self.discriminator.state_dict()
            state['d_optimizer_state_dict'] = self.d_optimizer.state_dict()
            
        filename = f'checkpoint_{self.config.mode}_epoch_{epoch}.pth'
        torch.save(state, os.path.join(self.config.checkpoint_dir, filename))
        if is_best:
            best_filename = f'best_model_{self.config.mode}.pth'
            torch.save(state, os.path.join(self.config.checkpoint_dir, best_filename))

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        
        if self.config.mode == 'finetune' and 'discriminator_state_dict' in checkpoint:
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            # You can also load optimizer states if resuming an interrupted finetune
        
        self.current_epoch = checkpoint.get('epoch', 0)
        print(f"✓ Loaded checkpoint from '{path}' at epoch {self.current_epoch}")
        
    def train(self):
        print(f"Starting training on {self.device} in '{self.config.mode}' mode.")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            train_losses = self.train_epoch()
            psnr, ssim = self.validate()
            
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")
            
            self.g_scheduler.step()
            if self.config.mode == 'finetune':
                self.d_scheduler.step()

            is_best = psnr > self.best_psnr
            if is_best: self.best_psnr = psnr
                
            if (epoch + 1) % self.config.save_interval == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
        
        print(f"\nTraining completed! Best PSNR: {self.best_psnr:.2f}")
        self.writer.close()

def main():
    parser = argparse.ArgumentParser(description='ESRGAN Training V2')
    # --- New arguments for two-stage training ---
    parser.add_argument('--mode', type=str, required=True, choices=['pretrain', 'finetune'], help='Training mode')
    parser.add_argument('--pretrain_checkpoint', type=str, default=None, help='Path to pre-trained generator for fine-tuning')
    
    # Existing arguments
    parser.add_argument('--hr_train_dir', type=str, default=DEFAULT_TRAIN_HR)
    parser.add_argument('--hr_val_dir', type=str, default=DEFAULT_VAL_HR)
    parser.add_argument('--num_residual_blocks', type=int, default=23, help='Number of RRDB blocks')
    parser.add_argument('--base_channels', type=int, default=64)
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--patch_size', type=int, default=128, help='HR patch size. LR will be patch_size // scale_factor')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--g_lr', type=float, default=1e-4)
    parser.add_argument('--d_lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr_step_size', type=int, default=50)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--content_weight', type=float, default=0.01)
    parser.add_argument('--perceptual_weight', type=float, default=1.0)
    parser.add_argument('--adversarial_weight', type=float, default=0.005)
    parser.add_argument('--log_dir', type=str, default='logs_v2')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_v2')
    parser.add_argument('--sample_dir', type=str, default='samples_v2')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None, help='Resume full training state from a checkpoint')
    
    config = parser.parse_args()
    
    trainer = SRGANTrainerV2(config)
    
    # --- Checkpoint loading logic ---
    if config.resume:
        trainer.load_checkpoint(config.resume)
    elif config.mode == 'finetune' and config.pretrain_checkpoint:
        print(f"Loading pre-trained generator from: {config.pretrain_checkpoint}")
        trainer.load_checkpoint(config.pretrain_checkpoint)
    elif config.mode == 'finetune' and not config.pretrain_checkpoint:
        raise ValueError("Must provide --pretrain_checkpoint when starting a new finetune session.")
        
    trainer.train()

if __name__ == '__main__':
    main()
