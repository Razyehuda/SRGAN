#!/usr/bin/env python3
"""
Script to create training plots from TensorBoard logs.
This script extracts training data from the logs_v2 directory and creates
comprehensive training plots showing loss curves, metrics, and learning rates.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

def extract_tensorboard_data(log_dir):
    """Extract data from TensorBoard logs."""
    print(f"Extracting data from: {log_dir}")
    
    # Find all event files
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    
    if not event_files:
        print(f"No TensorBoard event files found in {log_dir}")
        return None
    
    # Use the most recent event file
    latest_event_file = max(event_files, key=os.path.getctime)
    print(f"Using event file: {latest_event_file}")
    
    # Load the event file
    ea = EventAccumulator(latest_event_file)
    ea.Reload()
    
    # Extract scalar data
    data = {}
    
    # Get all scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Found scalar tags: {scalar_tags}")
    
    for tag in scalar_tags:
        events = ea.Scalars(tag)
        data[tag] = {
            'steps': [event.step for event in events],
            'values': [event.value for event in events],
            'wall_times': [event.wall_time for event in events]
        }
    
    return data

def create_training_plots(data, output_dir='training_plots'):
    """Create comprehensive training plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    if not data:
        print("No data to plot!")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('SRGAN V2 Training Progress', fontsize=16, fontweight='bold')
    
    # Plot 1: Generator Loss
    if 'Pretrain/Loss_G' in data or 'Finetune/Loss_G' in data:
        ax = axes[0, 0]
        if 'Pretrain/Loss_G' in data:
            ax.plot(data['Pretrain/Loss_G']['steps'], data['Pretrain/Loss_G']['values'], 
                   label='Pretrain', color='blue', linewidth=2, marker='o', markersize=4)
        if 'Finetune/Loss_G' in data:
            ax.plot(data['Finetune/Loss_G']['steps'], data['Finetune/Loss_G']['values'], 
                   label='Finetune', color='red', linewidth=2, marker='s', markersize=4)
        ax.set_title('Generator Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot 2: Discriminator Loss
    if 'Finetune/Loss_D' in data:
        ax = axes[0, 1]
        ax.plot(data['Finetune/Loss_D']['steps'], data['Finetune/Loss_D']['values'], 
               color='red', linewidth=2, marker='s', markersize=4)
        ax.set_title('Discriminator Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss Value')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: PSNR
    if 'Metrics/PSNR' in data:
        ax = axes[0, 2]
        ax.plot(data['Metrics/PSNR']['steps'], data['Metrics/PSNR']['values'], 
               color='green', linewidth=2, marker='o', markersize=4)
        ax.set_title('PSNR (Peak Signal-to-Noise Ratio)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)')
        ax.grid(True, alpha=0.3)
    
    # Plot 4: SSIM
    if 'Metrics/SSIM' in data:
        ax = axes[1, 0]
        ax.plot(data['Metrics/SSIM']['steps'], data['Metrics/SSIM']['values'], 
               color='orange', linewidth=2, marker='o', markersize=4)
        ax.set_title('SSIM (Structural Similarity Index)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('SSIM')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Combined PSNR and SSIM
    if 'Metrics/PSNR' in data and 'Metrics/SSIM' in data:
        ax = axes[1, 1]
        ax.plot(data['Metrics/PSNR']['steps'], data['Metrics/PSNR']['values'], 
               color='blue', linewidth=2, marker='o', markersize=4, label='PSNR')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('PSNR (dB)', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        ax.grid(True, alpha=0.3)
        
        ax2 = ax.twinx()
        ax2.plot(data['Metrics/SSIM']['steps'], data['Metrics/SSIM']['values'], 
                color='red', linewidth=2, marker='s', markersize=4, label='SSIM')
        ax2.set_ylabel('SSIM', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        ax.set_title('PSNR and SSIM Combined')
    
    # Plot 6: Training Summary
    ax = axes[1, 2]
    ax.text(0.1, 0.8, 'Training Summary', fontsize=14, fontweight='bold', transform=ax.transAxes)
    
    summary_text = []
    if 'Metrics/PSNR' in data:
        final_psnr = data['Metrics/PSNR']['values'][-1]
        best_psnr = max(data['Metrics/PSNR']['values'])
        summary_text.append(f'Final PSNR: {final_psnr:.2f} dB')
        summary_text.append(f'Best PSNR: {best_psnr:.2f} dB')
    
    if 'Metrics/SSIM' in data:
        final_ssim = data['Metrics/SSIM']['values'][-1]
        best_ssim = max(data['Metrics/SSIM']['values'])
        summary_text.append(f'Final SSIM: {final_ssim:.4f}')
        summary_text.append(f'Best SSIM: {best_ssim:.4f}')
    
    if 'Pretrain/Loss_G' in data or 'Finetune/Loss_G' in data:
        if 'Finetune/Loss_G' in data:
            final_g_loss = data['Finetune/Loss_G']['values'][-1]
        else:
            final_g_loss = data['Pretrain/Loss_G']['values'][-1]
        summary_text.append(f'Final G Loss: {final_g_loss:.4f}')
    
    if 'Finetune/Loss_D' in data:
        final_d_loss = data['Finetune/Loss_D']['values'][-1]
        summary_text.append(f'Final D Loss: {final_d_loss:.4f}')
    
    for i, text in enumerate(summary_text):
        ax.text(0.1, 0.7 - i*0.1, text, fontsize=10, transform=ax.transAxes)
    
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save data to CSV
    save_data_to_csv(data, output_dir)
    
    print(f"✓ Training plots saved to: {output_dir}")
    return data

def save_data_to_csv(data, output_dir):
    """Save extracted data to CSV files."""
    # Create a combined DataFrame
    all_data = {}
    
    for tag, tag_data in data.items():
        # Convert to DataFrame
        df = pd.DataFrame({
            'epoch': tag_data['steps'],
            'value': tag_data['values'],
            'wall_time': tag_data['wall_times']
        })
        
        # Save individual tag data
        tag_name = tag.replace('/', '_').replace(' ', '_')
        csv_path = os.path.join(output_dir, f'{tag_name}.csv')
        df.to_csv(csv_path, index=False)
        
        # Add to combined data
        all_data[tag] = df
    
    # Create a combined CSV with all metrics
    if 'Metrics/PSNR' in all_data and 'Metrics/SSIM' in all_data:
        combined_df = pd.DataFrame({
            'epoch': all_data['Metrics/PSNR']['epoch'],
            'psnr': all_data['Metrics/PSNR']['value'],
            'ssim': all_data['Metrics/SSIM']['value']
        })
        
        # Add loss data if available
        if 'Finetune_Loss_G' in all_data:
            combined_df['generator_loss'] = all_data['Finetune_Loss_G']['value']
        elif 'Pretrain_Loss_G' in all_data:
            combined_df['generator_loss'] = all_data['Pretrain_Loss_G']['value']
        
        if 'Finetune_Loss_D' in all_data:
            combined_df['discriminator_loss'] = all_data['Finetune_Loss_D']['value']
        
        combined_csv_path = os.path.join(output_dir, 'training_metrics.csv')
        combined_df.to_csv(combined_csv_path, index=False)
        print(f"✓ Combined metrics saved to: {combined_csv_path}")

def main():
    """Main function to create training plots."""
    print("SRGAN V2 Training Plot Generator")
    print("=" * 40)
    
    # Extract data from TensorBoard logs
    log_dir = 'logs_v2'
    data = extract_tensorboard_data(log_dir)
    
    if data:
        # Create plots
        create_training_plots(data)
        
        # Print summary
        print("\n" + "=" * 40)
        print("TRAINING SUMMARY")
        print("=" * 40)
        
        if 'Metrics/PSNR' in data:
            final_psnr = data['Metrics/PSNR']['values'][-1]
            best_psnr = max(data['Metrics/PSNR']['values'])
            print(f"Final PSNR: {final_psnr:.2f} dB")
            print(f"Best PSNR: {best_psnr:.2f} dB")
        
        if 'Metrics/SSIM' in data:
            final_ssim = data['Metrics/SSIM']['values'][-1]
            best_ssim = max(data['Metrics/SSIM']['values'])
            print(f"Final SSIM: {final_ssim:.4f}")
            print(f"Best SSIM: {best_ssim:.4f}")
        
        print(f"Total training epochs: {len(data.get('Metrics/PSNR', {}).get('steps', []))}")
        print("=" * 40)
    else:
        print("❌ No training data found!")
        print("Make sure you have completed training and have TensorBoard logs in the logs_v2 directory.")

if __name__ == "__main__":
    main() 