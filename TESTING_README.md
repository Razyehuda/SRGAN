# SRGAN V2 Model Testing Guide

This guide explains how to test your trained SRGAN models from `train_v2.py`.

## Quick Start

### 1. Test a Single Image
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode single \
    --input path/to/your/image.jpg \
    --output output_sr_image.png \
    --compare
```

### 2. Evaluate on Validation Set
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode validation \
    --output validation_results \
    --batch_size 8
```

### 3. Test with HR Images (Create LR from HR)
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode hr_test \
    --input path/to/hr/images/ \
    --output hr_test_results \
    --max_size 512
```

## Testing Modes

### 1. Single Image Mode (`--mode single`)
- Process one image at a time
- Good for testing specific images
- Use `--compare` for side-by-side comparison

**Example:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode single \
    --input test_image.jpg \
    --output sr_test_image.png \
    --compare
```

### 2. Batch Mode (`--mode batch`)
- Process all images in a directory
- Good for processing multiple images

**Example:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode batch \
    --input input_images/ \
    --output output_images/ \
    --max_size 512
```

### 3. Validation Mode (`--mode validation`)
- Evaluate model on validation set
- Calculate PSNR and SSIM metrics
- Uses the same validation set as training

**Example:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode validation \
    --output validation_results \
    --batch_size 16 \
    --patch_size 128
```

### 4. HR Test Mode (`--mode hr_test`)
- Create low-resolution images from high-resolution images
- Test the model's ability to reconstruct HR from LR
- Creates comparisons automatically

**Example:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode hr_test \
    --input hr_images/ \
    --output hr_test_results \
    --max_size 512
```

## Available Checkpoints

Based on your training, you have these checkpoints available:

### Pretrain Models (L1 Loss Only)
- `checkpoints_v2/best_model_pretrain.pth` - Best pretrain model
- `checkpoints_v2/checkpoint_pretrain_epoch_*.pth` - Specific epochs

### Finetune Models (GAN Loss)
- `checkpoints_v2/best_model_finetune.pth` - Best finetune model
- `checkpoints_v2/checkpoint_finetune_epoch_*.pth` - Specific epochs

## Model Parameters

The test script automatically uses the same model parameters as training:
- `--num_residual_blocks 23` (default)
- `--base_channels 64` (default)

If you trained with different parameters, specify them:
```bash
python test_v2_model.py \
    --checkpoint your_checkpoint.pth \
    --num_residual_blocks 16 \
    --base_channels 32 \
    --mode single \
    --input test.jpg \
    --output output.jpg
```

## Memory Optimization

For large images or limited GPU memory:

1. **Limit image size:**
```bash
--max_size 512
```

2. **Use CPU instead of GPU:**
```bash
--device cpu
```

3. **Reduce batch size for validation:**
```bash
--batch_size 4
```

## Output Files

### Single Image Mode
- `output.png` - Super-resolved image
- `output_comparison.png` - Side-by-side comparison (if `--compare` used)

### Batch Mode
- `sr_filename.png` - Super-resolved images for each input

### Validation Mode
- `validation_results/validation_results.txt` - PSNR and SSIM metrics

### HR Test Mode
- `sr_filename.png` - Super-resolved images
- `comparison_filename.png` - Side-by-side comparisons
- `lr_temp_filename.png` - Temporary LR images (auto-deleted)

## Example Workflow

1. **Quick Test with Single Image:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode single \
    --input test.jpg \
    --output sr_test.jpg \
    --compare
```

2. **Evaluate Model Performance:**
```bash
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode validation \
    --output results \
    --batch_size 8
```

3. **Compare Pretrain vs Finetune:**
```bash
# Test pretrain model
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_pretrain.pth \
    --mode validation \
    --output pretrain_results

# Test finetune model
python test_v2_model.py \
    --checkpoint checkpoints_v2/best_model_finetune.pth \
    --mode validation \
    --output finetune_results
```

## Troubleshooting

### Memory Issues
- Use `--max_size 512` or smaller
- Use `--device cpu`
- Reduce `--batch_size`

### Model Loading Errors
- Ensure checkpoint path is correct
- Check if model parameters match training
- Verify checkpoint file is not corrupted

### Validation Set Issues
- Ensure validation HR directory path is correct
- Check if validation images exist
- Verify data loader parameters match training

## Running Examples

You can also run the example script to see all modes in action:
```bash
python test_examples.py
```

This will run all the example commands and show you the output. 