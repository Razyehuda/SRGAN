# test_examples.py
# Example usage of test_v2_model.py

import os
import subprocess
import sys

def run_test_command(command):
    """Run a test command and print the result."""
    print(f"\n{'='*60}")
    print(f"Running: {command}")
    print('='*60)
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        print(f"Return code: {result.returncode}")
    except Exception as e:
        print(f"Error running command: {e}")

def main():
    print("SRGAN V2 Model Testing Examples")
    print("="*40)
    
    # Example 1: Test a single image
    print("\n1. Testing a single image:")
    command1 = (
        "python test_v2_model.py "
        "--checkpoint checkpoints_v2/best_model_finetune.pth "
        "--mode single "
        "--input dummy_data/dummy.png "
        "--output test_output_single.png "
        "--compare"
    )
    run_test_command(command1)
    
    # Example 2: Evaluate on validation set
    print("\n2. Evaluating on validation set:")
    command2 = (
        "python test_v2_model.py "
        "--checkpoint checkpoints_v2/best_model_finetune.pth "
        "--mode validation "
        "--output validation_results "
        "--batch_size 8 "
        "--patch_size 128"
    )
    run_test_command(command2)
    
    # Example 3: Test with HR images (create LR from HR and test)
    print("\n3. Testing with HR images:")
    command3 = (
        "python test_v2_model.py "
        "--checkpoint checkpoints_v2/best_model_finetune.pth "
        "--mode hr_test "
        "--input dummy_data/ "
        "--output hr_test_results "
        "--max_size 512"
    )
    run_test_command(command3)
    
    # Example 4: Batch processing
    print("\n4. Batch processing:")
    command4 = (
        "python test_v2_model.py "
        "--checkpoint checkpoints_v2/best_model_finetune.pth "
        "--mode batch "
        "--input dummy_data/ "
        "--output batch_results "
        "--max_size 512"
    )
    run_test_command(command4)
    
    # Example 5: Test pretrain model
    print("\n5. Testing pretrain model:")
    command5 = (
        "python test_v2_model.py "
        "--checkpoint checkpoints_v2/best_model_pretrain.pth "
        "--mode validation "
        "--output pretrain_validation_results "
        "--batch_size 8 "
        "--patch_size 128"
    )
    run_test_command(command5)

if __name__ == '__main__':
    main() 