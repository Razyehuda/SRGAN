import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

def test_cosine_annealing():
    """Test the CosineAnnealingLR scheduler implementation."""
    
    # Create a simple model
    model = torch.nn.Linear(10, 1)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # Create CosineAnnealingLR scheduler
    num_epochs = 100
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=num_epochs,
        eta_min=1e-4 * 0.01  # 1% of initial LR
    )
    
    # Track learning rates
    learning_rates = []
    
    print("Testing CosineAnnealingLR scheduler...")
    print(f"Initial LR: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Minimum LR: {1e-4 * 0.01:.2e}")
    print(f"Number of epochs: {num_epochs}")
    print("\nEpoch | Learning Rate")
    print("-" * 25)
    
    for epoch in range(num_epochs):
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        if epoch % 10 == 0:
            print(f"{epoch:5d} | {current_lr:.2e}")
        
        # Step the scheduler
        scheduler.step()
    
    print(f"{num_epochs:5d} | {learning_rates[-1]:.2e}")
    
    # Plot the learning rate curve
    epochs = list(range(num_epochs))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, learning_rates, 'b-', linewidth=2, label='CosineAnnealingLR')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('CosineAnnealingLR Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('cosine_annealing_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nLearning rate plot saved as 'cosine_annealing_test.png'")
    print(f"Final LR: {learning_rates[-1]:.2e}")
    print(f"Expected minimum LR: {1e-4 * 0.01:.2e}")
    
    # Verify the cosine annealing behavior
    expected_final_lr = 1e-4 * 0.01
    actual_final_lr = learning_rates[-1]
    
    # Allow for small numerical precision differences
    tolerance = 1e-7
    if abs(actual_final_lr - expected_final_lr) < tolerance:
        print("✓ CosineAnnealingLR test passed!")
    else:
        print(f"✗ CosineAnnealingLR test failed! Expected {expected_final_lr:.2e}, got {actual_final_lr:.2e}")
    
    # Check that the learning rate decreases smoothly
    lr_decreasing = all(learning_rates[i] >= learning_rates[i+1] for i in range(len(learning_rates)-1))
    if lr_decreasing:
        print("✓ Learning rate decreases monotonically!")
    else:
        print("✗ Learning rate does not decrease monotonically!")
    
    # Check that the learning rate starts at the initial value
    if abs(learning_rates[0] - 1e-4) < 1e-8:
        print("✓ Learning rate starts at the correct initial value!")
    else:
        print(f"✗ Learning rate does not start at the correct value! Expected 1.00e-04, got {learning_rates[0]:.2e}")

if __name__ == '__main__':
    test_cosine_annealing() 