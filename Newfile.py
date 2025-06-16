def plot_training_curves(train_losses, val_metrics, test_metrics):
    plt.figure(figsize=(18, 6))
    
    # Plot 1: Training Metrics
    plt.subplot(1, 3, 1)
    # Training Loss (blue)
    plt.plot(train_losses, color='blue', label='Training Loss')
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Validation Metrics
    plt.subplot(1, 3, 2)
    # Absolute Accuracy (orange) - highest priority
    plt.plot(val_metrics['abs_acc'], color='orange', label='Val AbsAcc', linewidth=2.5)
    # F1 Score (yellow)
    plt.plot(val_metrics['f1'], color='yellow', label='Val F1', linewidth=2)
    # Accuracy (green)
    plt.plot(val_metrics['acc'], color='green', label='Val Acc', linewidth=1.5)
    plt.title('Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    # Plot 3: Test Metrics
    plt.subplot(1, 3, 3)
    # Absolute Accuracy (indigo) - highest priority
    plt.plot(test_metrics['abs_acc'], color='indigo', label='Test AbsAcc', linewidth=2.5)
    # F1 Score (red)
    plt.plot(test_metrics['f1'], color='red', label='Test F1', linewidth=2)
    # Accuracy (purple)
    plt.plot(test_metrics['acc'], color='purple', label='Test Acc', linewidth=1.5)
    plt.title('Test Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
