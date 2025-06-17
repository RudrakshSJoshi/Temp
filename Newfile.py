import matplotlib.pyplot as plt
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Epoch range
epochs = np.arange(1, 51)

# Train Loss: Monotonically decreasing hyperbolic-like with light noise
train_loss = 5 / (0.3 * epochs + 1) + 0.05 * np.random.randn(50)
train_loss = np.maximum.accumulate(train_loss[::-1])[::-1]  # force monotonic decrease

# Train Accuracy: Sigmoid-like rise with mild noise
train_acc = 0.238 + (0.78064 - 0.238) * (1 - np.exp(-0.15 * epochs))
train_acc[46] = 0.78064  # precise value at epoch 47
train_acc += np.random.normal(0, 0.01, size=train_acc.shape)  # mild random noise

# Test Accuracy: Rises to 0.62498 by epoch 6, then decays with fluctuations
test_acc = np.zeros(50)
test_acc[:6] = np.linspace(0.34115, 0.62498, 6)
decay = np.linspace(0.62498, 0.57779, 44)
test_acc[6:] = decay + np.random.normal(0, 0.015, 44)  # realistic troughs

# Validation Accuracy: Similar to test but spikier
val_acc = test_acc - 0.01 + np.random.normal(0, 0.02, size=test_acc.shape)

# Plotting
plt.figure(figsize=(12, 8))

# Plot Train Loss
plt.subplot(2, 1, 1)
plt.plot(epochs, train_loss, label='Train Loss', color='red')
plt.title('Training Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot Accuracies
plt.subplot(2, 1, 2)
plt.plot(epochs, train_acc, label='Train Accuracy', color='green')
plt.plot(epochs, test_acc, label='Test Accuracy', color='blue')
plt.plot(epochs, val_acc, label='Validation Accuracy', color='orange')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_plot_updated.png')  # Save as PNG
plt.show()
