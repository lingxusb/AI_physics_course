"""
Simplified MNIST Example with PyTorch
A streamlined version for teaching that runs quickly in notebooks
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available and set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load a tiny subset of MNIST dataset
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Use only 1000 samples (much faster!)
X = X[:1000]
y = y[:1000]

# Normalize pixel values to [0, 1]
X = X / 255.0

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert DataFrame to NumPy and reshape for CNN input
X_train = X_train.to_numpy().reshape(-1, 1, 28, 28)
X_test = X_test.to_numpy().reshape(-1, 1, 28, 28)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)

# Create data loaders with smaller batch size
batch_size = 32
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# Define a smaller CNN model
class TinyCNN(nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Simplified architecture with fewer parameters
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Reduced filters from 8 to 4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Reduced filters from 16 to 8
        self.fc1 = nn.Linear(8 * 7 * 7, 32)  # Reduced hidden units from 128 to 32
        self.fc2 = nn.Linear(32, 10)  # Output layer (10 digits)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Second conv block
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # Flatten
        x = x.view(-1, 8 * 7 * 7)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


# Initialize the model
model = TinyCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop - just 3 epochs
num_epochs = 3
train_losses = []
test_accuracies = []


def compute_accuracy(model, data_loader, device):
    """Calculate accuracy on a dataset"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


# Simplified training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Calculate metrics once per epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)

    test_acc = compute_accuracy(model, test_loader, device)
    test_accuracies.append(test_acc)

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Test Acc: {test_acc:.4f}')

# Plot training history
plt.figure(figsize=(12, 4))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()


# Visualize some predictions (limited to 6 samples)
def visualize_predictions(model, data_loader, num_samples=6):
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)

    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    # Plot images with predictions
    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    axes = axes.flatten()

    for i in range(num_samples):
        img = images[i].cpu().numpy().reshape(28, 28)
        axes[i].imshow(img, cmap='gray')
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        color = 'green' if true_label == pred_label else 'red'
        axes[i].set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


# Show predictions
visualize_predictions(model, test_loader)

# Print final accuracy
test_accuracy = compute_accuracy(model, test_loader, device)
print(f'Final Test Accuracy: {test_accuracy:.4f}')