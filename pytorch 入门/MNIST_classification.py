"""
Example 3: Image Classification with PyTorch using MNIST Subset
This example demonstrates how to implement a simple convolutional neural network (CNN)
for image classification using PyTorch. We'll use a small subset of the MNIST dataset
to speed up training while still demonstrating the concepts.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset from sklearn
# Use verbose=0 to suppress progress messages
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Use only the first 5000 samples to make the example faster
X = X[:5000]
y = y[:5000]

# Normalize pixel values to the range [0, 1]
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert DataFrame to NumPy array and then reshape for CNN input (batch_size, channels, height, width)
X_train = X_train.to_numpy().reshape(-1, 1, 28, 28)
X_test = X_test.to_numpy().reshape(-1, 1, 28, 28)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train.values)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test.values)

# Create data loaders
batch_size = 64
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a simple CNN model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # First convolutional layer
        # input: 1 channel (grayscale), output: 8 feature maps, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        
        # Second convolutional layer
        # input: 8 feature maps, output: 16 feature maps, 3x3 kernel
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        
        # Fully connected layers
        # After two 2x2 max pooling operations, the image size is reduced to 7x7
        self.fc1 = nn.Linear(16 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)  # 10 classes (digits 0-9)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # First convolutional block
        x = self.conv1(x)        # Apply conv layer
        x = F.relu(x)            # Apply ReLU activation
        x = F.max_pool2d(x, 2)   # Apply max pooling (reduces size by half)
        
        # Second convolutional block
        x = self.conv2(x)        # Apply conv layer
        x = F.relu(x)            # Apply ReLU activation
        x = F.max_pool2d(x, 2)   # Apply max pooling (reduces size by half)
        
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 7 * 7)
        
        # First fully connected layer with dropout
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Output layer
        x = self.fc2(x)
        
        # We don't apply softmax here because it's included in our loss function
        return x

# Initialize the model
model = SimpleCNN().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

def compute_accuracy(model, data_loader, device):
    """Function to calculate accuracy on a dataset"""
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    train_losses.append(epoch_loss)
    
    # Calculate accuracies
    train_acc = compute_accuracy(model, train_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)
    
    # Calculate test loss
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    test_losses.append(test_loss)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, '
          f'Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

# Plot training and validation history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Training and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Training and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to visualize predictions
def visualize_predictions(model, data_loader, num_samples=10):
    """Function to visualize model predictions"""
    model.eval()
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Get predictions
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 6))
    for i in range(num_samples):
        ax = fig.add_subplot(2, 5, i+1)
        # Convert image from tensor to numpy and reshape
        img = images[i].cpu().numpy().reshape(28, 28)
        ax.imshow(img, cmap='gray')
        true_label = labels[i].item()
        pred_label = predicted[i].item()
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize some predictions
visualize_predictions(model, test_loader)

# Evaluate the final model
test_accuracy = compute_accuracy(model, test_loader, device)
print(f'Final Test Accuracy: {test_accuracy:.4f}')

# Function to visualize feature maps
def visualize_feature_maps(model, image_tensor):
    """Function to visualize feature maps for a single image"""
    # Get the feature maps from the first conv layer
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)  # Add batch dimension
        
        # Get activation of the first conv layer
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        
        # Register hook to get activations
        handle = model.conv1.register_forward_hook(get_activation('conv1'))
        
        # Forward pass
        output = model(image_tensor)
        
        # Remove the hook
        handle.remove()
        
        # Get the feature maps
        feature_maps = activation['conv1'].cpu().squeeze(0)
        
    # Plot the original image and its feature maps
    fig = plt.figure(figsize=(12, 8))
    
    # Plot original image
    ax = fig.add_subplot(3, 3, 1)
    ax.imshow(image_tensor.cpu().squeeze().numpy(), cmap='gray')
    ax.set_title('Original Image')
    ax.axis('off')
    
    # Plot feature maps
    for i in range(min(8, feature_maps.size(0))):
        ax = fig.add_subplot(3, 3, i+2)
        ax.imshow(feature_maps[i].numpy(), cmap='viridis')
        ax.set_title(f'Feature Map {i+1}')
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# Visualize feature maps for a sample image
sample_image = X_test_tensor[0]
visualize_feature_maps(model, sample_image)
