"""
Example 2: Binary Classification with PyTorch
This example demonstrates how to implement a simple neural network for binary classification using PyTorch.
We'll use the sklearn.datasets to load a binary classification dataset, train a small neural network,
and visualize the decision boundary and training progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate a synthetic dataset - moons dataset is a classic binary classification problem
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Reshape to match PyTorch expectations

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Define the neural network model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BinaryClassifier, self).__init__()
        # A simple feedforward neural network with one hidden layer
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()  # Sigmoid activation for binary output
        
    def forward(self, x):
        # Forward pass through the network
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)
        return x

# Initialize the model
input_dim = 2      # Number of features (2D dataset)
hidden_dim = 10    # Number of hidden neurons
model = BinaryClassifier(input_dim, hidden_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
num_epochs = 200
losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    with torch.no_grad():
        y_train_pred_class = (y_pred >= 0.5).float()
        train_acc = (y_train_pred_class == y_train_tensor).float().mean()
        train_accuracies.append(train_acc.item())
        
        # Calculate test accuracy
        y_test_pred = model(X_test_tensor)
        y_test_pred_class = (y_test_pred >= 0.5).float()
        test_acc = (y_test_pred_class == y_test_tensor).float().mean()
        test_accuracies.append(test_acc.item())
    
    # Print progress
    if (epoch+1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, '
              f'Train Accuracy: {train_acc.item():.4f}, Test Accuracy: {test_acc.item():.4f}')

# Plot training history
plt.figure(figsize=(12, 5))

# Plot loss
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)

# Plot accuracy
plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    # Define mesh grid of points to evaluate the model
    h = 0.02  # Step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Predict for all points in the mesh grid
    Z = np.c_[xx.ravel(), yy.ravel()]
    Z_scaled = scaler.transform(Z)
    Z_tensor = torch.FloatTensor(Z_scaled)
    
    with torch.no_grad():
        Z_pred = model(Z_tensor).numpy()
    Z_pred = Z_pred.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z_pred, cmap=plt.cm.Spectral, alpha=0.8)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.grid(True)
    plt.show()

# Evaluate the model on the test set
with torch.no_grad():
    model.eval()
    y_test_pred = model(X_test_tensor)
    y_test_pred_class = (y_test_pred >= 0.5).float()
    test_acc = (y_test_pred_class == y_test_tensor).float().mean()
    print(f'Final Test Accuracy: {test_acc.item():.4f}')

# Plot the decision boundary
plot_decision_boundary(model, X, y)
