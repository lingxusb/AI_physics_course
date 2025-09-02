"""
Example 1: Linear Regression with PyTorch
This example demonstrates how to implement a simple linear regression model using PyTorch.
We'll generate some synthetic data, train a linear model, and visualize the results.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
X = np.random.rand(100, 1) * 10  # 100 samples with 1 feature
y = 2 * X + 1 + np.random.randn(100, 1) * 1.5  # y = 2x + 1 + noise

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test_scaled)
y_test_tensor = torch.FloatTensor(y_test)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        # Define a linear layer with input_dim inputs and 1 output
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # Simple linear function
        return self.linear(x)

# Initialize the model
input_dim = 1  # We have 1 feature
model = LinearRegressionModel(input_dim)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error loss
optimizer = optim.SGD(model.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimization
    optimizer.zero_grad()  # Clear previous gradients
    loss.backward()        # Compute gradients
    optimizer.step()       # Update parameters
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():  # No need to track gradients during evaluation
    model.eval()       # Set the model to evaluation mode
    y_pred_test = model(X_test_tensor)
    test_loss = criterion(y_pred_test, y_test_tensor)
    print(f'Test Loss: {test_loss.item():.4f}')

# Visualize the results
plt.figure(figsize=(10, 6))

# Plot training data
plt.scatter(X_train, y_train, color='blue', label='Training data')

# Plot test data
plt.scatter(X_test, y_test, color='red', label='Test data')

# Plot the regression line
# We need to convert our standardized predictions back to the original scale
X_range = np.linspace(0, 10, 100).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
X_range_tensor = torch.FloatTensor(X_range_scaled)
with torch.no_grad():
    y_range_pred = model(X_range_tensor).numpy()

plt.plot(X_range, y_range_pred, color='green', linewidth=2, label='Regression line')

plt.title('Linear Regression with PyTorch')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Get the learned parameters
w = model.linear.weight.item()
b = model.linear.bias.item()
print(f"Learned parameters: w = {w:.4f}, b = {b:.4f}")
print(f"True parameters: w = 2, b = 1")

# Compute R-squared score manually
with torch.no_grad():
    y_test_mean = torch.mean(y_test_tensor)
    ss_tot = torch.sum((y_test_tensor - y_test_mean) ** 2)
    ss_res = torch.sum((y_test_tensor - y_pred_test) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    print(f"R-squared: {r2.item():.4f}")
