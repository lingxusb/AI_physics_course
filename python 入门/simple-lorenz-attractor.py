import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def lorenz(state, t, sigma=10, beta=8/3, rho=28):
    """The Lorenz equations."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Time points
t = np.linspace(0, 40, 10000)

# Initial condition
initial_state = [0.1, 0.0, 0.0]

# Solve the system
states = odeint(lorenz, initial_state, t)

# Extract the states
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

# Create a 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the trajectory
ax.plot(x, y, z, lw=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')

plt.tight_layout()
plt.show()
