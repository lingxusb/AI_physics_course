import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint

def lorenz(state, t, sigma=10, beta=8/3, rho=28):
    """The Lorenz equations."""
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return [dx, dy, dz]

# Time points
t = np.linspace(0, 40, 4000)

# Initial condition
initial_state = [0.1, 0.0, 0.0]

# Solve the system
states = odeint(lorenz, initial_state, t)

# Extract the states
x = states[:, 0]
y = states[:, 1]
z = states[:, 2]

# Create the figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Set axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Lorenz Attractor')

# Set the limits based on data to keep view stable
ax.set_xlim([min(x), max(x)])
ax.set_ylim([min(y), max(y)])
ax.set_zlim([min(z), max(z)])

# Initialize an empty line plot
line, = ax.plot([], [], [], lw=0.5, color='blue')

# Function to initialize the animation
def init():
    line.set_data([], [])
    line.set_3d_properties([])
    return line,

# Animation function that will be called for each frame
def animate(i):
    # Set the line data up to the current frame
    # Increase this number to speed up the animation
    frame = i * 10
    # Ensure we don't go out of bounds
    max_frame = min(frame, len(x))
    line.set_data(x[:max_frame], y[:max_frame])
    line.set_3d_properties(z[:max_frame])
    # Rotate the view slightly for a 3D effect
    ax.view_init(30, i / 5)
    return line,

# Create the animation
ani = FuncAnimation(fig, animate, frames=400, init_func=init, interval=20, blit=True)

# Save the animation as a movie file
ani.save('lorenz_attractor.gif', writer='pillow', fps=30, dpi=100)

# To display in a notebook, use:
# from IPython.display import HTML
# HTML(ani.to_html5_video())

# For interactive display
plt.show()
