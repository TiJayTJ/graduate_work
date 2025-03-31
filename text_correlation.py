import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load the image
image = Image.open("images/lena.png")
image_array = np.array(image)

# Split channels
r_channel, g_channel, b_channel = (image_array[192:320, 192:320, 0],
                                   image_array[192:320, 192:320, 1],
                                   image_array[192:320, 192:320, 2])

channels = [r_channel, g_channel, b_channel]
colors = ['r', 'g', 'b']

# Plot settings
fig = plt.figure(figsize=(20, 5))

# Correlation analysis
for idx, (channel, color) in enumerate(zip(channels, colors)):
    # Get pixel values and adjacent pixel values in three directions
    h_adj = channel[:, :-1].flatten(), channel[:, 1:].flatten()
    v_adj = channel[:-1, :].flatten(), channel[1:, :].flatten()
    d_adj = channel[:-1, :-1].flatten(), channel[1:, 1:].flatten()

    pixel_values = np.concatenate([h_adj[0], v_adj[0], d_adj[0]])
    adjacent_values = np.concatenate([h_adj[1], v_adj[1], d_adj[1]])
    directions = np.concatenate([np.zeros_like(h_adj[0]), np.ones_like(v_adj[0]), 2 * np.ones_like(d_adj[0])])

    # 3D Scatter plot
    ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
    ax.scatter(directions, pixel_values, adjacent_values, c=color, s=1)
    ax.set_xlabel('Направление')
    ax.set_ylabel('Значение пикселя')
    ax.set_zlabel('Значение соседнего пикселя')
    ax.set_title(f'Корреляция {color.upper()}-канала')

plt.tight_layout()
plt.show()
