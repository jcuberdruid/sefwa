import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.colors as mcolors

# Assuming there are N channels
N = 64  # Update this with the actual number of channels

# Store accuracies
# It's now a 2D list: accuracies[slice][channel]
accuracies = [[] for _ in range(20)]  # Assuming 20 slices

for slice_num in range(20):
    file_name = f"results_MI_RLH_T2.csv_MI_RLH_T1.csv_slice_{slice_num}"

    try:
        with open(file_name, mode='r') as file:
            csv_reader = csv.DictReader(file)

            # Initialize a temporary list for this slice
            accuracies_slice = [None] * N

            for row in csv_reader:
                try:
                    # Attempt to convert the channel to an integer
                    channel = int(row['Channel']) - 1  # Subtract 1 if channel numbers start at 1
                    if 0 <= channel < N:
                        accuracies_slice[channel] = float(row['Testing Accuracy'])
                except ValueError:
                    # Skip rows where the channel is not a valid number
                    continue

            accuracies[slice_num] = accuracies_slice

    except FileNotFoundError:
        print(f"File not found: {file_name}")
        accuracies[slice_num] = [None] * N

# Create a mesh for slices and channels
slice_indices, channel_indices = np.meshgrid(np.arange(20), np.arange(N))

# 3D Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

## Flatten the mesh for plotting
slice_indices = slice_indices.flatten()
channel_indices = channel_indices.flatten()

# Convert accuracies to a numpy array and replace 'None' with a placeholder (e.g., 0)
accuracies_array = np.array(accuracies, dtype=float)
accuracies_array = np.nan_to_num(accuracies_array, nan=0)  # Replace 'nan' (converted from 'None') with 0

accuracies_flat = accuracies_array.flatten()

# Ensure the arrays are of the same length
assert len(slice_indices) == len(channel_indices) == len(accuracies_flat)

# Normalize the accuracies for the colormap
norm = mcolors.Normalize(vmin=accuracies_flat.min(), vmax=accuracies_flat.max())

# Create a scalar mappable object with the chosen colormap
colormap = plt.cm.viridis
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])

# Plot each bar with color mapped from the accuracy
for slc, chn, acc in zip(slice_indices, channel_indices, accuracies_flat):
    ax.bar3d(slc, chn, 0, 1, 1, acc, color=colormap(norm(acc)))

# Adding a color bar
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Accuracy')

plt.show()

