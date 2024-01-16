import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

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
channel_names = ["FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "FP1", "FPZ", "FP2", "AF7", "AF3", "AFZ", "AF4", "AF8", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7", "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "PO7", "PO3", "POZ", "PO4", "PO8", "O1", "OZ", "O2", "IZ"]

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

# Plot
ax.bar3d(slice_indices, channel_indices, np.zeros_like(accuracies_flat), 
         dx=0.8, dy=0.8, dz=accuracies_flat, shade=True)

ax.set_xlabel('Slice Number')
ax.set_ylabel('Channel Number')
ax.set_zlabel('Testing Accuracy')
ax.set_title('Testing Accuracy Across Different Slices and Channels')

# Set the y-ticks to channel names
ax.set_yticks(np.arange(N))
ax.set_yticklabels(channel_names, fontsize=8, rotation=45)

plt.show()

