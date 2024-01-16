import csv
import matplotlib.pyplot as plt

# Index for channel 'F7' (hardcoded based on its position in your array)
channel_index = 29

# Store accuracies and slices
accuracies = []
slices = list(range(12))  # Assuming slices are numbered from 0 to 11

for slice_num in slices:
    # Construct file name
    file_name = f"results_MI_RLH_T1.csv_MI_FF_T2.csv_slice_{slice_num}"

    try:
        with open(file_name, mode='r') as file:
            csv_reader = csv.DictReader(file)
            
            # Find the row for channel 'F7' and extract accuracy
            accuracy_found = False
            for row in csv_reader:
                if int(row['Channel']) == channel_index:
                    accuracies.append(float(row['Testing Accuracy']))
                    accuracy_found = True
                    break
            
            if not accuracy_found:
                accuracies.append(None)
                print(f"Channel 'F7' not found in file: {file_name}")

    except FileNotFoundError:
        print(f"File not found: {file_name}")
        accuracies.append(None)

# Plotting
plt.bar(slices, accuracies, color='blue')
plt.xlabel('Slice Number')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy for Channel F7 Across Different Slices')
plt.xticks(slices)
plt.show()

