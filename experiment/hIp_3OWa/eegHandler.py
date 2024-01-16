import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class EEGDataHandler:
    def __init__(self, filepath, slice_index, load_percentage=None):
        print(f"EEGDaraHandler->Init: slice_index = {slice_index}")
        self.loadEpochs(filepath, slice_index, load_percentage)
    def loadEpochs(self, filepath, slice_index, load_percentage):
        print(f"EEGDataHandler->loadEpochs: slice_index = {slice_index}")
        if load_percentage is not None:
            total_rows = sum(1 for line in open(filepath)) - 1  # Subtract 1 for the header
            nrows = int(total_rows * load_percentage)
            data = pd.read_csv(filepath, nrows=nrows)
        else:
            data = pd.read_csv(filepath)
        grouped = data.groupby(['subject', 'run', 'epoch'])
        arrays = []
        for _, group in grouped:
            group_array = group.values
            if group.shape == (961, 69):
                start = 1 + (slice_index * 40)
                arrays.append(group_array[start:start+40])
        
        if all(a.shape[0] == arrays[0].shape[0] for a in arrays):
            self.data = np.stack(arrays)
        else:
            print("Groups have different number of rows, cannot stack into a 3D array without padding or truncation.")
            exit(0)

        print(self.data.shape)

    def split_data(self, train_size=0.8, test_size=0.1, random_state=None):
        unique_subjects = np.unique(self.data[:, :, 1])

        if random_state is not None:
            np.random.seed(random_state)
        np.random.shuffle(unique_subjects)

        num_subjects = len(unique_subjects)
        num_train = int(num_subjects * train_size)
        num_test = int(num_subjects * test_size)
        num_val = num_subjects - num_train - num_test

        train_subjects = unique_subjects[:num_train]
        val_subjects = unique_subjects[num_train:num_train + num_val]
        test_subjects = unique_subjects[num_train + num_val:]

        def extract_data(subjects):
            mask = np.isin(self.data[:, 0, 1], subjects)
            return self.data[mask]

        train_data = extract_data(train_subjects)
        val_data = extract_data(val_subjects)
        test_data = extract_data(test_subjects)

        return train_data, val_data, test_data
'''
handler = EEGDataHandler("../frequency_band_experiment/classes/MM_RLH_T1.csv")
train_data, val_data, test_data = handler.split_data()

# Print shapes
print("Training Data Shape:", train_data.shape)
print("Validation Data Shape:", val_data.shape)
print("Testing Data Shape:", test_data.shape)
'''
