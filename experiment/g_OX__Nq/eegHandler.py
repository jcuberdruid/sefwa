import pandas as pd
from multiprocessing import Pool
import numpy as np
from sklearn.model_selection import train_test_split
#from PyEMD import EMD 
from PyEMD import EMD

import scipy.signal as signal

def compute_hht(channel_data):
	emd = EMD()
	max_length = 5 * 160  # Maximum possible length (5 IMFs, each of length 40)

	IMFs = emd(channel_data)
	selected_IMFs = IMFs[:5] if len(IMFs) >= 5 else IMFs
	hht_result = np.concatenate([np.abs(signal.hilbert(imf)) for imf in selected_IMFs])

	if hht_result.size < max_length:
		hht_result = np.pad(hht_result, (0, max_length - hht_result.size), 'constant')

	return hht_result

class EEGDataHandler:
	def __init__(self, filepath, slice_index, load_percentage=None):
		print(f"EEGDaraHandler->Init: slice_index = {slice_index}")
		self.loadEpochs(filepath, slice_index, load_percentage)
	def loadEpochs(self, filepath, slice_index, load_percentage):
		print(f"EEGDataHandler->loadEpochs: slice_index = {slice_index}")

		# Load data
		if load_percentage is not None:
			total_rows = sum(1 for line in open(filepath)) - 1
			nrows = int(total_rows * load_percentage)
			data = pd.read_csv(filepath, nrows=nrows)
		else:
			data = pd.read_csv(filepath)

		# Keep the grouping columns and the numeric data separately
		grouping_columns = ['subject', 'run', 'epoch']
		numeric_columns = data.columns[5:]  # Assuming these are your EEG data columns

		grouped = data.groupby(grouping_columns)
		arrays = []

		for _, group in grouped:
			group_numeric_data = group[numeric_columns].values  # Only use numeric data for processing
			if group_numeric_data.shape == (961, len(numeric_columns)):
				start = 1 + (slice_index * 160)
				segment = group_numeric_data[start:start+160]

				# Process each channel separately
				channel_data_list = segment.T
				# Create a multiprocessing pool with 12 workers
				with Pool(12) as pool:
						channel_results = pool.map(compute_hht, channel_data_list)

				combined_result = np.stack(channel_results, axis=-1)  # Stack along a new last axis
				arrays.append(combined_result)


		# Stack and check the results
		if all(a.shape[0] == arrays[0].shape[0] for a in arrays):
			self.data = np.stack(arrays)
		else:
			print("Groups have different number of rows, cannot stack into a 3D array.")
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
