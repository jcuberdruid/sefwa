import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split

#locally source imports 

#from .. import paths
from cl.userExtension.subject import Subject
from cl.userExtension.epoch import Epoch

class EEGDataHandler:
	def __init__(self, filepath, load_percentage=None):
		self.loadEpochs(filepath, load_percentage)
	def loadEpochs(self, filepath, load_percentage):
		categoryName = (os.path.basename(os.path.normpath(filepath))).replace(".csv", "")
		if load_percentage is not None:
			with open(filepath) as file:
				total_rows = sum(1 for line in file) - 1  # Subtract 1 for the header
			nrows = int(total_rows * load_percentage)
			data = pd.read_csv(filepath, nrows=nrows)
		else:
			data = pd.read_csv(filepath)
		grouped = data.groupby(['subject', 'run', 'epoch'])
		print("EEGDataHandler->LoadEpochs: Function should be modified if number of subjects exceeds 256")
		subjectArray = [0] * 256
		for _, group in grouped:
			print(f"group shape {group.shape}")
			if group.shape == (641, 69):
				subject_index = group.iloc[0]['subject']  # Assuming 'subject' is a column
				if subjectArray[subject_index-1] == 0:
					subjectArray[subject_index-1] = Subject(group.iloc[0]['subject'], categoryName)
				channel_dict = {}
				for channel in group.columns[4:][1:]:
					channel_dict[channel] = group[channel].values
				epoch = Epoch(group.iloc[0]['run'], group.iloc[0]['epoch'], channel_dict)
				subjectArray[subject_index-1].epochs.append(epoch)
		subjectArray = [subj for subj in subjectArray if subj != 0]
		self.subjects = subjectArray

	def split_data(self, train_size=0.8, test_size=0.1, random_state=None):
		print("EEGDateHandler->split_data: This function will not work currently sorry")
		exit(0)
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
