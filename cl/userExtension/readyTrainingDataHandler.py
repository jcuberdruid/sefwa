import pickle
import os
import glob
import numpy as np
from datetime import datetime

from cl import paths
from cl.ut.sds import ExperimentManager
'''
	#as hyper parameter tuning is a key point of repetition regenerating the training data each time 
	is prohibitively expensive especially for things like CSP and HHT. Therefore the  
'''
class ReadyTrainingDataHandler:
	def __init__(self):
		self.id = None
		self.data = None
		self.csp_filter_dict = {}
	def create_kfolds(self, subjects_class_1, subjects_class_2, n_folds):
		self.data = [[], []]
		max_splits = min(len(subjects_class_1), len(subjects_class_2), n_folds)
		subjects_class_1.sort(key=lambda x: x.number)
		subjects_class_2.sort(key=lambda x: x.number)
		split_sizes = np.linspace(0, len(subjects_class_1), max_splits + 1, dtype=int)
		for i in range(max_splits):
			start_idx, end_idx = split_sizes[i], split_sizes[i+1]
			fold_subjects_class_1 = subjects_class_1[start_idx:end_idx]
			fold_subjects_class_2 = subjects_class_2[start_idx:end_idx]
			self.data[0].append(fold_subjects_class_1)
			self.data[1].append(fold_subjects_class_2)

	def remove_channels_from_epochs(self, channels_to_remove):
		"""
		Remove specified channels from all epochs of all subjects in self.data.

		:param channels_to_remove: List of channel names (strings) to be removed.
		"""
		# Iterate over both classes in self.data
		for class_group in self.data:
			# Iterate over each fold
			for subjects in class_group:
				# Iterate over each subject
				for subject in subjects:
					# Remove channels from each subject
					self.remove_channels_from_subject(subject, channels_to_remove)

	@staticmethod
	def remove_channels_from_subject(subject, channels_to_remove):
		"""
		Remove specified channels from all epochs of a single subject.

		:param subject: Subject object.
		:param channels_to_remove: List of channel names (strings) to be removed.
		"""
		# Assuming subject has an attribute 'epochs' which is a list of epoch objects
		# and each epoch object has a dictionary 'channels_dict' with channel data
		for epoch in subject.epochs:
			for channel in channels_to_remove:
				if channel in epoch.channels_dict:
					del epoch.channels_dict[channel]

	def saveTrainingSet(self):
		if self.data == None: 
			print("ReadyTrainingDataHandler->saveTrainingSet: Error data not prepared, no data to save") 
			exit(1)
		exp_manager = ExperimentManager()
		exp_manager.new_data('name', 'description')
		if hasattr(exp_manager, 'uid'):
			self.id = exp_manager.uid
		else:
			print("ReadyTrainingDataHandler->saveTrainingSet: Error experiment manager not exposing uid")
			exit(1)
		saveFile = ("savedDataset_" + datetime.now().strftime("%Y%m%d_%H%M%S" + '.pkl'))
		savePath = os.path.join(paths.dataDir, self.id, saveFile)
		save_data = [self.csp_filter_dict, self.data]
		with open(savePath, 'wb') as file:			
			pickle.dump(save_data, file)
		return self.id
	def loadTrainingSet(self, id_to_load): 
		loadPath = os.path.join(paths.dataDir, id_to_load)
		if not os.path.isdir(loadPath):
			print(f"ReadyTrainingDataHandler->loadTrainingSet: Error no directory with id: {id_to_load}")
			exit(1)
		search_pattern = os.path.join(loadPath, 'savedDataset_*')
		files_that_match = glob.glob(search_pattern)
		if(len(files_that_match) > 1):
			print(f"ReadyTrainingDataHandler->loadTrainingSet: Error multiple savedDataset files in directory")
			exit(1)
		elif (len(files_that_match) == 0):
			print(f"ReadyTrainingDataHandler->loadTrainingSet: Error no savedDataset files in directory")
		loadPath = os.path.join(loadPath, files_that_match[0])
		with open(loadPath, 'rb') as file:
			self.data = pickle.load(file)
			self.csp_filter_dict = self.data[0]
			self.data = self.data[1]	
			self.id = id_to_load
	def kfold_set(self, kfold):
		if self.data is None or len(self.data[0]) == 0 or len(self.data[1]) == 0:
			print("ReadyTrainingDataHandler->kfold_set: Error - K-folds not created")
			exit(1)
		testing_data_class_1 = self.data[0][kfold]
		testing_data_class_2 = self.data[1][kfold]
		testing_data = np.concatenate((testing_data_class_1, testing_data_class_2))
		testing_labels = np.concatenate((np.zeros(len(testing_data_class_1)), np.ones(len(testing_data_class_2))))

		training_data_class_1 = np.concatenate([self.data[0][i] for i in range(len(self.data[0])) if i != kfold])
		training_data_class_2 = np.concatenate([self.data[1][i] for i in range(len(self.data[1])) if i != kfold])
		training_data = np.concatenate((training_data_class_1, training_data_class_2))
		training_labels = np.concatenate((np.zeros(len(training_data_class_1)), np.ones(len(training_data_class_2))))

		testing_indices = np.arange(len(testing_data))
		#np.random.shuffle(testing_indices)
		testing_data = testing_data[testing_indices]
		testing_labels = testing_labels[testing_indices]

		training_indices = np.arange(len(training_data))
		np.random.shuffle(training_indices)
		training_data = training_data[training_indices]
		training_labels = training_labels[training_indices]

		return testing_data, testing_labels, training_data, training_labels
	def kfold_csp(self, kfold, csp_object):
		self.csp_filter_dict[kfold] = csp_object

	def flatten_array(self, arr):
		return [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]

	def get_raw(self, subjects, target_keys=None):
		if target_keys == None: # bad hard coded and target keys should be singular since doesn't take a list
			target_keys = ['4-8']
			#target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']
		target_channels = list(subjects[0].epochs[0].channels_dict.keys())
		def dict_to_3d_array(dict_of_arrays):
			# Calculate total number of arrays
			list_length = sum(item.shape[0] if isinstance(item, np.ndarray) else len(item) for item in next(iter(dict_of_arrays.values())))

			# Find the maximum length of the inner arrays
			array_length = max(sub_item.shape[1] if isinstance(sub_item, np.ndarray) else len(sub_item) 
							for item in dict_of_arrays.values() 
							for sub_item in (item if isinstance(item, list) else [item]))

			# Initialize the result array
			result_array = np.empty((list_length, len(dict_of_arrays), array_length))

			for i, key in enumerate(dict_of_arrays):
				pos = 0
				for array in dict_of_arrays[key]:
					if isinstance(array, np.ndarray):
						# If the element is a numpy array
						for sub_array in array:
							result_array[pos, i, :len(sub_array)] = sub_array
							pos += 1
					elif isinstance(array, list):
						# If the element is a single list
						result_array[pos, i, :len(array)] = array
						pos += 1
					else:
						raise TypeError("Unsupported type in dict_of_arrays")

			return result_array
		raw_data_dict = {}
		for channel in target_channels:
			channel_data = []
			for subject in subjects: 
				subject.gather_all_waveforms([channel], return_high_level=True)
				for high_level_dict in subject.waveforms:
					for key in target_keys:
						channel_data.append(high_level_dict[key])
					#channel_data.append(high_level_dict[target_keys])
				subject.waveforms = []
			channel_data = self.flatten_array(channel_data) # flatten data to ensure just a list of ndarrays (in case of subepoching)
			raw_data_dict[channel] = channel_data # the key for the channel holds an array of waveforms or lists of waveforms
		# need to take a dict of channels with raw data and turn it into the form of an ndarray with shape (n_epochs, n_channels, n_times)
		raw_data_array = dict_to_3d_array(raw_data_dict)
		#print(f"CSP->get_raw 3d array length: {raw_data_array.shape}")
		return raw_data_array	

	def merge_arrays(self, dict_of_arrays):
		arrays_list = []
		for key in dict_of_arrays:
			arrays_list.append(dict_of_arrays[key])
		merged_array = np.stack(arrays_list, axis=1)
		return merged_array

	def epochs_per_subject(self, subjects, target_keys=None):
		arr_epochs_per_subject = []
		for subject in subjects:
			single_arr = [subject]
			arr_epochs_per_subject.append(self.get_raw(single_arr, target_keys).shape[0])

		print(f"Epochs per subject -> {arr_epochs_per_subject}")
		return arr_epochs_per_subject

	def expand_labels(self, subjects, labels, target_keys=None):
		if target_keys==None:
			target_keys = ['4-8']
		expanded_labels = []	
		arr_epochs_per_subject = self.epochs_per_subject(subjects, target_keys)
		for index, num_epochs in enumerate(arr_epochs_per_subject):
			addition = [labels[index]] * num_epochs
			expanded_labels = expanded_labels + addition
		return expanded_labels

	def shuffle_in_unison(self, labels, data):
		assert len(labels) == data.shape[0], "Length of labels must match the first dimension of data array"
		indices = np.arange(data.shape[0])
		np.random.shuffle(indices)
		shuffled_labels = np.array(labels)[indices]
		shuffled_data = data[indices]
		return shuffled_labels, shuffled_data

	def kfold_training_set_keys(self, kfold, target_channels=None, target_keys=None): #may have broken things here 
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		print(target_keys)
		if target_keys == None:
			target_keys = ['8-12']
		print(target_keys)
		testing_labels = self.expand_labels(testing_data, testing_labels, target_keys)
		training_labels = self.expand_labels(training_data, training_labels, target_keys)
#target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']    # bad hard coded need filter solution

		print("###############################################################")
		print("# Starting Testing Data Aggreg")
		print("###############################################################")
		testing_data_raw = []
		for subject in testing_data:
			raw_all_freq_keys = []
			for key in target_keys: #put all keys in a 1 * 400 * n keys 
				print(key)
				raw_all_freq_keys.append(self.get_raw([subject], target_keys=[key]).reshape(-1, 5 * 80))
			raw_all_freq_keys_combined = np.vstack(raw_all_freq_keys).reshape(-1, len(target_keys)*400)
			print(f"raw_all_freq_keys_combined shape {raw_all_freq_keys_combined.shape}")
			testing_data_raw.append(raw_all_freq_keys_combined)
		testing_data = np.vstack(testing_data_raw)
		print(f"The testing data is of shape {testing_data.shape}")
		print("###############################################################")
		print("# Starting Training Data Aggreg")
		print("###############################################################")
	
		training_data_raw = []
		for subject in training_data:
			raw_all_freq_keys = []
			for key in target_keys: #put all keys in a 1 * 400 * n keys 
				raw_all_freq_keys.append(self.get_raw([subject], target_keys=[key]).reshape(-1, 5 * 80))
			raw_all_freq_keys_combined = np.vstack(raw_all_freq_keys).reshape(-1, len(target_keys)*400)
			print(f"raw_all_freq_keys_combined shape {raw_all_freq_keys_combined.shape}")
			training_data_raw.append(raw_all_freq_keys_combined)
		training_data = np.vstack(training_data_raw)
		print(f"The training data is of shape {training_data.shape}")

## for each subject -> get the raw data 
#shuffle training and testing data + labels 
		testing_labels, testing_data = self.shuffle_in_unison(testing_labels, testing_data)
		training_labels, training_data = self.shuffle_in_unison(training_labels, training_data)
		return testing_data, testing_labels, training_data, training_labels
	
	def kfold_training_set(self, kfold, target_channels=None, target_keys=None): #may have broken things here 
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		if target_keys == None:
			target_keys = ['8-12']
		testing_labels = self.expand_labels(testing_data, testing_labels, target_keys)
		training_labels = self.expand_labels(training_data, training_labels, target_keys)
		#target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']    # bad hard coded need filter solution

		testing_data_raw = []
		for subject in testing_data:
			testing_data_raw.append(self.get_raw([subject], target_keys=target_keys).reshape(-1, 5 * 80))
		testing_data = np.vstack(testing_data_raw)

		training_data_raw = []
		for subject in training_data:
			training_data_raw.append(self.get_raw([subject], target_keys=target_keys).reshape(-1, 5 * 80))
		training_data = np.vstack(training_data_raw)

		## for each subject -> get the raw data 
		#shuffle training and testing data + labels 
		#testing_labels, testing_data = self.shuffle_in_unison(testing_labels, testing_data)
		#training_labels, training_data = self.shuffle_in_unison(training_labels, training_data)
		return testing_data, testing_labels, training_data, training_labels
	def kfold_csp_set_tuned(self, kfold, target_channels=None, target_keys=None, tuning_percentage=0.1):
		# Fetch testing and training data and labels
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		testing_labels = self.expand_labels(testing_data, testing_labels, target_keys=target_keys)
		training_labels = self.expand_labels(training_data, training_labels, target_keys=target_keys)

		# Set default target_keys if None
		if target_keys is None:
			target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']

		# Prepare CSP output dictionaries
		testing_csp_output_dict = {}
		training_csp_output_dict = {}

		# Apply CSP filtering to each target frequency range
		for key in target_keys:
			testing_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(testing_data, target_keys=[key]))
			training_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(training_data, target_keys=[key]))

		# Merge the filtered arrays
		testing_data = self.merge_arrays(testing_csp_output_dict)
		training_data = self.merge_arrays(training_csp_output_dict)

		# Ensure data has the shape (None, 8) by removing any extra dimension
		testing_data = np.squeeze(testing_data)
		training_data = np.squeeze(training_data)

		# Determine the split index for tuning data based on the tuning percentage
		tuning_size = int(len(testing_data) * tuning_percentage)
		tune_data, tune_labels = testing_data[:tuning_size], testing_labels[:tuning_size]
		testing_data, testing_labels = testing_data[tuning_size:], testing_labels[tuning_size:]

		# Shuffle the testing and training data + labels separately, preserving split
		testing_labels, testing_data = self.shuffle_in_unison(testing_labels, testing_data)
		training_labels, training_data = self.shuffle_in_unison(training_labels, training_data)

		# Ensure labels are numpy arrays of native Python floats
		testing_labels = np.array([float(label) for label in testing_labels])
		training_labels = np.array([float(label) for label in training_labels])
		tune_labels = np.array([float(label) for label in tune_labels])

		# Return all datasets, including tuning set
		return testing_data, testing_labels, training_data, training_labels, tune_data, tune_labels

	def kfold_csp_set(self, kfold, target_channels=None, target_keys=None): #may have broken things here 
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		testing_labels = self.expand_labels(testing_data, testing_labels, target_keys = target_keys)	
		training_labels = self.expand_labels(training_data, training_labels, target_keys = target_keys)	
	
		if target_keys == None:
			target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']	# bad hard coded need filter solution
		testing_csp_output_dict = {}
		training_csp_output_dict = {}
		#print(len(testing_data))
		for key in target_keys:
			testing_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(testing_data, target_keys=[key]))
			#print(testing_csp_output_dict[key].shape)
			training_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(training_data, target_keys=[key]))
			#print(training_csp_output_dict[key].shape)

		testing_data = self.merge_arrays(testing_csp_output_dict)
		training_data = self.merge_arrays(training_csp_output_dict)
		#shuffle training and testing data + labels 
		print(f"testing_labels len {len(testing_labels)}")
		print(f"training_labels len {len(training_labels)}")
		print(f"testing_data len {len(testing_data)}")
		print(f"training_data len {len(training_data)}")

		testing_labels, testing_data = self.shuffle_in_unison(testing_labels, testing_data)
		training_labels, training_data = self.shuffle_in_unison(training_labels, training_data)
		return testing_data, testing_labels, training_data, training_labels
	def apply_binary_filter(self, kfold, binary_filter):
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		testing_labels = self.expand_labels(testing_data, testing_labels)
		training_labels = self.expand_labels(training_data, training_labels)
		print()
