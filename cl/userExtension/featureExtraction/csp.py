import numpy as np
import mne
import random

from cl.userExtension import subject 
from cl.userExtension import epoch 


class CSP:
	def __init__(self, max_attempts = 10):
		self.filters = None
		self.max_attempts = max_attempts

	def flatten_array(self, arr):
		return [item for sublist in arr for item in (sublist if isinstance(sublist, list) else [sublist])]

	def get_raw(self, subjects, target_channels = None, target_keys=None):
		def dict_to_3d_array(dict_of_arrays):
			list_length = len(next(iter(dict_of_arrays.values())))
			array_length = len(dict_of_arrays[next(iter(dict_of_arrays))][0])
			result_array = np.empty((list_length, len(dict_of_arrays), array_length))
			for i, key in enumerate(dict_of_arrays):
				for j, array in enumerate(dict_of_arrays[key]):
					result_array[j, i, :] = array
			return result_array

		raw_data_dict = {}
		for channel in target_channels:
			channel_data = []
			for subject in subjects: 
				subject.gather_all_waveforms([channel], return_high_level=True)
				for high_level_dict in subject.waveforms:
					channel_data.append(high_level_dict[target_keys])
				subject.waveforms = []
			channel_data = self.flatten_array(channel_data) # flatten data to ensure just a list of ndarrays (in case of subepoching)
			raw_data_dict[channel] = channel_data # the key for the channel holds an array of waveforms or lists of waveforms

		# need to take a dict of channels with raw data and turn it into the form of an ndarray with shape (n_epochs, n_channels, n_times)
		raw_data_array = dict_to_3d_array(raw_data_dict)
		#print(f"CSP->get_raw 3d array length: {raw_data_array.shape}")
		return raw_data_array

	def make_filters(self, class_1, class_2, n_components = 8, target_channels=None, target_keys = None):
		self.filter_dict = {}

		if len(class_1) == len(class_2):
			print("Both classes are of the same length:", len(class_1))
		else:
			print("The classes have different lengths. Length of class_1:", len(class_1), "and length of class_2:", len(class_2))
			print(f"This may cause issues when creating filters, if it fails it will be retried {max_attempts} times")

		if target_channels==None:
			target_channels = list(class_1[0].epochs[0].channels_dict.keys())
		
		if target_keys == None: # hard coded so needs to be changed but no sure way of getting them from epochs yet XXX
			target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']
		
		for key in target_keys:
			print("################################################")
			print(f"CSP->make_filters: Training filters for key {key}")
			print("##################################################")
			raw_class_2 = self.get_raw(class_2, target_channels, key)
			raw_class_1 = self.get_raw(class_1, target_channels, key)
			try:
				csp_model = self.train_csp_with_random_epoch_removal(raw_class_1, raw_class_2, n_components, target_channels)
				self.filter_dict[key] = csp_model
			except Exception as e:
				print(f"CSP->make_filters: train_csp_with_random_epoch_removal() failed: {e}")
			

	def train_csp(self, epochs_data_class1, epochs_data_class2, n_components=8, target_channels = None):
		#print(epochs_data_class2)
		#epochs_data_class1 = [[channel_data] for channel_data in epochs_data_class1]
		#epochs_data_class2 = [[channel_data] for channel_data in epochs_data_class2]
		epochs_data = np.concatenate([epochs_data_class1, epochs_data_class2], axis=0)
		epochs_labels = np.concatenate([np.zeros(len(epochs_data_class1)), np.ones(len(epochs_data_class2))])
		print(epochs_data.shape)
		#ch_names = df_class1.columns[5:].tolist()
		ch_types = ['eeg'] * len(target_channels)
		
		sfreq = 160  # Modify this as per your sampling frequency
		montage_1010 = mne.channels.make_standard_montage('standard_1005')
		info = mne.create_info(ch_names=target_channels, sfreq=sfreq, ch_types=ch_types)
		info.set_montage(montage_1010, match_case=False)
		custom_epochs = mne.EpochsArray(epochs_data, info, tmin=0, event_id=None, verbose=False)
		csp = mne.decoding.CSP(n_components=n_components, reg=None, log=True, norm_trace=False)
		csp.fit(custom_epochs.get_data(), epochs_labels)
		return csp

	def train_csp_with_random_epoch_removal(self, class1, class2, n_components, target_channels = None):
		for attempt in range(self.max_attempts):
			try:
				if attempt > 0:
					drop_index_class1 = np.random.randint(class1.shape[0])
					class1 = np.concatenate([class1[:drop_index_class1], class1[drop_index_class1 + 1:]], axis=0)

					drop_index_class2 = np.random.randint(class2.shape[0])
					class2 = np.concatenate([class2[:drop_index_class2], class2[drop_index_class2 + 1:]], axis=0)

				csp_model = self.train_csp(class1, class2, n_components, target_channels)
				print(f"CSP->train_csp_with_random_epoch_removal: Attempt {attempt}")
				return csp_model
			except np.linalg.LinAlgError:
				continue

		print(f"Failed to train CSP model after {self.max_attempts} attempts.")
		return None