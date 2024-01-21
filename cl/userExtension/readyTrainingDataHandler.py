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
		np.random.shuffle(testing_indices)
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
		target_channels = list(subjects[0].epochs[0].channels_dict.keys())
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

	def kfold_csp_set(self, kfold, target_channels=None):
		testing_data, testing_labels, training_data, training_labels = self.kfold_set(kfold)
		target_keys = ['4-8', '8-12', '12-16', '16-20', '20-24', '24-28', '28-32', '32-36', '36-40']	
		testing_csp_output_dict = {}
		training_csp_output_dict = {}
		print(len(testing_data))
		for key in target_keys:
			testing_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(testing_data, target_keys=key))
			print(testing_csp_output_dict[key].shape)
			training_csp_output_dict[key] = self.csp_filter_dict[kfold][key].transform(self.get_raw(training_data, target_keys=key))
			print(training_csp_output_dict[key].shape)
		exit(0)
		#for each epoch (all channels particular frequency bank):
			#gather epochs
		#for each in gathered epochs:
			#each =csp_model.transform()
		return testing_data, testing_labels, training_data, training_labels

	
