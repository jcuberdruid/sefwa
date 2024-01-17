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
		with open(savePath, 'wb') as file:			
			pickle.dump(self.data, file)
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
			self.id = id_to_load
