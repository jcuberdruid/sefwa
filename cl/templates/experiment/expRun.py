import os
import pickle 
import numpy as np
import mne

from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler
from cl.userExtension.filterBank import FilterBank 
from cl.userExtension.readyTrainingDataHandler import ReadyTrainingDataHandler
from cl.userExtension.featureExtraction.csp import CSP

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

def save_waveform_plots(waveforms, sample_rate=160, num_samples=961, filename="waveforms.png"):
	# Determine the grid size for the subplots
	num_plots = len(waveforms)
	grid_size = int(num_plots**0.5)

	fig, axs = plt.subplots(grid_size, grid_size, figsize=(60, 15))
	axs = axs.flatten()  # Flatten the array of axes for easy iteration

	# Create a time axis in seconds
	time = [i / sample_rate for i in range(num_samples)]

	# Iterate over the dictionary and create subplots
	for i, (range_hz, waveform) in enumerate(waveforms.items()):
		axs[i].plot(time, waveform)
		axs[i].set_title(f"Range: {range_hz} Hz")
		axs[i].set_xlabel("Time (s)")
		axs[i].set_ylabel("Amplitude")

	# Hide any unused subplots
	for j in range(i+1, len(axs)):
		axs[j].set_visible(False)

	# Adjust layout and save the figure
	plt.tight_layout()
	fig.savefig(filename)

def test_function(data):
	if isinstance(data, dict):
		print(data.keys())

def main():
	rtdh = ReadyTrainingDataHandler()
	rtdh.loadTrainingSet('5uiKvhhy')
	d = rtdh.data
	
	testing_data, testing_labels, training_data, training_labels = rtdh.kfold_set(0)
	target_channels = list(testing_data[0].epochs[0].channels_dict.keys())	

	testing_data, testing_labels, training_data, training_labels = rtdh.kfold_csp_set(0)

	print("testing")
	ch_types = ['eeg'] * len(target_channels)
	sfreq = 160  # Modify this as per your sampling frequency
	montage_1010 = mne.channels.make_standard_montage('standard_1005')
	info = mne.create_info(ch_names=target_channels, sfreq=sfreq, ch_types=ch_types)
	info.set_montage(montage_1010, match_case=False)

	csp_figure = rtdh.csp_filter_dict[0]['8-12'].plot_patterns(info, ch_type="eeg", units="Patterns (AU)", size=1.5)
	print(type(csp_figure))
	csp_figure.savefig("test_csp_plot_8-12.png")

	csp_figure = rtdh.csp_filter_dict[0]['12-16'].plot_patterns(info, ch_type="eeg", units="Patterns (AU)", size=1.5)
	print(type(csp_figure))
	csp_figure.savefig("test_csp_plot_12-16.png")

	exit(0)	
	
	print("class 1 kfold lengths:")
	for kfold_list in rtdh.data[0]:
		print(len(kfold_list))

	print("class 2 kfold lengths:")
	for kfold_list in rtdh.data[1]:
		print(len(kfold_list))
	for kfold in range(len(rtdh.data[1])):	
		class_1_training =  np.concatenate([rtdh.data[0][i] for i in range(len(rtdh.data[0])) if i != kfold])
		class_2_training =  np.concatenate([rtdh.data[1][i] for i in range(len(rtdh.data[0])) if i != kfold])
		csp = CSP()
		csp.make_filters(class_1_training, class_2_training)
		rtdh.kfold_csp(kfold, csp.filter_dict)
	rtdh.saveTrainingSet()
	


	
def main2():
	class_1_path = os.path.join(paths.projectDir, "data/JxGxSzB3/MI_RLH_T1.csv")
	class_2_path = os.path.join(paths.projectDir, "data/JxGxSzB3/MI_RLH_T2.csv")

	eeg_handler_class_1 = EEGDataHandler(class_1_path, 1)
	eeg_handler_class_2 = EEGDataHandler(class_2_path, 1)

	fb_1 = FilterBank(4, 40, 4)
	subjects = eeg_handler_class_1.subjects.copy()
	filteredSubjects_1 = fb_1.bankSubjects(subjects)

	fb_2 = FilterBank(4, 40, 4)
	subjects = eeg_handler_class_2.subjects.copy()
	filteredSubjects_2 = fb_2.bankSubjects(subjects)
	'''
	for x in filteredSubjects_1: 
		x.split_epochs(320, 960, 160)
	
	for x in filteredSubjects_2: 
		x.split_epochs(320, 960, 160)
	'''	
	rtdh = ReadyTrainingDataHandler()
	rtdh.create_kfolds(filteredSubjects_1, filteredSubjects_2, 5)
	rtdh.saveTrainingSet()
	
main()
