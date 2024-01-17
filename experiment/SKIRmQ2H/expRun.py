import os
import pickle 

from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler
from cl.userExtension.filterBank import FilterBank 
from cl.userExtension.readyTrainingDataHandler import ReadyTrainingDataHandler


import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend suitable for headless servers
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

def main():
	rtdh = ReadyTrainingDataHandler()
	rtdh.loadTrainingSet('5GZ6Ulbq')
	d = rtdh.data
	print(len(d))
	print(len(d[0]))
	print("class 1")
	for x in d[0]:
		print(len(x))
	print("class 2")
	for x in d[1]:
		print(len(x))


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

	for x in filteredSubjects_1: 
		x.split_epochs(320, 960, 160)
	
	for x in filteredSubjects_2: 
		x.split_epochs(320, 960, 160)

	rtdh = ReadyTrainingDataHandler()
	rtdh.create_kfolds(filteredSubjects_1, filteredSubjects_2, 5)
	rtdh.saveTrainingSet()
	
'''
    with open('test_pickle_of_subjects_arr.pkl', 'wb') as f:
    	pickle.dump(filteredSubjects, f)
'''

main()
