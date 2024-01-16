import os
from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler
from cl.userExtension.filterBank import FilterBank 


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
    class_1_path = os.path.join(paths.projectDir, "data/JxGxSzB3/MI_RLH_T1.csv")
    eeg_handler_class_1 = EEGDataHandler(class_1_path, 0.10)
    sample_dict = eeg_handler_class_1.subjects[1].epochs[4].channels_dict
    print(sample_dict.keys())

    fb = FilterBank(4, 40, 4)
    subjects = eeg_handler_class_1.subjects.copy()
    filteredSubjects = fb.bankSubjects(subjects)
    
    sample_dict = filteredSubjects[1].epochs[4].channels_dict
    print(sample_dict.keys())
    print(len(sample_dict['F7']))
    print(sample_dict['F7'].keys())
    print(type(sample_dict['F7']['16-20']))
    print(type(sample_dict['F7']['16-20'][0]))
    save_waveform_plots(sample_dict['F7'])

    #for x in subjects: 
    for x in filteredSubjects: 
	    x.split_epochs(320, 960, 160)

main()
