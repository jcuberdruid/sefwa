import os
from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler
from cl.userExtension.filterBank import FilterBank 

def main():
    class_1_path = os.path.join(paths.projectDir, "data/JxGxSzB3/MI_RLH_T1.csv")
    eeg_handler_class_1 = EEGDataHandler(class_1_path, 0.25)
    sample_dict = eeg_handler_class_1.subjects[1].epochs[4].channels_dict
    print(sample_dict.keys())

    fb = FilterBank(4, 40, 4)
    subjects = eeg_handler_class_1.subjects.copy()
    filteredSubjects = fb.bankSubjects(subjects)
    
    sample_dict = filteredSubjects[1].epochs[4].channels_dict
    print(sample_dict.keys())
    print(sample_dict['F7'])

main()
