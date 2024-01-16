import os
from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler

def main():
    class_1_path = os.path.join(paths.projectDir, "data/JxGxSzB3/MI_RLH_T1.csv")
    eeg_handler_class_1 = EEGDataHandler(class_1_path, 1)
    print("bla")
    print(f"number of subject: {len(eeg_handler_class_1.data)}")
    print(f"number of epochs in first subject: {len(eeg_handler_class_1.data[0].epochs)}")
    print(f"number of epochs in 105th subject: {len(eeg_handler_class_1.data[104].epochs)}")
    print(f"number of samples in electrode F7: {len(eeg_handler_class_1.data[0].epochs[0].channels_dict['F7'])}")

main()
