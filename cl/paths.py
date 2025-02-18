# File: paths.py
import os
paths_py_path = os.path.dirname(os.path.abspath(__file__))
print(paths_py_path)
projectDir = os.path.dirname(paths_py_path) 

#projectDir = "/home/jason/eeg_bci_projects/sefwa"
#projectDir = "/keras/eeg_bci_projects/sefwa"
dataDir = os.path.join(projectDir, "data")
experimentDir = os.path.join(projectDir, "experiment")
analysisDir = os.path.join(projectDir, "analysis")
resultDir = os.path.join(projectDir, "results")
archiveDir = os.path.join(projectDir, "archive")
