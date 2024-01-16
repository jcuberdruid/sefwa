import numpy as np
import pandas as pd
import os 
import subprocess

path = "classification/kfolder/" #where to save intermediate npy files

nFold = 10

def extract_data_for_subject(csv_path, subject_number):
    df = pd.read_csv(csv_path)
    df_subject = df[df['subject'] == subject_number]
    data_array = df_subject[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy().reshape(-1, 1, 8)
    return data_array

def stratified_split(arr1, arr2, n_splits=nFold):
    # Determine the maximum number of splits based on the smallest array
    max_splits = min(len(arr1), len(arr2), n_splits)

    # Shuffle the arrays to randomize the order
    np.random.shuffle(arr1)
    np.random.shuffle(arr2)

    # Calculate split sizes for each array, ensuring that each element appears once in a validation fold
    split_sizes = np.linspace(0, len(arr1), max_splits + 1, dtype=int)
    feature_folds = []
    label_folds = []
    for i in range(max_splits):
        # Get the start and end indices for the current fold
        start_idx, end_idx = split_sizes[i], split_sizes[i+1]

        # Extract the current fold from each array
        fold_arr1 = arr1[start_idx:end_idx]
        fold_arr2 = arr2[start_idx:end_idx]

        # Concatenate the current folds and shuffle
        fold_features = np.concatenate((fold_arr1, fold_arr2), axis=0)
        fold_labels = np.concatenate((np.zeros(len(fold_arr1)), np.ones(len(fold_arr2))))

        # Shuffle the combined folds together to maintain corresponding features and labels
        indices = np.arange(len(fold_features))
        np.random.shuffle(indices)
        fold_features = fold_features[indices]
        fold_labels = fold_labels[indices]

        feature_folds.append(fold_features)
        label_folds.append(fold_labels)

    return feature_folds, label_folds

def fullKfold(subject):
    subjectString = "S"+str(subject)
    base_path = f"../data/datasets/processed9_moving_average_demeaning_include_2s_before/train_set_homogenious_csp/filter_sub_{subject}"
    path_class_1 = f"{base_path}/Transformed_MI_RLH_T1.csv"
    path_class_2 = f"{base_path}/Transformed_MI_RLH_T2.csv"


    data_class_1 = extract_data_for_subject(path_class_1, subject) 
    p = np.random.permutation(data_class_1.shape[0])
    data_class_1 = data_class_1[p]

    data_class_2 = extract_data_for_subject(path_class_2, subject)
    p = np.random.permutation(data_class_2.shape[0])
    data_class_2 = data_class_2[p]
     
    kfolds, klabels = stratified_split(data_class_1, data_class_2)
    
    for x in kfolds:
        print(x.shape)

    for x in range(0,nFold):
        tmp_kfolds = kfolds.copy()
        tmp_klabels = klabels.copy()
        testFold =  tmp_kfolds.pop(x) 
        testLabels =tmp_klabels.pop(x) 

        tmp_kfolds_all = np.concatenate(tmp_kfolds, axis=0) 
        tmp_klabels_all = np.concatenate(tmp_klabels, axis=0) 

        np.save(path+"train_data.npy", tmp_kfolds_all)
        np.save(path+"train_labels.npy", tmp_klabels_all)
        np.save(path+"test_data.npy", testFold)
        np.save(path+"test_labels.npy", testLabels)
        subprocess.run(['python3', "-m", "classification.runCrossValGroup", "-subject", str(subject), "-kfold", str(x)])

#exclude = {87, 92, 100, 104}
exclude = {88, 92, 100, 104}
subjects = []
for x in range(1, 109+1):
    subjects.append(x)
for x in exclude:
  subjects.remove(x)
for x in subjects:
    print(x)
    if x not in exclude:
        fullKfold(x)
    else:
        continue 
