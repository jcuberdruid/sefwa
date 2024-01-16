import argparse
from sklearn.metrics import accuracy_score
from eegHandler import EEGDataHandler
from channelTransformer import ChannelTransformer
import numpy as np
import pandas as pd
from multiprocessing import Pool
import os

# Function to train and evaluate a model for a given channel
def train_and_evaluate_channel(channel, cl1_train, cl2_train, cl1_val, cl2_val, cl1_test, cl2_test):
    svm = ChannelTransformer(channel)
    svm.train(cl1_train, cl2_train)
    svm.validate(cl1_val, cl2_val)
    svm.test(cl1_test, cl2_test)
    return svm

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='EEG Channel Classification')
    parser.add_argument('class_1_path', type=str, help='Path to class 1 CSV file')
    parser.add_argument('class_2_path', type=str, help='Path to class 2 CSV file')
    parser.add_argument('slice_index', type=str, help='which slice of the epoch to try')
    args = parser.parse_args()

    # Paths to your data
    class_1_path = args.class_1_path
    class_2_path = args.class_2_path
    slice_index = int(args.slice_index)
    print(slice_index)

    slice_index_2 = slice_index

    # Initialize EEGDataHandler instances
    eeg_handler_class_1 = EEGDataHandler(class_1_path, slice_index, 0.5) 
    eeg_handler_class_2 = EEGDataHandler(class_2_path, slice_index_2, 0.5)

    # Split data into training, validation, and testing sets
    cl1_train, cl1_val, cl1_test = eeg_handler_class_1.split_data(train_size=0.8, test_size=0.1, random_state=41)
    cl2_train, cl2_val, cl2_test = eeg_handler_class_2.split_data(train_size=0.8, test_size=0.1, random_state=41)

    # Number of electrodes/channels
    num_channels = 1

    # Using multiprocessing to train Transformer models in parallel
    num_processes = 1  # Using half of the available CPU cores
    svm_models = []
    #train_and_evaluate_channel(channel, cl1_train, cl2_train, cl1_val, cl2_val, cl1_test, cl2_test)
    svm = train_and_evaluate_channel(0, cl1_train, cl2_train, cl1_val, cl2_val, cl1_test, cl2_test) #just channel F7
    svm_models.append(svm)

    # Prepare data for CSV
    results_data = []
    for channel, svm in enumerate(svm_models):
        results_data.append({
            'Channel': channel,
            'Training Accuracy': svm.get_training_accuracy(),
            'Validation Accuracy': svm.get_validation_accuracy(),
            'Testing Accuracy': svm.get_testing_accuracy()
        })

    # Step 1: Calculate Weights
    weights = [svm.get_validation_accuracy() for svm in svm_models]
    # Step 2: Predict on Test Data
    channel_predictions = np.zeros((len(cl1_test) + len(cl2_test), num_channels))
    for i, svm in enumerate(svm_models):
        #predictions = svm.predict(np.vstack((cl1_test, cl2_test)))
        predictions = svm.predict(np.vstack((cl1_test, cl2_test))) 
        channel_predictions[:, i] = predictions
    
    print(channel_predictions)
    # Step 3: Apply Weighted Voting
    weighted_votes = np.dot(channel_predictions, weights)
    final_predictions = np.where(weighted_votes > sum(weights) / 2, 1, 0)

    # Step 4: Evaluate the Ensemble Model
    test_labels = np.hstack((np.zeros(len(cl1_test)), np.ones(len(cl2_test))))
    ensemble_accuracy = accuracy_score(test_labels, final_predictions)
    print(f"Ensemble Model Accuracy: {ensemble_accuracy}")

    # Add ensemble accuracy
    results_data.append({
        'Channel': 'Ensemble',
        'Training Accuracy': None,
        'Validation Accuracy': None,
        'Testing Accuracy': ensemble_accuracy
    })

    # Convert to DataFrame and save to CSV
    results_df = pd.DataFrame(results_data)
    csv_filename = f"resultsTransformerHHT_hyperTuningHalfData/results_{os.path.basename(class_1_path)}_{os.path.basename(class_2_path)}_slice_{slice_index}"
    results_df.to_csv(csv_filename, index=False)

    print(f"Results saved to {csv_filename}")

if __name__ == "__main__":
    main()

