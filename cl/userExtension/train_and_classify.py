import multiprocessing
from tensorflow.keras.preprocessing import image as kimage
import pandas as pd
import sys
import os
import json
from keras import backend as K
from keras.callbacks import Callback
import numpy as np
import datetime
import csv
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal
from keras.models import Model
from tensorflow.keras.optimizers import Adam
#import jsonLog as JL
import importlib
from sklearn.preprocessing import LabelEncoder

class Classifier:

    ##############################################################
    # Global Vars 
    ##############################################################

    batch_size = 1
    epochs = 10
    module = ""
    run_note = "unspecified"
    
    def init_tf(self):
        #tf.keras.mixed_precision.set_global_policy('mixed_float16')
        # Set GPU memory growth option
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    def __init__(self, run_note, subject, batch_size, epochs, model_name, dataset):
        self.init_tf()
        model_name = model_name.strip(".py")
        module_name = f"cl.userExtension.models.{model_name}"
        self.module = importlib.import_module(module_name)
        self.batch_size = batch_size
        self.epochs = epochs
        self.run_note = run_note 

    def unison_shuffled_copies(a, b):
        assert a.shape[0] == b.shape[0], "First dimension must be the same size for both arrays"
        p = np.random.permutation(a.shape[0])
        return a[p], b[p]

    def classify(self, train_data, train_labels, test_data, test_labels):

        # Number of unique labels (classes)
        numLabels = len(set(train_labels))

        ######################################################################
        # Main Model 
        ######################################################################

        model = self.module.model(numLabels) 
        model.summary()
        config = model.to_json()
        #JL.model_log(config)

        ######################################################################
        # Optimize
        ######################################################################

        learning_rate = 9.8747e-04  # Updated to the learning rate used in the tuning section
        optimizer = Adam(learning_rate=learning_rate)

        ######################################################################
        # Call backs 
        ######################################################################

        filepath = "best_model.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
        #json_logger = JL.JSONLogger('epoch_performance.json')
        earlystop = EarlyStopping(monitor='val_accuracy', patience=300,verbose=1, mode='max')

        callbacks_list = [checkpoint, earlystop] #, json_logger]

        ######################################################################
        # Compile and fit model : training
        ######################################################################
        #for transformer loss ='binary_crossentropy'
        model.compile(optimizer=optimizer,loss='binary_crossentropy', metrics=['accuracy'])
        model.fit(train_data, train_labels, epochs=self.epochs, batch_size=self.batch_size, validation_data=(test_data, test_labels), callbacks=callbacks_list)

        ######################################################################
        # Evaluate the best model
        ######################################################################
        print("Evaluating Model")
        model.load_weights("best_model.hdf5")
        test_loss, test_accuracy = model.evaluate(test_data, test_labels)

        print('Test Loss:', test_loss)
        global accuracy
        global loss
        accuracy = test_accuracy
        loss = test_loss
        return test_accuracy

    def extract_data_for_subject(csv_path, subject_number):
        df = pd.read_csv(csv_path)
        df_subject = df[df['subject'] == subject_number]
        data_array = df_subject[["0", "1", "2", "3", "4", "5", "6", "7"]].to_numpy().reshape(-1, 1, 8)
        return data_array

    def combine_and_label(array1, array2):
        labels1 = np.zeros((array1.shape[0],))
        labels2 = np.ones((array2.shape[0],))
        data = np.concatenate([array1, array2], axis=0)
        labels = np.concatenate([labels1, labels2], axis=0)
        return data, labels

    def runSubject(testingSubjects, source_directory):
        print("###############################################################")
        print(f"running subject: {testingSubjects[0]}") 
        print("###############################################################")

        subjectString = "S"+str(testingSubjects[0])
        base_path = f"../data/datasets/processed9_moving_average_demeaning_include_2s_before/train_set_homogenious_csp/filter_sub_{testingSubjects[0]}"
        path_class_1 = f"{base_path}/Transformed_MI_RLH_T1.csv"
        path_class_2 = f"{base_path}/Transformed_MI_RLH_T2.csv"

        data_class_1 = extract_data_for_subject(path_class_1, int(testingSubjects[0]))
        data_class_2 = extract_data_for_subject(path_class_2, int(testingSubjects[0]))

        data, labels = combine_and_label(data_class_1, data_class_2)

        #shuffle data
        data, labels = unison_shuffled_copies(data, labels)

        split_index = int(0.90 * data.shape[0])

        train_data = data[:split_index]
        test_data = data[split_index:]

        train_labels = labels[:split_index]
        test_labels = labels[split_index:]

        global batch_size

        classify(train_data, train_labels, test_data, test_labels)
        global run_note
        subjects = []
        training_files = []
        training_files.append("/../../keras/data/datasets/processed8_79high_MM/binary_test/class_inference")
        testing_files = training_files
        #JL.output_log(subjects, testingSubjects, training_files, testing_files, accuracy, run_note, dataset, batch_size)
        #JL.make_logs()

