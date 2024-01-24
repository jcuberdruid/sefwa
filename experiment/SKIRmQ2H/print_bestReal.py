import os
import pickle 
import numpy as np
import mne

#classification: tensorflow
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from kerastuner.tuners import BayesianOptimization
from tensorflow.keras.callbacks import EarlyStopping

# tf.keras.mixed_precision.set_global_policy('mixed_float16')

from cl import paths
from cl.userExtension.eegHandler import EEGDataHandler
from cl.userExtension.filterBank import FilterBank 
from cl.userExtension.readyTrainingDataHandler import ReadyTrainingDataHandler
from cl.userExtension.featureExtraction.csp import CSP

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt


class AutoencoderHyperModel(HyperModel):
	def __init__(self, input_shape):
		self.input_shape = input_shape

	def build(self, hp):
		inputs = Input(shape=self.input_shape, name='input_layer')
		flattened = Flatten(name='flatten_layer')(inputs)

		encoded = Dense(
			units=hp.Int('encoder_units', min_value=32, max_value=256, step=16),
			activation='relu',
			name='encoder_dense'
		)(flattened)

		decoded = Dense(np.prod(self.input_shape), activation='linear', name='decoder_dense')(encoded)
		decoded = Reshape(self.input_shape, name='reshape_layer')(decoded)

		autoencoder = Model(inputs, decoded, name='autoencoder_model')
		autoencoder.compile(
			optimizer=tf.keras.optimizers.Adam(
				hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
			),
			loss='mse'
		)

		return autoencoder

def tune_autoencoder(training_data, testing_data):
    input_shape = (9, 8)
    hypermodel = AutoencoderHyperModel(input_shape)
    tuner = BayesianOptimization(
        hypermodel,
        objective='val_loss',
        max_trials=10,
        num_initial_points=2,
        directory='autoencoder_tuning',
        project_name='autoencoder_bayesian'
    )
    tuner.search(training_data, training_data, epochs=100, validation_data=(testing_data, testing_data))

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]
    print("Best hyperparameters for autoencoder:", best_hp.values)

    # Return the best model
    return tuner.get_best_models(num_models=1)[0]


class ClassifierHyperModel(HyperModel):
    def __init__(self, encoder):
        self.encoder = encoder

    def build(self, hp):
        for layer in self.encoder.layers:
            layer.trainable = False

        # Start building the model from the encoder output
        x = self.encoder.layers[-2].output

        # Dynamically add dense layers
        for i in range(hp.Int('num_dense_layers', 1, 3)):
            x = Dense(
                units=hp.Int(f'units_{i}', min_value=512, max_value=4096, step=16),
                activation='sigmoid',
                name=f'classifier_dense_{i}'
            )(x)
            x = Dropout(
                hp.Float(f'dropout_{i}', min_value=0.0, max_value=0.1, step=0.1)
            )(x)

        # Final output layer
        classifier_output = Dense(2, activation='softmax', name='classifier_output')(x)

        # Create and compile the model
        classifier = Model(self.encoder.input, classifier_output, name='classifier_model')
        classifier.summary()
        classifier.compile(
            optimizer=tf.keras.optimizers.Adam(
                hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')
            ),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return classifier

def tune_classifier(encoder, training_data, training_labels, testing_data, testing_labels):
    hypermodel = ClassifierHyperModel(encoder)

    tuner = BayesianOptimization(
        hypermodel,
        objective='val_accuracy',
        max_trials=1000,
        num_initial_points=20,
        directory='classifier_tuning',
        project_name='classifier_bayesian'
    )

    early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

    tuner.search(
        training_data,
        training_labels,
        epochs=100,
        validation_data=(testing_data, testing_labels),
        callbacks=[early_stopping]
    )

    # Get the best hyperparameters
    best_hp = tuner.get_best_hyperparameters()[0]

    # Access the best trial's details
    best_trial = tuner.oracle.get_best_trials(num_trials=1)[0]
    best_val_accuracy = best_trial.metrics.get_best_value('val_accuracy')

    print("Best hyperparameters for classifier:", best_hp.values)
    print("Best validation accuracy:", best_val_accuracy)

    # Return the best model
    return tuner.get_best_models(num_models=1)[0]



def classify_test():
	rtdh = ReadyTrainingDataHandler()
	rtdh.loadTrainingSet('dz05QLQr')
	d = rtdh.data
	
	testing_data, testing_labels, training_data, training_labels = rtdh.kfold_csp_set(1)

	print("testing")
	print(testing_data.shape)
	print(len(testing_labels))

	print("training")
	print(training_data.shape)
	print(len(training_labels))

	best_autoencoder = tune_autoencoder(training_data, testing_data)
	encoder = Model(inputs=best_autoencoder.input, outputs=best_autoencoder.layers[-2].output)
	best_classifier = tune_classifier(encoder, training_data, training_labels, testing_data, testing_labels)
	# split testing to be per subject:
		
		

def main():
	rtdh = ReadyTrainingDataHandler()
	rtdh.loadTrainingSet('UJY1Rvga')
	d = rtdh.data
	
	print("class 2 kfold lengths:")
	for kfold_list in rtdh.data[1]:
		print(len(kfold_list))

	print("making filters")
	for kfold in range(len(rtdh.data[1])):	
		class_1_training =  np.concatenate([rtdh.data[0][i] for i in range(len(rtdh.data[0])) if i != kfold])
		class_2_training =  np.concatenate([rtdh.data[1][i] for i in range(len(rtdh.data[0])) if i != kfold])
		csp = CSP()
		csp.make_filters(class_1_training, class_2_training)
		rtdh.kfold_csp(kfold, csp.filter_dict)
	rtdh.saveTrainingSet()
	
	
	exit(0)
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
	
	for x in filteredSubjects_1: 
		x.split_epochs(320, 960, 160)
	
	for x in filteredSubjects_2: 
		x.split_epochs(320, 960, 160)
		
	rtdh = ReadyTrainingDataHandler()
	rtdh.create_kfolds(filteredSubjects_1, filteredSubjects_2, 5)
	rtdh.saveTrainingSet()
	
#main()
classify_test()
