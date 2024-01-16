################################################################
# Simple SVM model for 64x640x3 2d fft 
################################################################

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def model(num_classes):
    # Specify the input shape based on the image shape (64, 640, 3)
    input_shape = (64, 640, 3)
    
    # Create the SVM model
    svm_model = keras.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Flatten(),  # This will flatten the input
        layers.Dense(units=num_classes, activation=None, kernel_regularizer=tf.keras.regularizers.l2(0.01))
    ])
    
    # Compile the model
    svm_model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])
    
    return svm_model
