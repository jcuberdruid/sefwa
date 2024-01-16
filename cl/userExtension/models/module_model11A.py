import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
################################################################
# Same as model4 but with batch normalization and changed weights and kernal sizes 
################################################################

from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l1

import tensorflow as tf
from tensorflow.keras import Sequential, layers
from tensorflow_addons.layers import VisionTransformer

def model(numLabels):
    model = Sequential()
    model.add(layers.Input(shape=(64, 640, 3)))  # Input image shape

    # Vision Transformer layer
    model.add(VisionTransformer(
        image_size=(64, 640),
        patch_size=(16, 16),
        num_layers=4,
        num_heads=4,
        mlp_dim=128,
        dropout_rate=0.1,
        attention_dropout=0.1,
    ))
    
    # Flatten output from ViT
    model.add(layers.Flatten())
    
    # Reshape to have 3D data before GRU, (batch_size, timesteps, features)
    model.add(layers.Reshape((20, -1)))  # Adjust these dimensions as needed
    
    # GRU Layers
    model.add(layers.GRU(units=80, return_sequences=True))
    model.add(layers.Dropout(0.1))
    model.add(layers.BatchNormalization())
    model.add(layers.GRU(units=80))
    
    # Fully connected layers
    model.add(layers.Dense(units=512, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.001), name='dense'))
    model.add(layers.Dropout(rate=0.1, name="dropout_2"))
    model.add(layers.Dense(units=256, activation='relu', kernel_regularizer=tf.keras.regularizers.l1(0.002), name='dense_1'))
    model.add(layers.Dense(units=numLabels, activation='softmax', name='dense_3'))

    return model
