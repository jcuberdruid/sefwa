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


def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(64, 960, 3)))  # Shape of input images
    model.add(Conv2D(filters=8, kernel_size=(9, 5), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=16, kernel_size=(7, 4), activation='relu'))
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(units=128, activation='sigmoid', kernel_regularizer=l1(0.001), name='dense'))
    model.add(Dropout(rate=0.2, name="dropout_2"))
    model.add(Dense(units=64, activation='sigmoid', kernel_regularizer=l1(0.003), name='dense_1'))
    model.add(Dense(units=numLabels, activation='sigmoid', name='dense_2'))
    model.add(Dense(units=numLabels, activation='softmax', name='dense_3'))


    return model
