import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv3D, Flatten, Dense, Dropout, Input, MaxPool3D, GRU, Reshape, TimeDistributed, LSTM, GlobalMaxPool2D, MaxPool2D, BatchNormalization
from tensorflow.keras.initializers import GlorotUniform, Zeros, Orthogonal

from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from tensorflow.keras.regularizers import l2
################################################################
# same as 14A but motifying for tuning  
################################################################

from tensorflow.keras.layers import BatchNormalization


def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(1, 4)))
    model.add(Flatten())

    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.0001)))
    
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))
    model.add(Dense(units=512, activation='relu', kernel_regularizer=l2(0.001)))
    
    model.add(Dense(units=128, activation='relu', kernel_regularizer=l2(0.001)))
    #model.add(BatchNormalization())
    #model.add(Dropout(rate=0.1))

    model.add(Dense(units=numLabels, activation='softmax'))

    return model

