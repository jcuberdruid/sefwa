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
# same as model 14A but for 8 class  
################################################################

from tensorflow.keras.layers import BatchNormalization


def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(1, 8)))
    model.add(Flatten())

    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))
    
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    model.add(Dense(units=8, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.1))

    model.add(Dense(units=numLabels, activation='sigmoid'))

    return model

