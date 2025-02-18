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
# same as model 10 B but for 1dFFTs 
################################################################

from tensorflow.keras.layers import BatchNormalization

def model(numLabels):
    model = Sequential()
    model.add(Input(shape=(960, 64, 1)))
    model.add(Conv2D(filters=8, kernel_size=(9, 9), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    
    model.add(Conv2D(filters=16, kernel_size=(7, 7), activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    
    model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu'))
    model.add(BatchNormalization())
    
    model.add(Flatten())
    
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=16, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=8, activation='relu', kernel_regularizer=l2(0.002)))
    model.add(Dropout(rate=0.1))
    model.add(Dense(units=numLabels, activation='sigmoid'))

    return model

