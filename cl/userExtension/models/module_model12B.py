################################################################
# Simple SVM model for 64x640x3 2d fft 
################################################################
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def model(num_classes):
    # Specify the input shape based on the image shape (64, 640, 3)
    input_shape = (64, 640, 3)

    # Initialize a Sequential model
    model = Sequential()

    # First Convolutional layer with 32 filters and kernel size of (3, 3)
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))

    # Max Pooling layer to reduce the spatial dimensions
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Second Convolutional layer with 64 filters and kernel size of (3, 3)
    model.add(Conv2D(64, (3, 3), activation='relu'))

    # Max Pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Third Convolutional layer with 128 filters and kernel size of (3, 3)
    model.add(Conv2D(128, (3, 3), activation='relu'))

    # Max Pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output to feed it into a fully connected layer
    model.add(Flatten())

    # Fully connected layer with 128 units
    model.add(Dense(512, activation='relu'))

    # Adding dropout to avoid overfitting
    model.add(Dropout(0.2))

    # Output layer with softmax activation function
    model.add(Dense(num_classes, activation='softmax'))

    return model
