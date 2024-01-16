from tensorflow.keras import layers, models, regularizers
import numpy as np
import tensorflow as tf

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, num_labels=1):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for units in mlp_units:
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(num_labels, activation='sigmoid')(x)
    return models.Model(inputs, outputs)


class ChannelTransformer:
    def __init__(self, channel_index, num_labels=1):
        self.channel_index = channel_index + 5
        self.num_labels = num_labels
        self.model = self._build_model()
        self.training_accuracy = None
        self.validation_accuracy = None
        self.testing_accuracy = None

    def _build_model(self):
        # Define model parameters
        input_shape = (1, 800)
        head_size = 800
        num_heads = 10 
        ff_dim = 128
        num_transformer_blocks = 3
        mlp_units = [64]

        # Build and compile the transformer model
        model = build_transformer_model(
            input_shape=input_shape,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units=mlp_units,
            dropout=0.01,
            mlp_dropout=0.01,
        )
            #num_labels=self.num_labels
        model.compile(optimizer="adam", loss="binary_crossentropy",metrics=['accuracy'])
        return model

    def train(self, class_1_data, class_2_data, epochs=30, batch_size=64):
        X, y = self._prepare_data(class_1_data, class_2_data)
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        # Optionally calculate training accuracy
        self.training_accuracy = np.mean(history.history['accuracy'])

    def validate(self, class_1_data, class_2_data):
        X, y = self._prepare_data(class_1_data, class_2_data)
        self.validation_accuracy = self.model.evaluate(X, y)[1]

    def test(self, class_1_data, class_2_data):
        X, y = self._prepare_data(class_1_data, class_2_data)
        self.testing_accuracy = self.model.evaluate(X, y)[1]

    def predict(self, data):
        X = self._prepare_predict_data(data)
        # Get prediction probabilities
        probabilities = self.model.predict(X)

        # Convert probabilities to class labels
        # Assuming binary classification and using 0.5 as threshold
        predictions = (probabilities > 0.5).astype(int)
        predictions = predictions.squeeze()  # Convert from shape (n, 1) to (n,)

        return predictions
    def _prepare_predict_data(self, data):
        # Extract the specific channel data
        X = data[:, :, self.channel_index]
        # Reshape X to 3D array where each sample is 1x40
        X = X.reshape(-1, 1, 800).astype('float32')
        return X

    def _prepare_data(self, class_1_data, class_2_data):
        # Extract the specific channel data and create labels
        X1 = class_1_data[:, :, self.channel_index]
        X2 = class_2_data[:, :, self.channel_index]
        y1 = np.zeros(class_1_data.shape[0])
        y2 = np.ones(class_2_data.shape[0])

        # Combine data
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)

        # Reshape X to 3D array where each sample is 1x40
        # Each sample will have shape (1, 40) representing 1 channel with 40 time steps
        X = X.reshape(-1, 1, 800).astype('float32')
        y = y.astype('float32')
        return X, y

    # Accessor methods for accuracies
    def get_training_accuracy(self):
        return self.training_accuracy

    def get_validation_accuracy(self):
        return self.validation_accuracy

    def get_testing_accuracy(self):
        return self.testing_accuracy
