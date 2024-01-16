from sklearn import svm
import numpy as np

class ChannelSVM:
    def __init__(self, channel_index):
        self.channel_index = channel_index + 5
        self.model = svm.SVC()  # You can customize the SVM parameters here
        self.training_accuracy = None
        self.validation_accuracy = None
        self.testing_accuracy = None

    def train(self, class_1_data, class_2_data):
        # Preprocess and combine data
        X, y = self._prepare_data(class_1_data, class_2_data)
        # Train the SVM
        self.model.fit(X, y)
        # Optionally calculate training accuracy
        self.training_accuracy = self.model.score(X, y)

    def validate(self, class_1_data, class_2_data):
        X, y = self._prepare_data(class_1_data, class_2_data)
        self.validation_accuracy = self.model.score(X, y)

    def test(self, class_1_data, class_2_data):
        X, y = self._prepare_data(class_1_data, class_2_data)
        self.testing_accuracy = self.model.score(X, y)

    def _prepare_data(self, class_1_data, class_2_data):
        # Extract the specific channel data and create labels
        X1 = class_1_data[:, :, self.channel_index]
        X2 = class_2_data[:, :, self.channel_index]
        y1 = np.zeros(class_1_data.shape[0])
        y2 = np.ones(class_2_data.shape[0])
        # Combine and reshape data
        X = np.concatenate((X1, X2), axis=0)
        y = np.concatenate((y1, y2), axis=0)
        # Reshape X to 2D array where each row is a sample and each column is a feature
        X = X.reshape(-1, X.shape[1])  # Reshaping to (n, 961)
        return X, y

    def predict(self, data):
        # Preprocess the data
        X = self._prepare_predict_data(data)
        # Use the trained model to make predictions
        predictions = self.model.predict(X)
        return predictions

    def _prepare_predict_data(self, data):
        # Extract the specific channel data
        X = data[:, :, self.channel_index]
        # Reshape X to 2D array where each row is a sample and each column is a feature
        X = X.reshape(-1, X.shape[1])  # Reshaping to (n, 961 or the number of features)
        return X

    # Accessor methods for accuracies
    def get_training_accuracy(self):
        return self.training_accuracy

    def get_validation_accuracy(self):
        return self.validation_accuracy

    def get_testing_accuracy(self):
        return self.testing_accuracy

