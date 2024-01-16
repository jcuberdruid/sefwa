from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import numpy as np

class ChannelLDA:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.training_accuracy = None
        self.validation_accuracy = None
        self.testing_accuracy = None

    def train(self, class_1_data, class_2_data):
        # Generate labels
        labels_class_1 = np.zeros(len(class_1_data))
        labels_class_2 = np.ones(len(class_2_data))

        # Combine data and labels
        X_train = np.vstack((class_1_data, class_2_data))
        y_train = np.concatenate((labels_class_1, labels_class_2))

        # Train the LDA
        self.model.fit(X_train, y_train)

        # Calculate training accuracy
        predictions = self.model.predict(X_train)
        self.training_accuracy = accuracy_score(y_train, predictions)

    def validate(self, class_1_data, class_2_data):
        # Similar to train method, but for validation data
        labels_class_1 = np.zeros(len(class_1_data))
        labels_class_2 = np.ones(len(class_2_data))

        X_val = np.vstack((class_1_data, class_2_data))
        y_val = np.concatenate((labels_class_1, labels_class_2))

        predictions = self.model.predict(X_val)
        self.validation_accuracy = accuracy_score(y_val, predictions)

    def test(self, class_1_data, class_2_data):
        # Similar to train method, but for testing data
        labels_class_1 = np.zeros(len(class_1_data))
        labels_class_2 = np.ones(len(class_2_data))

        X_test = np.vstack((class_1_data, class_2_data))
        y_test = np.concatenate((labels_class_1, labels_class_2))

        predictions = self.model.predict(X_test)
        self.testing_accuracy = accuracy_score(y_test, predictions)

    def get_training_accuracy(self):
        return self.training_accuracy

    def get_validation_accuracy(self):
        return self.validation_accuracy

    def get_testing_accuracy(self):
        return self.testing_accuracy

