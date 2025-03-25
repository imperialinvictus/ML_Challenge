"""
This Python file is example of how your `pred.py` script should
look. Your file should contain a function `predict_all` that takes
in the name of a CSV file, and returns a list of predictions.

Your `pred.py` script can use different methods to process the input
data, but the format of the input it takes and the output your script produces should be the same.

Here's an example of how your script may be used in our test file:

    from example_pred import predict_all
    predict_all("example_test_set.csv")
"""

# basic python imports are permitted
import sys
import csv
import random

# numpy and pandas are also permitted
import numpy as np
import pandas as pd

from text_cluster.standalone_data_clean import get_dataframe_from_file


class DataCleaner:
    """
    A class to clean and preprocess the data
    TODO: complete this class
    """
    categorical_questions = {
        "Q3": 'Q3: In what setting would you expect this food to be served? Please check all that apply',
        "Q5": 'Q5: What movie do you think of when thinking of this food item?',
        "Q7": 'Q7: When you think about this food item, who does it remind you of?',
        "Q8": 'Q8: How much hot sauce would you add to this food item?',
        "Q6": 'Q6: What drink would you pair with this food item?'
    }

    def clean_data(X: pd.DataFrame):
        df = X.dropna()
        df = df.drop(columns=['id'])
        return df

    def encode_data(X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode categorical features
        """
        return X
    
    def filename_to_dataframe(filename: str) -> pd.DataFrame:
        """
        Parses from raw csv file to categories using string clustering and flattens to one-hot encodings 
        """
        return get_dataframe_from_file(filename, 'text_cluster', fuzzy_cutoff=85)

class NN:
    """
    Class for forward pass through the neural network
    """
    def __init__(self, filename):
        params = np.load(filename)
        self.num_layers = len([key for key in params.keys() if key.startswith('weights_')])
        self.weights = [params[f'weights_{i}'] for i in range(self.num_layers)]
        self.biases = [params[f'biases_{i}'] for i in range(self.num_layers)]

    def softmax(self, x):
        """
        Compute softmax values for each sets of scores in X.
        """
        m = np.max(x, axis=1, keepdims=True)
        s = np.sum(np.exp(x - m), axis=1, keepdims=True)
        return np.exp(x - m) / s

    def ReLU(self, x):
        """
        Compute ReLU activation function
        """
        return np.maximum(0, x)

    def predict(self, X):
        """
        Perform a forward pass through the network
        """
        for i in range(self.num_layers):
            X = np.dot(X, self.weights[i]) + self.biases[i]
            if i < self.num_layers - 1:
                X = self.ReLU(X)
            else:
                X = X.astype(np.float64)
                X = self.softmax(X)

        return np.argmax(X, axis=1) # return the class with the highest probability

class BaggedNN(NN):
    """
    Class for bagged neural network
    """
    classes = [0, 1, 2]

    def __init__(self, filename):
        params = np.load(filename)
        self.num_estimators = params['num_estimators']
        self.num_layers = params['num_layers']
        self.estimators_params = []
        for i in range(self.num_estimators):
            estimator_params = {}
            estimator_params['weights'] = [params[f'estimator_{i}_weights_{j}'] for j in range(self.num_layers)]
            estimator_params['intercepts'] = [params[f'estimator_{i}_intercepts_{j}'] for j in range(self.num_layers)]
            self.estimators_params.append(estimator_params)

    def predict(self, X):
        all_predictions = np.zeros((X.shape[0], self.num_estimators), dtype=int)
        for i, params in enumerate(self.estimators_params):
            h = X  # Start with original input for each estimator
            for j in range(self.num_layers):
                h = np.dot(h, params['weights'][j]) + params['intercepts'][j]
                if j < self.num_layers - 1:
                    h = super().ReLU(h)
                else:
                    h = h.astype(np.float64)
                    probs = super().softmax(h)
            predictions = np.argmax(probs, axis=1)
            all_predictions[:, i] = predictions

        # Majority voting
        final_predictions = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=len(self.classes)).argmax(),
            axis=1,
            arr=all_predictions
        )
        return final_predictions


def predict_all(filename):
    """
    Make predictions for the data in filename using text-clustered encoding
    """
    nn = NN('nn/model_params.npz')
    baggedNN = BaggedNN('nn/bagged_model_params.npz')

    encoded_inputs = DataCleaner.filename_to_dataframe(filename)
    predictions = nn.predict(encoded_inputs)
    bagged_predictions = baggedNN.predict(encoded_inputs)

    remap = {0: "Pizza", 1: "Shawarma", 2: "Sushi"}
    mapped_predictions = [remap[p] for p in predictions]
    mapped_bagged_predictions = [remap[p] for p in bagged_predictions]

    # expected = pd.read_csv('text_cluster/example_test_y.csv')  # TODO: delete when submitting
    # baseCorrect = sum(mapped_predictions[i] == expected.iloc[i, 0] for i in range(len(predictions)))
    # baseAccuracy = baseCorrect / len(predictions)
    # baggedCorrect = sum(mapped_bagged_predictions[i] == expected.iloc[i, 0] for i in range(len(predictions)))
    # baggedAccuracy = baggedCorrect / len(predictions)

    # print(f"Base Accuracy: {baseAccuracy:.2f}")
    # print(f"Bagged Accuracy: {baggedAccuracy:.2f}")

    return mapped_bagged_predictions


def predict_all_old(filename):
    """
    Make predictions for the data in filename
    """
    nn = NN('nn/model_params.npz')
    baggedNN = BaggedNN('nn/bagged_model_params.npz')

    inputs = pd.read_csv(filename)
    predictions = nn.predict(inputs)
    baggedPredictions = baggedNN.predict(inputs)

    expected = pd.read_csv('nn/y_valid.csv') # TODO: delete when submitting
    baseCorrect = sum(predictions[i] == expected.iloc[i, 0] for i in range(len(predictions)))
    baseAccuracy = baseCorrect / len(predictions)
    baggedCorrect = sum(baggedPredictions[i] == expected.iloc[i, 0] for i in range(len(predictions)))
    baggedAccuracy = baggedCorrect / len(predictions)

    print(f"Base Accuracy: {baseAccuracy:.2f}")
    print(f"Bagged Accuracy: {baggedAccuracy:.2f}")

    return predictions
    

if __name__ == '__main__':
    print(predict_all('text_cluster/example_test.csv'))