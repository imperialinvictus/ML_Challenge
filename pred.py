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

class NN:
    """
    Class for forward pass through the neural network
    """
    def __init__(self, weights, biases):
        self.weights = weights
        self.biases = biases

    def make_NN(filename):
        """
        Load the model parameters from the file and return a neural network object
        """
        params = np.load('nn/model_params.npz')
        num_layers = len([key for key in params.keys() if key.startswith('weights_')])
        weights = [params[f'weights_{i}'] for i in range(num_layers)]
        biases = [params[f'biases_{i}'] for i in range(num_layers)]
        return NN(weights, biases)

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
        for i in range(len(self.weights)):
            X = np.dot(X, self.weights[i]) + self.biases[i]
            if i < len(self.weights) - 1:
                X = self.ReLU(X)
            else:
                X = X.astype(np.float64)
                X = self.softmax(X)
        
        return np.argmax(X, axis=1) # return the class with the highest probability

def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    nn = NN.make_NN('nn/model_params.npz')

    inputs = pd.read_csv(filename)
    predictions = nn.predict(inputs)
    
    expected = pd.read_csv('nn/y_valid.csv') # TODO: delete when submitting
    correct = sum(predictions[i] == expected.iloc[i, 0] for i in range(len(predictions)))
    accuracy = correct / len(predictions)
    print(f"Accuracy: {accuracy:.2f}")
    
    return predictions
    

if __name__ == '__main__':
    predict_all('nn/X_valid.csv')