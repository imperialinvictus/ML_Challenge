import numpy as np
import pandas as pd

class NN:
    """
    Class for forward pass through the neural network
    """
    def __init__(self, filename):
        params = np.load(filename)
        self.num_layers = len([key for key in params.keys() if key.startswith('weights_')])
        self.weights = [params[f'weights_{i}'] for i in range(self.num_layers)]
        self.biases = [params[f'biases_{i}'] for i in range(self.num_layers)]

    def softmax(self, x) -> np.ndarray:
        """
        Compute softmax values for each sets of scores in X.
        """
        m = np.max(x, axis=1, keepdims=True)
        s = np.sum(np.exp(x - m), axis=1, keepdims=True)
        return np.exp(x - m) / s
    
    def ReLU(self, x) -> np.ndarray:
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