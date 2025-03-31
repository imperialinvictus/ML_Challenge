import itertools
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import loguniform, uniform, randint

import csv
import os



#Adding this one just as proof of work unless we might need it for some reason
class ModelParameterTuner:
    def __init__(self, X_train, y_train, X_valid, y_valid, results_file='model_parameter_tuning.csv'):
        """
        Initialize the parameter tuning process
        
        Parameters:
        - X_train: Training features
        - y_train: Training labels
        - X_valid: Validation features
        - y_valid: Validation labels
        - results_file: CSV file to log parameter tuning results
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.results_file = results_file
        
        # Define parameter grid for comprehensive testing
        self.param_grid = {

            #TOO MANY DAMN PRAMATERS 
            'hidden_layer_sizes': [
                (50,), (100,), (50, 50), (100, 50), (100, 100),
                (25, 25), (75, 25), (75, 75)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': loguniform(1e-4, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-2),
            'learning_rate': ['adaptive'],
            'bagging_estimators': [10],
            'bagging_samples': [0.6, 0.8]
        }
        
        # Prepare results file
        self._prepare_results_file()
    
    def _prepare_results_file(self):
        """
        Prepare the CSV file for logging results
        """
        fieldnames = [
            'Model_Type', 'Hidden_Layers', 'Activation', 
            'Alpha', 'Learning_Rate_Init', 'Learning_Rate', 
            'Bagging_Estimators', 'Bagging_Samples', 
            'Train_Score', 'Valid_Score'
        ]
        
        # Create file if it doesn't exist
        if not os.path.exists(self.results_file):
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    
    def _log_results(self, model_type, params, train_score, valid_score):
        """
        Log model parameters and performance to CSV
        """
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'Model_Type', 'Hidden_Layers', 'Activation', 
                'Alpha', 'Learning_Rate_Init', 'Learning_Rate', 
                'Bagging_Estimators', 'Bagging_Samples', 
                'Train_Score', 'Valid_Score'
            ])
            writer.writerow({
                'Model_Type': model_type,
                'Hidden_Layers': str(params.get('hidden_layer_sizes', 'N/A')),
                'Activation': params.get('activation', 'N/A'),
                'Alpha': params.get('alpha', 'N/A'),
                'Learning_Rate_Init': params.get('learning_rate_init', 'N/A'),
                'Learning_Rate': params.get('learning_rate', 'N/A'),
                'Bagging_Estimators': params.get('n_estimators', 'N/A'),
                'Bagging_Samples': params.get('max_samples', 'N/A'),
                'Train_Score': train_score,
                'Valid_Score': valid_score
            })
    
    def tune_models(self):
        """
        Systematically tune and test different model configurations
        """
        # Generate all possible parameter combinations
        param_combinations = list(itertools.product(
            self.param_grid['hidden_layer_sizes'],
            self.param_grid['activation'],
            self.param_grid['alpha'],
            self.param_grid['learning_rate_init'],
            self.param_grid['learning_rate'],
            self.param_grid['bagging_estimators'],
            self.param_grid['bagging_samples']
        ))
        
        best_score = 0
        best_model = None
        
        for (hidden_size, activation, alpha, 
             learning_rate_init, learning_rate, 
             n_estimators, max_samples) in param_combinations:
            
            # Create base MLP model
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_size,
                activation=activation,
                alpha=alpha,
                learning_rate_init=learning_rate_init,
                learning_rate=learning_rate,
                random_state=42,
                max_iter=500,
                early_stopping=True
            )
            
            # Regular MLP
            mlp.fit(self.X_train, self.y_train)
            train_score = mlp.score(self.X_train, self.y_train)
            valid_score = mlp.score(self.X_valid, self.y_valid)
            
            self._log_results('Base_MLP', mlp.get_params(), train_score, valid_score)
            
            # Bagged MLP
            bagged_mlp = BaggingClassifier(mlp,n_estimators=n_estimators,max_samples=max_samples, random_state=42)
            bagged_mlp.fit(self.X_train, self.y_train)
            bagged_train_score = bagged_mlp.score(self.X_train, self.y_train)
            bagged_valid_score = bagged_mlp.score(self.X_valid, self.y_valid)
            
            self._log_results('Bagged_MLP', 
                              {**bagged_mlp.estimator_.get_params(), 
                               'n_estimators': n_estimators, 
                               'max_samples': max_samples}, 
                              bagged_train_score, 
                              bagged_valid_score)
            
            # Update best model if needed
            if valid_score > best_score:
                best_score = valid_score
                best_model = mlp
        
        return best_model

# Workflow function remains the same
def parameter_tuning_workflow(X_train, y_train, X_valid, y_valid):
    """
    Workflow to perform parameter tuning
    """
    tuner = ModelParameterTuner(X_train, y_train, X_valid, y_valid)
    best_model = tuner.tune_models()
    
    # Optional: display summary of results
    results_df = pd.read_csv('results/model_parameter_tuning.csv')
    print("Tuning Results Summary:")
    print(results_df.groupby('Model_Type')[['Train_Score', 'Valid_Score']].describe())
    
    return best_model