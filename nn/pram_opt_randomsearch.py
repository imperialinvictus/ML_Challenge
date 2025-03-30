import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.stats import loguniform, uniform, randint
import os
import csv
from typing import Dict, Any, Tuple

class RandomizedSearchModelOptimizer:
    def __init__(self, X_train, y_train, X_valid, y_valid, X_test, y_test, columns_to_scale=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_valid = X_valid
        self.y_valid = y_valid
        self.X_test = X_test
        self.y_test = y_test
        
        # Default columns to scale (first three columns as specified)
        self.columns_to_scale = columns_to_scale if columns_to_scale is not None else [0, 1, 2]
        
        # Create normalizer using ColumnTransformer
        self.normalizer = ColumnTransformer(
            transformers=[
                ('scaler', StandardScaler(), self.columns_to_scale)
            ],
            remainder='passthrough'  # Leave other columns unchanged
        )
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Define comprehensive parameter search space
        self.param_dist = {
            'hidden_layer_sizes': [
                (50,), (100,), (50, 50), (100, 50), (100, 100),
                (25, 25), (75, 25), (75, 75), (100, 100, 100), (75, 75, 75)
            ],
            'activation': ['relu', 'tanh', 'logistic'],
            'solver': ['adam', 'sgd', 'lbfgs'],
            'alpha': loguniform(1e-4, 1e-1),
            'learning_rate_init': loguniform(1e-4, 1e-2),
            'learning_rate': ['constant', 'adaptive'],
        }
        
        # Bagging parameters
        self.bagging_param_dist = {
            'n_estimators': [5, 10, 15],
            'max_samples': [0.6, 0.8, 1.0]
        }
        
        # Results tracking file
        self.results_file = 'results/randomized_search_pram.csv'
        
    def create_results_file(self):
        """Create results file with headers if it doesn't exist"""
        if not os.path.exists(self.results_file):
            headers = [
                'Model Type', 'Model Subtype', 'Param Name', 'Param Value', 
                'Training Score', 'Validation Score', 'Test Score', 
                'Bias', 'Variance'
            ]
            with open(self.results_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_model_results(self, model_type: str, model_subtype: str, 
                           params: Dict[str, Any], train_score: float, 
                           valid_score: float, test_score: float,
                           bias: float = None, variance: float = None):
        """Log model results to CSV"""
        with open(self.results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Log each parameter separately for easy filtering/analysis
            for param_name, param_value in params.items():
                writer.writerow([
                    model_type, model_subtype, param_name, str(param_value),
                    train_score, valid_score, test_score, bias, variance
                ])
    
    def normalize_data(self, X):
        """Apply normalization to the data using ColumnTransformer"""
        return self.normalizer.fit_transform(X)
    
    def optimize_mlp(self, normalized: bool = False) -> Tuple[MLPClassifier, Dict[str, Any]]:
        """
        Optimize MLP Classifier with RandomizedSearchCV
        """
        # Select appropriate training data
        X_train = self.X_train if not normalized else self.normalize_data(self.X_train)
        X_valid = self.X_valid if not normalized else self.normalizer.transform(self.X_valid)
        X_test = self.X_test if not normalized else self.normalizer.transform(self.X_test)
        
        # Create base MLP
        mlp = MLPClassifier(
            random_state=1, 
            n_iter_no_change=50, 
            early_stopping=True,
            validation_fraction=0.2
        )
        
        # Create stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        
        # Randomized Search
        random_search = RandomizedSearchCV(
            mlp, 
            param_distributions=self.param_dist, 
            n_iter=100, 
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,  
            random_state=42,
            verbose=2
        )
        
        # Fit the random search
        random_search.fit(X_train, self.y_train)
        
        # Get best model
        best_mlp = random_search.best_estimator_
        
        # Evaluate
        train_score = best_mlp.score(X_train, self.y_train)
        valid_score = best_mlp.score(X_valid, self.y_valid)
        test_score = best_mlp.score(X_test, self.y_test)
        
        # Log results
        model_type = 'MLP'
        model_subtype = 'Normalized' if normalized else 'Base'
        self.log_model_results(
            model_type, 
            model_subtype, 
            random_search.best_params_, 
            train_score, 
            valid_score, 
            test_score
        )
        
        return best_mlp, random_search.best_params_
    
    def optimize_bagging(self, base_mlp: MLPClassifier, normalized: bool = False):
        """
        Optimize Bagging Classifier
        """
        # Select appropriate training data
        X_train = self.X_train if not normalized else self.normalize_data(self.X_train)
        X_valid = self.X_valid if not normalized else self.normalizer.transform(self.X_valid)
        X_test = self.X_test if not normalized else self.normalizer.transform(self.X_test)

        # Create Bagging Classifier
        bagging = BaggingClassifier(
            estimator=base_mlp,  # Set the base estimator
            random_state=1
        )
    
        # Create stratified cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Randomized Search for Bagging
        bagging_random_search = RandomizedSearchCV(
            bagging,
            param_distributions=self.bagging_param_dist, 
            n_iter=30,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            random_state=42,
            verbose=2
        )
        
        # Fit the random search
        bagging_random_search.fit(X_train, self.y_train)
        
        # Get best bagging model
        best_bagging = bagging_random_search.best_estimator_
        
        # Evaluate
        train_score = best_bagging.score(X_train, self.y_train)
        valid_score = best_bagging.score(X_valid, self.y_valid)
        test_score = best_bagging.score(X_test, self.y_test)
        
        # Log results
        model_type = 'Bagging'
        model_subtype = 'Normalized' if normalized else 'Base'
        self.log_model_results(
            model_type, 
            model_subtype, 
            bagging_random_search.best_params_, 
            train_score, 
            valid_score, 
            test_score
        )
        
        return best_bagging, bagging_random_search.best_params_