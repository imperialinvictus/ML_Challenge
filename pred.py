import numpy as np
import pandas as pd

from text_cluster.standalone_data_clean import get_dataframe_from_file
from randomforest_dump import *

def predict_all(filename: str) -> list:
    """
    Predict classes for a test set using pre-dumped random forest trees
    
    :param test_csv_path: Path to the CSV file containing test data
    :return: List of predictions
    """
    
    encoded_input = get_dataframe_from_file(filename, folder_path='text_cluster', has_labels=False)
    
    # Collect all tree functions
    tree_functions = [
        globals()[f'tree{i}'] 
        for i in range(0, 250)
    ]
    
    predictions = []
    
    for _, row in encoded_input.iterrows():
        tree_predictions = [
            tree_func(*row.values) 
            for tree_func in tree_functions
        ]
        
        prob_sum = np.sum(tree_predictions, axis=0)
        
        # Get the class with the highest total probability
        prediction = np.argmax(prob_sum)
        predictions.append(int(prediction))
    
    return predictions

remap = {0: "Pizza", 1: "Shawarma", 2: "Sushi"}
predictions = predict_all('text_cluster/example_test.csv')
expected = pd.read_csv("text_cluster/example_test_y.csv")
print(predict_all('text_cluster/example_test.csv'))
correct = [remap[predictions[i]] == expected.iloc[i, 0] for i in range(len(predictions))]
print(correct, sum(correct)/len(correct))