import pandas as pd

from inference import NN, BaggedNN
import sys
sys.path.insert(0, '..')
from text_cluster.standalone_data_clean import get_dataframe_from_file


def predict_all(filename):
    """
    Make predictions for the data in filename
    """
    nn = NN('models/mlp_model.npz')
    baggedNN = BaggedNN('models/normBagged_model.npz')

    inputs = get_dataframe_from_file(filename, "../text_cluster/", has_labels=False)
    predictions = nn.predict(inputs)
    baggedPredictions = baggedNN.predict(inputs)
    
    expected = pd.read_csv('dataset/ai_test_y.csv') # TODO: delete when submitting
    baseCorrect = sum(predictions[i] == expected.iloc[i, 1] for i in range(len(predictions)))
    baseAccuracy = baseCorrect / len(predictions)
    baggedCorrect = sum(baggedPredictions[i] == expected.iloc[i, 1] for i in range(len(predictions)))
    baggedAccuracy = baggedCorrect / len(predictions)
    
    print(f"Base Accuracy: {baseAccuracy:.2f}")
    print(f"Bagged Accuracy: {baggedAccuracy:.2f}")
    
    return predictions
    

if __name__ == '__main__':
    predict_all('dataset/ai_test.csv')