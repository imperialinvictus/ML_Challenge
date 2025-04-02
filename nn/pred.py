import pandas as pd

from inference import NN, BaggedNN
import sys
sys.path.insert(0, '..')
from text_cluster.standalone_data_clean import get_dataframe_from_file

def predict_all(filename, models):
    """
    Make predictions for the data in filename using a list of models
    """
    inputs = get_dataframe_from_file(filename, "../text_cluster/", has_labels=False)
    predictions = {}
    
    for model_name, model_path in models.items():
        model = NN(model_path) if 'bagged' not in model_name.lower() else BaggedNN(model_path)
        model_predictions = model.predict(inputs)
        predictions[model_name] = model_predictions

    # TODO: delete the following block when submitting
    expected = pd.read_csv('/dataset/ai_test_y.csv')
    for model_name, model_predictions in predictions.items():
        correct = sum(model_predictions[i] == expected.iloc[i, 1] for i in range(len(model_predictions)))
        accuracy = correct / len(model_predictions)
        print(f"{model_name} Accuracy: {accuracy:.2f}")
    
    return predictions


if __name__ == '__main__':
    models = {
        "BaseNN": "models/mlp_model.npz",
        "BaggedNN": "models/bagged_model.npz",
        "NormBaseNN": "models/normMlp_model.npz",
        "NormBaggedNN": "models/normBagged_model.npz",
        "TunedBaseNN": "models/best_mlp_model.npz",
        "TunedBaggedNN": "models/best_bagged_model.npz",
        "TunedNormBaseNN": "models/best_normMlp_model.npz",
        "TunedNormBaggedNN": "models/best_normBagged_model.npz"
    }
    predict_all('dataset/ai_test.csv', models)