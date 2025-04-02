from inference import NN, BaggedNN
from text_cluster.standalone_data_clean import get_dataframe_from_file

def predict_all(filename):
    inputs = get_dataframe_from_file(filename, "text_cluster", has_labels=False)
    
    model = BaggedNN('models/bagged_model.npz')
    return model.predict(inputs)
