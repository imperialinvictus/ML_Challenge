import pandas as pd

from nn.inference import NN, BaggedNN

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

def predict_all(filename):
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
    predict_all('nn/X_valid.csv')