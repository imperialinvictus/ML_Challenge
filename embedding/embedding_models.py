import argparse
import os
import re
from enum import Enum, IntEnum

import numpy as np
import pandas as pd
from model2vec import StaticModel
from sklearn.linear_model import LinearRegression


class FoodEnum(IntEnum):
    """Enum for food categories"""

    PIZZA = 0
    SHAWARMA = 1
    SUSHI = 2

    @classmethod
    def from_label(cls, label: str) -> int:
        """Convert label to enum value"""
        labels = {
            "pizza": cls.PIZZA,
            "shawarma": cls.SHAWARMA,
            "sushi": cls.SUSHI,
        }

        label = label.lower()
        assert label in labels, f"Unknown label: {label}"
        return labels[label]

    def to_label(self) -> str:
        """Convert enum value to label"""
        labels = {
            self.PIZZA: "Pizza",
            self.SHAWARMA: "Shawarma",
            self.SUSHI: "Sushi",
        }

        return labels.get(self, "Unknown")


class ModelType(Enum):
    """Enum for model types"""

    NEURAL_NETWORK = "NeuralNetwork"
    LINEAR_REGRESSION = "LinearRegression"


class TextEncoder:
    """Encodes text into feature vectors using character n-grams"""

    model = StaticModel.from_pretrained("minishlab/potion-base-2M")

    def __init__(self, food_embeddings: list[np.ndarray] | None = None):
        self.food_embeddings = food_embeddings or [np.zeros(0) for _ in range(len(FoodEnum))]
        assert len(self.food_embeddings) == len(FoodEnum), "Food embeddings must match food categories"

    def get_embedding(self, text) -> np.ndarray:
        """Get the embedding for a given text using character n-grams"""
        return self.model.encode(text)

    def create_food_embeddings(self, texts_by_food: list[list[str]]) -> list[np.ndarray]:
        """Create embeddings for each food category based on example texts"""
        for food_idx, texts in enumerate(texts_by_food):
            food = FoodEnum(food_idx)
            if not texts:
                self.food_embeddings[food] = np.zeros(0)
                continue

            embeddings = [self.get_embedding(text) for text in texts]
            self.food_embeddings[food] = np.mean(embeddings, axis=0)

        return self.food_embeddings

    def get_similarity(self, text: str) -> np.ndarray:
        """Get similarity scores for a given text against food categories"""
        text_embedding = self.get_embedding(text)
        similarities = []

        for food in FoodEnum:
            food_embedding = self.food_embeddings[food]
            if food_embedding.size == 0:
                similarities.append(0.0)
                continue

            # Cosine similarity calculation
            similarity = np.dot(text_embedding, food_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(food_embedding) + 1e-10
            )
            similarities.append(similarity)

        return np.array(similarities)


class DataProcessor:
    """Preprocesses data for neural network input"""

    col_mappings = {
        "Q1": "Q1: From a scale 1 to 5, how complex is it to make this food? (Where 1 is the most simple, and 5 is the most complex)",
        "Q2": "Q2: How many ingredients would you expect this food item to contain?",
        "Q3": "Q3: In what setting would you expect this food to be served? Please check all that apply",
        "Q4": "Q4: How much would you expect to pay for one serving of this food item?",
        "Q5": "Q5: What movie do you think of when thinking of this food item?",
        "Q6": "Q6: What drink would you pair with this food item?",
        "Q7": "Q7: When you think about this food item, who does it remind you of?",
        "Q8": "Q8: How much hot sauce would you add to this food item?",
    }

    def __init__(self, preprocess_params: dict | None = None):
        self.preprocess_params = preprocess_params or {}
        self.text_encoders = {
            "Q5": TextEncoder(),
            "Q6": TextEncoder(),
            "Q7": TextEncoder(),
        }

    def clean_numeric(self, value: np.ndarray | str | float, feature: str | None = None) -> float:
        """Convert string to numeric value"""
        if not value or (isinstance(value, float) and np.isnan(value)):
            return 0.0

        if not isinstance(value, str):
            try:
                return float(value)
            except (ValueError, TypeError):
                return 0.0

        # Map word numbers to digits
        # fmt: off
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
            'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
            'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
            'thirty': 30, 'forty': 40, 'fifty': 50
        }
        # fmt: on

        # Convert to lowercase for word matching
        text = value.lower()

        # Extract numbers (both digits and words)
        numbers = []

        # Check for word numbers
        for word, num in word_to_num.items():
            if word in text:
                numbers.append(num)

        # Check for digit numbers using regex
        digit_matches = re.findall(r"\d+", text)
        for match in digit_matches:
            numbers.append(int(match))

        # If we found any numbers, return the average
        if numbers:
            return sum(numbers) / len(numbers)
        return 0

    def normalize_feature(self, values: np.ndarray, feature: str) -> np.ndarray:
        """Normalize numeric features using min-max scaling"""
        min_val = np.min(values)
        max_val = np.max(values)

        # Store for inference
        self.preprocess_params[f"{feature}_min"] = min_val
        self.preprocess_params[f"{feature}_max"] = max_val

        if max_val <= min_val:
            return np.zeros_like(values)
        return (values - min_val) / (max_val - min_val)

    def process_categorical(self, values: np.ndarray, feature: str):
        """Process categorical features using one-hot encoding"""
        all_options = set()

        # Find all possible options
        for val in values:
            if isinstance(val, str):
                options = [opt.strip() for opt in val.split(",")]
                all_options.update(options)

        # Sort options for consistency
        all_options = sorted(list(all_options))

        # Store for inference
        self.preprocess_params[f"{feature}_options"] = all_options

        result = np.zeros((len(values), len(all_options)))

        for i, val in enumerate(values):
            if not isinstance(val, str):
                continue

            for j, option in enumerate(all_options):
                if option in val:
                    result[i, j] = 1

        return result

    # def process_text(self, values, feature, labels=None):
    def process_text(self, values: np.ndarray, feature: str, labels: np.ndarray | None = None):
        """Process text features using text encoder"""
        encoder = self.text_encoders[feature]

        # Training mode with labels
        if labels is not None:
            # Group texts by food label
            texts_by_food = [[] for _ in range(len(FoodEnum))]
            for i, (val, label) in enumerate(zip(values, labels)):
                if isinstance(val, str):
                    label_idx = np.argmax(label)
                    food = FoodEnum(label_idx)
                    texts_by_food[food].append(val.strip())

            # Create food embeddings
            food_embeddings = encoder.create_food_embeddings(texts_by_food)

            # Store for inference
            self.preprocess_params[f"{feature}_embeddings"] = food_embeddings
        else:
            # Use existing embeddings for inference
            encoder.food_embeddings = self.preprocess_params.get(f"{feature}_embeddings", {})

        # Calculate distances
        result = np.zeros((len(values), 3))
        for i, val in enumerate(values):
            if isinstance(val, str):
                result[i] = encoder.get_similarity(val.strip())

        return result

    def preprocess(self, df: pd.DataFrame, labels: np.ndarray | None = None, train_mode: bool = False):
        """Preprocess dataframe for neural network input"""
        # Map column names if necessary
        df_mapped = df.copy()
        for short_name, full_name in self.col_mappings.items():
            if full_name in df.columns and short_name not in df.columns:
                df_mapped[short_name] = df[full_name]
            elif short_name in df.columns:
                df_mapped[short_name] = df[short_name]

        features = []

        # Process numeric features
        for feature in ["Q1", "Q2", "Q4"]:
            if feature in df_mapped.columns:
                values = np.array([self.clean_numeric(v, feature) for v in df_mapped[feature]])
                if train_mode:
                    normalized = self.normalize_feature(values, feature)
                else:
                    min_val = self.preprocess_params.get(f"{feature}_min", 0)
                    max_val = self.preprocess_params.get(f"{feature}_max", 1)
                    normalized = (values - min_val) / (max_val - min_val)
                features.append(normalized.reshape(-1, 1))

        # Process categorical features
        for feature in ["Q3", "Q8"]:
            if feature in df_mapped.columns:
                if train_mode:
                    categorical = self.process_categorical(df_mapped[feature].to_numpy(), feature)
                else:
                    options = self.preprocess_params.get(f"{feature}_options", [])
                    categorical = np.zeros((len(df_mapped), len(options)))
                    for i, val in enumerate(df_mapped[feature]):
                        if not isinstance(val, str):
                            continue
                        for j, option in enumerate(options):
                            if option.lower() in val.lower():
                                categorical[i, j] = 1
                features.append(categorical)

        # Process text features
        for feature in ["Q5", "Q6", "Q7"]:
            if feature in df_mapped.columns:
                if train_mode and labels is not None:
                    text_features = self.process_text(df_mapped[feature].to_numpy(), feature, labels)
                else:
                    text_features = self.process_text(df_mapped[feature].to_numpy(), feature)
                features.append(text_features)

        # Combine all features
        if not features:
            return np.zeros((df.shape[0], 1))
        return np.hstack(features)


class NeuralNetwork:
    """Neural network implementation with training and inference capabilities"""

    weights: list[np.ndarray]
    biases: list[np.ndarray]
    input_size: int
    hidden_sizes: list[int]
    output_size: int
    learning_rate: float

    def __init__(
        self,
        input_size: int | None = None,
        hidden_sizes: list[int] | None = None,
        output_size: int | None = None,
        learning_rate: float = 0.01,
        params: dict | None = None,
    ):
        """Initialize neural network with specified architecture or from parameters"""
        self.learning_rate = learning_rate

        if params is not None:
            # Initialize from saved parameters (for inference)
            self.weights = []
            self.biases = []

            i = 0
            while True:
                weight_key = f"weights_{i}"
                bias_key = f"biases_{i}"
                if weight_key in params:
                    self.weights.append(params[weight_key])
                if bias_key in params:
                    self.biases.append(params[bias_key])
                if weight_key not in params and bias_key not in params:
                    break
                i += 1

            self.input_size = self.weights[0].shape[0]
            self.hidden_sizes = [w.shape[1] for w in self.weights[:-1]]
            self.output_size = self.weights[-1].shape[1]

        else:
            assert input_size is not None, "Input size must be specified"
            assert hidden_sizes is not None, "Hidden sizes must be specified"
            assert output_size is not None, "Output size must be specified"

            # Initialize weights and biases with He initialization
            self.weights = []
            self.biases = []
            self.input_size = input_size
            self.hidden_sizes = hidden_sizes
            self.output_size = output_size

            # Input layer to first hidden layer
            self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2 / input_size))
            self.biases.append(np.zeros((1, hidden_sizes[0])))

            # Hidden layers
            for i in range(1, len(hidden_sizes)):
                self.weights.append(
                    np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * np.sqrt(2 / hidden_sizes[i - 1])
                )
                self.biases.append(np.zeros((1, hidden_sizes[i])))

            # Last hidden layer to output layer
            self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * np.sqrt(2 / hidden_sizes[-1]))
            self.biases.append(np.zeros((1, output_size)))

    def relu(self, x: np.ndarray):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x: np.ndarray):
        """Derivative of ReLU"""
        return (x > 0).astype(float)

    def softmax(self, x: np.ndarray):
        """Softmax activation function"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X: np.ndarray):
        """Forward pass through the network"""
        activations = [X]
        z_values = []

        # Hidden layers with ReLU
        for i in range(len(self.weights) - 1):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            z_values.append(z)
            a = self.relu(z)
            activations.append(a)

        # Output layer with softmax
        z = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        z_values.append(z)
        output = self.softmax(z)
        activations.append(output)

        return activations, z_values

    def predict(self, X: np.ndarray):
        """Forward pass through the network (for inference)"""
        activations, _ = self.forward(X)
        return activations[-1]

    def backward(self, X: np.ndarray, y: np.ndarray, activations: list[np.ndarray], z_values: list[np.ndarray]):
        """Backward pass to update weights"""
        m = X.shape[0]

        # Output layer error
        delta = activations[-1] - y

        # Initialize gradients
        d_weights = [np.zeros_like(w) for w in self.weights]
        d_biases = [np.zeros_like(b) for b in self.biases]

        # Output layer gradients
        d_weights[-1] = np.dot(activations[-2].T, delta) / m
        d_biases[-1] = np.sum(delta, axis=0, keepdims=True) / m

        # Hidden layers
        for l in range(len(self.hidden_sizes), 0, -1):
            delta = np.dot(delta, self.weights[l].T) * self.relu_derivative(z_values[l - 1])
            d_weights[l - 1] = np.dot(activations[l - 1].T, delta) / m
            d_biases[l - 1] = np.sum(delta, axis=0, keepdims=True) / m

        # Clip gradients to prevent exploding gradients
        for i in range(len(d_weights)):
            d_weights[i] = np.clip(d_weights[i], -1.0, 1.0)
            d_biases[i] = np.clip(d_biases[i], -1.0, 1.0)

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * d_weights[i]
            self.biases[i] -= self.learning_rate * d_biases[i]

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        validation_data: tuple | None = None,
    ):
        """Train the neural network"""
        m = X.shape[0]
        losses = []
        val_losses = []

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            # Mini-batch training
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # Forward pass
                activations, z_values = self.forward(X_batch)

                # Backward pass
                self.backward(X_batch, y_batch, activations, z_values)

            # Compute training loss
            activations, _ = self.forward(X)
            predictions = activations[-1]
            loss = -np.sum(y * np.log(predictions + 1e-10)) / m  # type: ignore
            losses.append(loss)

            # Compute validation loss if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                val_activations, _ = self.forward(X_val)
                val_predictions = val_activations[-1]
                val_loss = -np.sum(y_val * np.log(val_predictions + 1e-10)) / X_val.shape[0]
                val_losses.append(val_loss)
                val_accuracy = np.mean(np.argmax(val_predictions, axis=1) == np.argmax(y_val, axis=1))

                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses, val_losses


def get_path_in_current_file_dir(file_name: str) -> str:
    """Get the current directory of the file"""
    return os.path.join(os.path.dirname(__file__), file_name)


def predict_all(filename: str, model_type: ModelType):
    """Make predictions for the data in filename"""

    if model_type == ModelType.NEURAL_NETWORK:
        # Use the original neural network model
        model_path = get_path_in_current_file_dir("neural_network_model.npz")
        try:
            params = np.load(model_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: Model file {model_path} not found.")
            raise

    elif model_type == ModelType.LINEAR_REGRESSION:
        try:
            # Load the linear regression model
            params_path = get_path_in_current_file_dir("linear_regression_model.npz")
            params = np.load(params_path, allow_pickle=True)
        except FileNotFoundError:
            print(f"Error: Model file {params_path} not found.")
            raise

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Extract embeddings from params
    processor = DataProcessor(params)

    # Load data
    data = pd.read_csv(filename)

    # Preprocess data
    X = processor.preprocess(data)

    # Make predictions using the appropriate model
    if model_type == ModelType.NEURAL_NETWORK:
        model = NeuralNetwork(params=params)

    elif model_type == ModelType.LINEAR_REGRESSION:
        model = LinearRegression()
        model.set_params(**params["model"])

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get predicted class indices
    probabilities = model.predict(X)
    predictions = np.argmax(probabilities, axis=1)

    # Check accuracy if labels are provided
    if "Label" in data.columns:
        labels = data["Label"].to_numpy()
        accuracy = np.mean(predictions == [FoodEnum.from_label(l) for l in labels])
        print(f"Prediction accuracy: {accuracy:.4f}")

    # Convert to food labels
    return [FoodEnum(p).to_label() for p in predictions]


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Neural network for food preference prediction")
    parser.add_argument(
        "filename", nargs="?", default="./cleaned_data_combined.csv", help="CSV file with data to predict"
    )
    parser.add_argument(
        "--model",
        choices=[model.value for model in ModelType],
        default="NeuralNetwork",
        help="Model type to use for prediction",
    )
    args = parser.parse_args()

    if args.filename:
        print(f"Making predictions for {args.filename}...")
        predictions = predict_all(args.filename, ModelType(args.model))
        print("Predictions:", predictions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
