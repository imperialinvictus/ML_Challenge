import argparse
import pickle

import numpy as np
import pandas as pd
from embedding_models import (
    DataProcessor,
    FoodEnum,
    LogisticRegression,
    ModelType,
    NeuralNetwork,
    get_path_in_current_file_dir,
    predict_all,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """Convert text labels to one-hot encoded vectors"""
    result = np.zeros((len(labels), len(FoodEnum)))

    for i, label in enumerate(labels):
        result[i, FoodEnum.from_label(label)] = 1

    return result


def train_model(model_type: ModelType, data_path: str | None = None, epochs: int = 300, batch_size: int = 32):
    """Train a neural network model"""
    # Load dataset
    if data_path is None:
        data_path = "./cleaned_data_combined.csv"
    df = pd.read_csv(data_path)

    # Split features and labels
    X = df.drop("Label", axis=1)
    y = df["Label"]

    # One-hot encode labels
    y_encoded = one_hot_encode(y.to_numpy())

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y)

    # Preprocess data
    processor = DataProcessor()
    X_train_processed = processor.preprocess(X_train, y_train, train_mode=True)
    X_val_processed = processor.preprocess(X_val, y_val, train_mode=False)

    print(f"Input feature dimension: {X_train_processed.shape[1]}")

    # Define and train neural network
    input_size = X_train_processed.shape[1]
    hidden_sizes = [64, 64, 64]
    output_size = 3

    # Save model parameters
    save_params = {}

    if model_type == ModelType.NEURAL_NETWORK:
        model = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate=0.001)
        losses, val_losses = model.train(
            X_train_processed, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_processed, y_val)
        )

        # Add weights and biases
        for i, (weights, biases) in enumerate(zip(model.weights, model.biases)):
            save_params[f"weights_{i}"] = weights
            save_params[f"biases_{i}"] = biases

        model_path = get_path_in_current_file_dir("neural_network_model.npz")

    elif model_type == ModelType.LINEAR_REGRESSION:
        model = LinearRegression()
        model.fit(X_train_processed, y_train)

        with open(get_path_in_current_file_dir("linear_regression_model.pkl"), "wb") as f:
            pickle.dump(model, f)

        model_path = get_path_in_current_file_dir("linear_regression_model.npz")

    elif model_type == ModelType.LOGISTIC_REGRESSION:
        model = LogisticRegression(solver="liblinear", max_iter=1000)
        model.fit_proba(X_train_processed, y_train)

        with open(get_path_in_current_file_dir("logistic_regression_model.pkl"), "wb") as f:
            pickle.dump(model, f)

        model_path = get_path_in_current_file_dir("logistic_regression_model.npz")

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Evaluate model
    train_preds = model.predict(X_train_processed)
    train_accuracy = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1))

    val_preds = model.predict(X_val_processed)
    val_accuracy = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Add other preprocessing parameters
    for key, value in processor.preprocess_params.items():
        save_params[key] = value

    np.savez(model_path, **save_params)
    print(f"Model saved to {model_path}")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Neural network for food preference prediction")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument(
        "--model", choices=[model.value for model in ModelType], default="NeuralNetwork", help="Model type to use"
    )
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("--max_iter", type=int, default=500, help="Max iterations for regression model (default: 500)")
    parser.add_argument("filename", nargs="?", help="CSV file with data to predict")

    args = parser.parse_args()

    if not args.evaluate:
        print("Training model...")
        train_model(ModelType(args.model), args.filename, epochs=args.epochs, batch_size=args.batch_size)
    elif args.filename:
        print(f"Making predictions for {args.filename}...")
        predictions = predict_all(args.filename, ModelType(args.model))
        print("Predictions:", predictions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
