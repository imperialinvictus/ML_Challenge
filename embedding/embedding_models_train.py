import argparse
import os

import numpy as np
import pandas as pd
from embedding_models import DataProcessor, FoodEnum, NeuralNetwork, predict_all
from sklearn.model_selection import train_test_split


def one_hot_encode(labels: np.ndarray) -> np.ndarray:
    """Convert text labels to one-hot encoded vectors"""
    result = np.zeros((len(labels), len(FoodEnum)))

    for i, label in enumerate(labels):
        result[i, FoodEnum.from_label(label)] = 1

    return result


def train_model(data_path: str | None = None, epochs: int = 300, batch_size: int = 32):
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
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Preprocess data
    processor = DataProcessor()
    X_train_processed = processor.preprocess(X_train, y_train, train_mode=True)
    X_val_processed = processor.preprocess(X_val, y_val, train_mode=False)

    print(f"Input feature dimension: {X_train_processed.shape[1]}")

    # Define and train neural network
    input_size = X_train_processed.shape[1]
    hidden_sizes = [64, 32]
    output_size = 3

    nn = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate=0.001)
    losses, val_losses = nn.train(
        X_train_processed, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val_processed, y_val)
    )

    # Evaluate model
    train_preds = nn.predict(X_train_processed)
    train_accuracy = np.mean(np.argmax(train_preds, axis=1) == np.argmax(y_train, axis=1))

    val_preds = nn.predict(X_val_processed)
    val_accuracy = np.mean(np.argmax(val_preds, axis=1) == np.argmax(y_val, axis=1))

    print(f"Final Training Accuracy: {train_accuracy:.4f}")
    print(f"Final Validation Accuracy: {val_accuracy:.4f}")

    # Save model parameters
    save_params = {}

    # Add weights and biases
    for i, (weights, biases) in enumerate(zip(nn.weights, nn.biases)):
        save_params[f"weights_{i}"] = weights
        save_params[f"biases_{i}"] = biases

    # Add other preprocessing parameters
    for key, value in processor.preprocess_params.items():
        save_params[key] = value

    model_path = os.path.join(os.path.dirname(__file__), "model_params.npz")
    np.savez(model_path, **save_params)
    print(f"Model saved to {model_path}")


def main():
    """Main function to handle command line arguments"""
    parser = argparse.ArgumentParser(description="Neural network for food preference prediction")
    parser.add_argument("--train", action="store_true", help="Train a new model")
    parser.add_argument("--epochs", type=int, default=300, help="Number of epochs for training (default: 300)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training (default: 32)")
    parser.add_argument("filename", nargs="?", help="CSV file with data to predict")

    args = parser.parse_args()

    if args.train or not args.filename:
        print("Training a new model...")
        train_model(args.filename, args.epochs, args.batch_size)
    elif args.filename:
        print(f"Making predictions for {args.filename}...")
        predictions = predict_all(args.filename)
        print("Predictions:", predictions)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
