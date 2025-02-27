#!/bin/bash

# Define variables
VENV_DIR=".venv"
KERNEL_NAME="myenv"
DISPLAY_NAME="Python (myenv)"

# Create virtual environment
echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv $VENV_DIR

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip (optional but recommended)
echo "Upgrading pip..."
pip install --upgrade pip

# Install Jupyter and ipykernel
echo "Installing Jupyter and ipykernel..."
pip install jupyter ipykernel

# Register the kernel for Jupyter
echo "Registering kernel '$KERNEL_NAME' as '$DISPLAY_NAME'..."
python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$DISPLAY_NAME"

# Notify user
echo "Setup complete! To start Jupyter, activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run:"
echo "  jupyter notebook"
echo "To deactivate later, simply run: deactivate"
