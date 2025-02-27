#!/bin/bash

# Define variables
VENV_DIR=".venv"
KERNEL_NAME="myenv"
DISPLAY_NAME="Python (myenv)"
REQUIREMENTS_FILE="requirements.txt"

# Check if requirements.txt exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "Error: $REQUIREMENTS_FILE not found in the current directory!"
    echo "Please create it with 'pip freeze > $REQUIREMENTS_FILE' first."
    exit 1
fi

# Check if virtual environment already exists
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment already exists in $VENV_DIR. Skipping creation..."
else
    # Create virtual environment if it doesn't exist
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv $VENV_DIR
fi

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_DIR/bin/activate

# Upgrade pip (optional but recommended)
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt
echo "Installing dependencies from $REQUIREMENTS_FILE..."
pip install -r $REQUIREMENTS_FILE

# Register the kernel for Jupyter (only if not already registered)
if jupyter kernelspec list | grep -q "$KERNEL_NAME"; then
    echo "Kernel '$KERNEL_NAME' already registered. Skipping registration..."
else
    echo "Registering kernel '$KERNEL_NAME' as '$DISPLAY_NAME'..."
    python -m ipykernel install --user --name="$KERNEL_NAME" --display-name="$DISPLAY_NAME"
fi

# Notify user
echo "Setup complete! To start Jupyter, activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run:"
echo "  jupyter notebook"
echo "To deactivate later, simply run: deactivate"
