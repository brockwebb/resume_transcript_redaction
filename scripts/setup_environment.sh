#!/bin/bash

# Prompt user for environment name
read -p "Enter the name for the new Conda environment (default: redaction_env): " ENV_NAME
ENV_NAME=${ENV_NAME:-redaction_env}  # Default to 'redaction_env' if no input

echo "Creating a new Conda environment: $ENV_NAME"

# Step 1: Create a new Conda environment with Python 3.11
conda create -n "$ENV_NAME" python=3.11 -y
if [ $? -ne 0 ]; then
    echo "Failed to create Conda environment. Exiting."
    exit 1
fi

# Step 2: Activate the new environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

# Verify environment activation
if [ "$(conda info --envs | grep '*' | awk '{print $1}')" != "$ENV_NAME" ]; then
    echo "Failed to activate Conda environment. Exiting."
    exit 1
fi
echo "Environment '$ENV_NAME' activated successfully."

# Step 3: PyTorch Installation
echo "PyTorch Installation:"
read -p "For Mac (CPU-only), press 'm'. For default setup, press any other key: " PREFERENCE

if [[ "$PREFERENCE" == "m" ]]; then
    echo "Installing PyTorch for Mac (CPU-only)..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
    echo "Installing default PyTorch (CPU/GPU)..."
    conda install pytorch torchvision torchaudio cpuonly -c pytorch -y
fi

if [ $? -ne 0 ]; then
    echo "Failed to install PyTorch. Exiting."
    exit 1
fi

# Step 4: Install additional dependencies
echo "Installing additional dependencies..."
pip install tqdm pymupdf presidio-analyzer presidio-anonymizer streamlit transformers

if [ $? -ne 0 ]; then
    echo "Failed to install additional dependencies. Exiting."
    exit 1
fi

# Step 5: Install spaCy language model
echo "Downloading spaCy language model..."
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
if [ $? -ne 0 ]; then
    echo "Failed to download spaCy language model. Exiting."
    exit 1
fi

# Step 6: Install additional dependencies and finalize environment setup
echo "Installing additional dependencies from requirements.txt..."
pip install -r ../requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install dependencies from requirements.txt. Exiting."
    exit 1
fi


# Step 7: Optional JupyterLab integration
read -p "Do you want to add this environment to JupyterLab? (y/n): " ADD_TO_JUPYTER
if [[ "$ADD_TO_JUPYTER" =~ ^[Yy]$ ]]; then
    echo "Adding environment to JupyterLab..."
    pip install ipykernel
    python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)"
    if [ $? -ne 0 ]; then
        echo "Failed to add environment to JupyterLab. Exiting."
        exit 1
    fi
    echo "Environment added to JupyterLab."
else
    echo "Skipping JupyterLab integration."
fi

echo "Setup complete! To activate the environment, run: conda activate $ENV_NAME"
