#!/bin/bash

function check_and_fix_pythonpath() {
    REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
    PYTHONPATH_ENTRY="export PYTHONPATH=\$PYTHONPATH:$REPO_ROOT"
    
    # Check if PYTHONPATH is set correctly
    if [[ ":$PYTHONPATH:" != *":$REPO_ROOT:"* ]]; then
        echo "✗ PYTHONPATH needs updating"
        
        # Add to current session
        export PYTHONPATH=$PYTHONPATH:$REPO_ROOT
        
        # Add to shell config files if they exist
        for rcfile in "${HOME}/.bashrc" "${HOME}/.zshrc"; do
            if [[ -f "$rcfile" ]] && ! grep -q "$PYTHONPATH_ENTRY" "$rcfile"; then
                echo "$PYTHONPATH_ENTRY" >> "$rcfile"
                echo "✓ Updated PYTHONPATH in $rcfile"
            fi
        done
        
        echo "✓ PYTHONPATH has been updated"
        return 0
    else
        echo "✓ PYTHONPATH already configured correctly"
        return 0
    fi
}

function check_and_fix_dependencies() {
    local missing_deps=()
    
    # Ensure we're in redaction_env
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate redaction_env
    
    # Get the Python and pip paths for redaction_env
    CONDA_BASE=$(conda info --base)
    ENV_PYTHON="$CONDA_BASE/envs/redaction_env/bin/python"
    ENV_PIP="$CONDA_BASE/envs/redaction_env/bin/pip"
    
    # Check core dependencies using correct python
    for package in "presidio_analyzer" "streamlit" "pymupdf" "spacy"; do
        if ! $ENV_PYTHON -c "import $package" 2>/dev/null; then
            echo "✗ $package missing in redaction_env"
            missing_deps+=("$package")
        else
            echo "✓ $package installed in redaction_env"
        fi
    done
    
    # Install missing dependencies in correct environment
    if [ ${#missing_deps[@]} -ne 0 ]; then
        echo "Installing missing dependencies in redaction_env..."
        $ENV_PIP install ${missing_deps[@]}
    fi
    
    # Check spaCy model specifically
    if ! $ENV_PYTHON -c "import spacy; spacy.load('en_core_web_lg')" 2>/dev/null; then
        echo "✗ spaCy large model missing, installing in redaction_env..."
        $ENV_PIP install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
    else
        echo "✓ spaCy large model installed in redaction_env"
    fi
}

function check_environment() {
    local has_errors=0
    
    echo "=== Checking Python Environment ==="
    # Verify Python version
    if python -c "import sys; assert sys.version_info >= (3, 11)" 2>/dev/null; then
        echo "✓ Python version >= 3.11"
    else
        echo "✗ Python version < 3.11"
        has_errors=1
    fi
    
    echo -e "\n=== Checking Dependencies ==="
    check_and_fix_dependencies
    
    echo -e "\n=== Checking PYTHONPATH ==="
    check_and_fix_pythonpath
    
    echo -e "\n=== Checking Project Structure ==="
    # Check critical directories exist
    for dir in "../app" "../redactor" "../data/redact_input" "../data/redact_output" "../logs"; do
        if [ -d "$dir" ]; then
            echo "✓ Directory exists: $dir"
        else
            echo "✗ Missing directory: $dir"
            has_errors=1
        fi
    done
    
    # Check critical files exist
    for file in "../app/config/config.yaml" "../redactor/config/detection_patterns.yaml"; do
        if [ -f "$file" ]; then
            echo "✓ File exists: $file"
        else
            echo "✗ Missing file: $file"
            has_errors=1
        fi
    done
    
    return $has_errors
}

function full_installation() {
    # Prompt user for environment name
    read -p "Enter the name for the new Conda environment (default: redaction_env): " ENV_NAME
    ENV_NAME=${ENV_NAME:-redaction_env}
    
    echo "Creating a new Conda environment: $ENV_NAME"
    
    # Create and activate environment
    conda create -n "$ENV_NAME" python=3.11 -y || exit 1
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_NAME" || exit 1
    
    # Verify activation
    if [ "$(conda info --envs | grep '*' | awk '{print $1}')" != "$ENV_NAME" ]; then
        echo "Failed to activate Conda environment. Exiting."
        exit 1
    fi
    
    # Install dependencies
    pip install tqdm pymupdf presidio-analyzer presidio-anonymizer streamlit || exit 1
    
    # Install spaCy with model
    pip install spacy || exit 1
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl || exit 1
    
    # Install requirements
    pip install -r ../requirements.txt || exit 1
    
    # Set up project structure
    for dir in "../data/redact_input" "../data/redact_output" "../logs"; do
        mkdir -p "$dir"
    done
    
    # Set up PYTHONPATH
    check_and_fix_pythonpath
    
    # Optional JupyterLab integration
    read -p "Do you want to add this environment to JupyterLab? (y/n): " ADD_TO_JUPYTER
    if [[ "$ADD_TO_JUPYTER" =~ ^[Yy]$ ]]; then
        pip install ipykernel || exit 1
        python -m ipykernel install --user --name="$ENV_NAME" --display-name "Python ($ENV_NAME)" || exit 1
        echo "✓ Environment added to JupyterLab"
    fi
    
    echo "Full installation complete!"
}

# Main script execution
echo "Resume Redaction Environment Setup"
echo "1. Full Installation (new environment)"
echo "2. Check/Fix Existing Environment"
read -p "Select an option (1 or 2): " OPTION

case $OPTION in
   1)
       echo "Starting full installation..."
       full_installation
       echo "Running environment check to verify installation..."
       if check_environment; then
           echo -e "\n✅ Installation successful and environment is properly configured!"
       else
           echo -e "\n⚠️  Installation completed but some checks failed. Please review warnings above."
       fi
       ;;
   2)
       echo "Checking existing environment..."
       # First check/activate correct conda environment
       CURRENT_ENV=$(conda info --envs | grep '*' | awk '{print $1}')
       if [ "$CURRENT_ENV" != "redaction_env" ]; then
           echo "✗ Wrong environment active: $CURRENT_ENV"
           echo "Attempting to activate redaction_env..."
           source "$(conda info --base)/etc/profile.d/conda.sh"
           if conda activate redaction_env; then
               echo "✓ Successfully activated redaction_env"
           else
               echo "✗ Could not activate redaction_env."
               read -p "Would you like to create the redaction_env environment? (y/n): " CREATE_ENV
               if [[ "$CREATE_ENV" =~ ^[Yy]$ ]]; then
                   echo "Running full installation..."
                   full_installation
                   exit 0
               else
                   echo "Environment check cancelled. Please create redaction_env first."
                   exit 1
               fi
           fi
       else
           echo "✓ Correct environment (redaction_env) is active"
       fi

       # Now check the environment configuration
       if check_environment; then
           echo -e "\n✅ All checks passed! Environment is properly configured."
       else
           echo -e "\n⚠️  Some checks failed."
           read -p "Would you like to attempt fixes? (y/n): " FIX
           if [[ "$FIX" =~ ^[Yy]$ ]]; then
               echo "Attempting to fix environment..."
               check_and_fix_dependencies
               check_and_fix_pythonpath
               mkdir -p "../data/redact_input" "../data/redact_output" "../logs"
               echo -e "\nRunning final verification..."
               if check_environment; then
                   echo -e "\n✅ All fixes successful! Environment is now properly configured."
               else
                   echo -e "\n⚠️  Some issues remain. You may need to run a full installation (Option 1)."
               fi
           fi
       fi
       ;;
   *)
       echo "Invalid option"
       exit 1
       ;;
esac