# Resume Redaction Pipeline and Web App

This repository provides two tools for detecting and redacting sensitive information (PII) from resumes:

1. **Redaction Pipeline Script**:
   - A script for batch processing PDF resumes using [Microsoft Presidio](https://github.com/microsoft/presidio).
   - Supports random sampling, PII detection, and PDF redaction with summary statistics.

2. **Streamlit Web App**:
   - An interactive web-based interface for selecting input/output directories, loading models, and processing resumes.
   - Integrates the **JobBERT** model for improved PII detection.

---

## Features

- **Presidio-based PII detection**:
  - Detects names, emails, phone numbers, and more.
- **True PDF redaction**:
  - Redacts detected PII with black rectangles in the output PDFs.
- **Random sampling**:
  - Processes a random subset of resumes during testing.
- **Streamlit Web App**:
  - Provides an intuitive interface for model loading, directory selection, and PDF redaction.
- **Summary statistics**:
  - Displays the amount of data processed and redacted.

---

## Project Structure

```
project/
├── app/                       
│   ├── redactor_gui.py       # Main Streamlit interface
│   ├── config/               
│   │   └── config.yaml       # App-specific config (paths, UI settings)
│   └── utils/
│       ├── __init__.py
│       ├── logger.py         
│       └── config_loader.py  
├── redactor/
│   ├── __init__.py
│   ├── redactor_logic.py     
│   ├── file_processor.py     
│   ├── config/               # Moved from app/config/patterns
│   │   ├── detection_patterns.yaml
│   │   ├── confidential_terms.yaml
│   │   └── custom_word_filters.yaml
│   └── detectors/            
│       ├── __init__.py
│       ├── pattern_matcher.py
│       └── entity_detector.py
├── data/                      # Input/output folders
│   ├── redact_input/          # Input PDFs
│   └── redact_output/         # Redacted PDFs
├── models/                    # Model artifacts
│   └── spacy_model/           # spaCy model directory
├── logs/                      # Log files directory
├── scripts/                   # Environment setup and utility scripts
│   └── setup_environment.sh   # Create Conda environment
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── .gitignore                # Git ignore rules
```

---

## Quick Start

### Prerequisites

1. **Install Conda**:
   Ensure you have Conda installed. You can download and install Miniconda or Anaconda from [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

2. **Clone the Repository**:
   Clone this repository and navigate to the project root:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

---

### Setting Up the Environment

3. **Create a Conda Environment**:
   Use the provided setup script to create a Conda environment and install all dependencies:
   ```bash
   cd scripts/
   ./setup_environment.sh
   ```
   - You will be prompted to name the environment (default: `redaction_env`).
   - Optionally, you can add the environment to JupyterLab during setup.

---

### Running the Application

4. **Start the Streamlit Web App**:
   Run the Streamlit app to begin redacting resumes:
   ```bash
   streamlit run app/app.py
   ```

5. **Open the App**:
   Open the app in your web browser. By default, it will run at:
   ```
   http://localhost:8501
   ```

6. **Use the Interface**:
   - Select input and output directories.
   - Load the JobBERT model.
   - Start the redaction process.

---

### Notes
- Ensure all directories (e.g., `data/redact_input`, `data/redact_output`) exist and are accessible.
- Test your setup with sample PDFs placed in the `data/redact_input` directory.

---

### Option 1: Run the Pipeline Script

1. Execute the redactor script:
   ```bash
   python redactor_gui.py
   ```

2. The script processes PDFs from `data/redact_input` and outputs redacted files to `data/redact_output`.

3. Check outputs:
   - Original files are preserved.
   - Redacted files are saved with `_redacted` appended to their filenames.
   - Summary statistics are printed in the console.

---

### Option 2: Run the Streamlit Web App

1. Start the Streamlit app:
   ```bash
   streamlit run app/app.py
   ```

2. Open the app in your browser. By default, it runs at:
   ```
   http://localhost:8501
   ```

3. Use the interface to:
   - **Select input and output directories**.
   - **Load the JobBERT model**.
   - **Start processing files** (PDF redaction logic is under active development).

---

## Configuring Directories

### Default Paths
- **Model Directory**: `models/jobbert_model`
- **Input Directory**: `data/redact_input`
- **Output Directory**: `data/redact_output`

### Customizing Paths
Paths can be updated:
1. **In the app interface** (Streamlit).
2. **Directly in the script/config file** (`config.yaml`).

---

## Data Source

- The sample resume data used for testing can be obtained from the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
- Place the resumes in the `/data/resumes/` directory for the pipeline or `/data/redact_input/` for the app.

---

## Known Limitations

- The Streamlit app currently provides directory selection and model loading functionality. Full PDF redaction capabilities are under development.
- Ensure directories exist and are accessible before running the app or script.

---

## License

This project is licensed under the Apache 2.0 License.

---

## Acknowledgments

- [Microsoft Presidio](https://github.com/microsoft/presidio)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [Streamlit](https://streamlit.io/)

> **Disclaimer**  
> This code is provided by the author in a personal capacity. The views and opinions expressed are those of the author alone and do not necessarily reflect the official policy or position of any agency or employer. No official endorsement by the author’s employer is implied.
