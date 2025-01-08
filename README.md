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
├── app/                       # Streamlit app and GUI code
│   └── app.py                 # Main Streamlit app
├── models/                    # Model artifacts
│   └── jobbert_model/         # Pre-trained JobBERT model directory
├── data/                      # Input/output folders
│   ├── redact_input/          # Input PDFs
│   └── redact_output/         # Redacted PDFs
├── scripts/                   # Environment setup and utility scripts
│   └── setup_environment.sh   # Script to create a new Conda environment
├── requirements.txt           # Python dependencies (pip installable)
├── README.md                  # Project description and instructions
└── .gitignore                 # Git ignore rules
```

---

## Quick Start

### Prerequisites

1. Install Python 3.7+ and `pip`.
2. Clone this repository and navigate to the project root:
   ```bash
   git clone https://github.com/your-repo-name.git
   cd your-repo-name
   ```

3. Install required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
## Setting Up the Environment

4. Set up a new Conda environment for this project, use the provided script:

```bash
cd scripts/
./setup_environment.sh
```   

5. Place your pre-trained **JobBERT** model in the `models/jobbert_model` directory. 

### Option 1: Run the Pipeline Script

1. Execute the redactor script:
   ```bash
   python redactor_app.py
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
