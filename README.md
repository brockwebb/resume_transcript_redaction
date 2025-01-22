# Resume Redaction System

A robust system for detecting and redacting sensitive information from resumes and professional documents, combining multiple detection approaches with configurable entity routing and validation. Available both as a Streamlit web application and a command-line evaluation framework.

## Core Components

### 1. Web Application
- **Streamlit Interface**: User-friendly web interface for document redaction
- **Interactive Controls**: Directory selection and processing options
- **Real-time Feedback**: Processing status and results display

### 2. Detection System
- **Ensemble Coordination**: Combines multiple detectors with configurable routing and confidence thresholds
- **Presidio Integration**: Custom recognizers for structured data (phones, SSNs, etc.)
- **spaCy Detection**: Context-aware named entity recognition
- **Pattern Matching**: Configurable regex patterns for specialized detection

### 3. Evaluation Framework
- **Ground Truth Comparison**: Compare detections against annotated test cases
- **Performance Metrics**: Track detection rates by entity type and detector
- **Error Analysis**: Identify missed entities and mapping conflicts
- **Configuration Tuning**: Tools for optimizing detection parameters

## Project Structure
```
project/
├── app/                       
│   ├── redactor_gui.py        # Main interface
│   ├── config/               
│   │   └── config.yaml        # App configuration
│   └── utils/
│       ├── logger.py          # Logging utilities
│       └── config_loader.py   # Configuration management
├── redactor/
│   ├── redactor_logic.py      # Core redaction logic
│   ├── file_processor.py      # PDF processing
│   ├── config/               
│   │   ├── core_patterns.yaml     # Base detection patterns
│   │   ├── entity_routing.yaml    # Detector routing rules
│   │   └── custom_word_filters.yaml
│   └── detectors/            
│       ├── ensemble_coordinator.py # Detector coordination
│       ├── presidio_detector.py    # Presidio integration
│       ├── spacy_detector.py      # spaCy integration
│       └── custom_recognizers.py   # Custom entity recognition
├── evaluation/
│   ├── evaluate.py            # Evaluation framework
│   └── test_suite/           
│       ├── originals/         # Original test PDFs
│       │   ├── alexis_rivera.pdf
│       │   └── ...
│       ├── annotations/       # Ground truth annotations
│       │   ├── alexis_rivera.yaml
│       │   └── ...
│       ├── results/          # Test run results
│       │   ├── run_20240121/ # Date-stamped test runs
│       │   │   ├── metrics.json
│       │   │   └── test_results.yaml
│       │   └── ...
│       └── test_manifest.yaml # Test suite metadata
├── data/                     
│   ├── redact_input/         # Documents to be redacted
│   └── redact_output/        # Redacted output documents
├── scripts/
│   └── setup_environment.sh  # Environment setup script
└── requirements.txt
```
## Setup and Usage

### Environment Setup
1. **Use the setup script** (Recommended):
   ```bash
   cd scripts
   ./setup_environment.sh
   ```
   This will:
   - Create the Conda environment (redaction_env)
   - Install all required dependencies from requirements.txt
   - Download necessary spaCy models
   - Configure project paths

2. **Manual setup** (if script doesn't work):
   ```bash
   conda create -n redaction_env python=3.9
   conda activate redaction_env
   pip install -r requirements.txt
   python -m spacy download en_core_web_lg
   ```

### Running the Application

1. **Start the Streamlit Web App**:
   ```bash
   conda activate redaction_env
   streamlit run app/redactor_gui.py
   ```

2. **Open the App**:
   The app will automatically open in your default browser, or visit:
   ```
   http://localhost:8501
   ```

3. **Use the Interface**:
   - Select input and output directories
   - Upload documents for redaction
   - View processing results

### Running the Evaluation Framework

For tuning and testing the detection system:
```bash
conda activate redaction_env
python -m evaluation.evaluate detect --test [test_id]
```

## Configuration

### Entity Routing
Configure detector assignments in `entity_routing.yaml`:
```yaml
routing:
  presidio_primary:
    entities:
      - EMAIL_ADDRESS
      - PHONE_NUMBER
      # ...
  spacy_primary:
    entities:
      - PERSON
      - LOCATION
      # ...
```

### Detection Patterns
Define custom patterns in `core_patterns.yaml`:
```yaml
phone_patterns:
  - name: "us_standard"
    regex: "\\(?[0-9]{3}\\)?[-. ]?[0-9]{3}[-. ]?[0-9]{4}"
    score: 0.75
```

## Current Development

The project is currently focused on:
1. Optimizing entity detection and mapping
2. Tuning confidence thresholds
3. Expanding pattern coverage
4. Improving evaluation metrics

## License

This project is licensed under the MIT License.

## Acknowledgments
- Microsoft Presidio
- spaCy
- PyMuPDF
- Streamlit

## Disclaimer

This code is provided by the author in a personal capacity. The views and opinions expressed are those of the author alone and do not necessarily reflect the official policy or position of any agency or employer. No official endorsement by the author's employer is implied.

The code in this repository is for educational and research purposes only. While care has been taken to ensure accuracy and reliability, users should use this software at their own risk. The author assumes no responsibility or liability for any errors or omissions in the content or consequences of using this software. This disclaimer is not intended to limit the liability of the author in contravention of applicable laws.

Users are responsible for:
- Validating all redactions before using redacted documents
- Ensuring compliance with relevant privacy laws and regulations
- Maintaining appropriate security measures when handling sensitive data
- Verifying the accuracy and completeness of any redactions
