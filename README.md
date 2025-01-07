# Resume Redaction Pipeline using Presidio

This repository showcases a prototype application that detects and redacts sensitive information (PII) from PDF resumes using [Microsoft Presidio](https://github.com/microsoft/presidio). It demonstrates the following features:

- **Random sampling** of PDF resumes from a given dataset
- **Presidio-based** detection of personal data such as names, emails, phone numbers, etc.
- **True PDF redaction** via black rectangles over detected text
- **Summary statistics** detailing the amount of data processed and redacted

## Data Source

- The sample resume data is obtained from the [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset).
- You will need to put them all in the `/data/resumes` diretory 

> **Disclaimer**  
> This code is provided by the author in a personal capacity. The views and opinions expressed are those of the author alone and do not necessarily reflect the official policy or position of any agency or employer. No official endorsement by the author’s employer is implied.

## Quick Start

1. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the redactor script**  
   ```bash
   python redactor_app.py
   ```
   This will take a random sample of PDFs from `/data/resumes`, copy them into `/data/redaction_test`, and create redacted PDFs alongside the originals.

3. **Check outputs**  
   - Original files: `*_input.pdf`  
   - Redacted files: `*_redacted.pdf`  
   - Summary statistics printed to the console.

## Contributing

Feel free to open issues or pull requests if you have suggestions for improvements. 
```
