import streamlit as st
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import os
import fitz  # PyMuPDF for PDF processing
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from typing import Dict, Any

# Define the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dynamically resolve paths
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "jobbert_model"
INPUT_DIRECTORY = PROJECT_ROOT / "data" / "redact_input"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "redact_output"

# Set up Streamlit page configuration
st.set_page_config(
    page_title="Resume Redaction App",
    page_icon="📄",
    layout="wide",
)

# Application Title
st.title("Resume Redaction Web App")
st.write("Use this tool to redact sensitive information from resumes.")

# Initialize session state
if "model" not in st.session_state:
    st.session_state.model = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "analyzer" not in st.session_state:
    st.session_state.analyzer = AnalyzerEngine()
if "anonymizer" not in st.session_state:
    st.session_state.anonymizer = AnonymizerEngine()

# Set up custom recognizers
def setup_custom_recognizers():
    """Setup custom recognizers with more precise patterns."""
    patterns = [
        Pattern(
            name="date_numeric",
            regex=r"(\d{1,2}/\d{4}|\d{4})–?(Present|\d{1,2}/\d{4})",
            score=0.85
        ),
        Pattern(
            name="date_alphanumeric",
            regex=r"(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
            score=0.85
        ),
        # Add recognizer for phone numbers
        Pattern(
            name="phone_number",
            regex=r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            score=0.95
        ),
        # Add recognizer for email addresses
        Pattern(
            name="email_address",
            regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            score=0.95
        )
    ]
    
    recognizers = []
    for pattern in patterns:
        recognizer = PatternRecognizer(
            supported_entity=pattern.name.upper(),
            patterns=[pattern]
        )
        recognizers.append(recognizer)
    
    return recognizers

def should_redact_text(text: str, analyzer: AnalyzerEngine) -> bool:
    """
    Determine if text should be redacted based on PII analysis.
    Returns True if text contains PII that should be redacted.
    """
    if not text.strip():
        return False
        
    # List of entities we want to redact
    pii_entities = {
        "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", 
        "DATE_NUMERIC", "DATE_ALPHANUMERIC"
    }
    
    # Skip common resume section headers and job titles
    skip_phrases = {
        "education", "experience", "skills", "qualifications", 
        "awards", "publications", "volunteer", "training",
        "professional experience", "job-related training",
        "key qualifications", "other expertise"
    }
    
    # Check if text is a section header we want to preserve
    if text.strip().lower() in skip_phrases:
        return False
    
    # Analyze text for PII
    results = analyzer.analyze(
        text=text,
        language="en",
        score_threshold=0.65  # Adjust confidence threshold
    )
    
    # Check if any detected entity is in our list of entities to redact
    return any(result.entity_type in pii_entities for result in results)


@st.cache_resource
def load_jobbert_model(model_directory: Path):
    """Load the JobBERT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = AutoModel.from_pretrained(model_directory)
    return tokenizer, model

# Function for redacting a PDF

def redact_pdf(input_path: str, output_path: str, analyzer: AnalyzerEngine, anonymizer: AnonymizerEngine) -> Dict[str, Any]:
    """Redact PII from a PDF document with improved accuracy."""
    try:
        # Add custom recognizers to the analyzer
        for recognizer in setup_custom_recognizers():
            analyzer.registry.add_recognizer(recognizer)
            
        # Open source PDF
        doc = fitz.open(input_path)
        
        # Create new PDF
        output_doc = fitz.open()
        
        total_words = 0
        redacted_words = 0
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Create new page with same dimensions
            new_page = output_doc.new_page(width=page.rect.width, height=page.rect.height)
            
            # Get text blocks with position information
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text = span["text"]
                            if text.strip():
                                total_words += 1
                                
                                # Check if text should be redacted
                                if should_redact_text(text, analyzer):
                                    redacted_words += 1
                                    # Redact the text area
                                    rect = fitz.Rect(span["bbox"])
                                    new_page.draw_rect(
                                        rect, 
                                        color=(0, 0, 0), 
                                        fill=(0, 0, 0)
                                    )
                                else:
                                    # Preserve non-PII text
                                    try:
                                        new_page.insert_text(
                                            point=(span["bbox"][0], span["bbox"][3]),
                                            text=text,
                                            fontname="helv",
                                            fontsize=span.get("size", 11),
                                            color=(0, 0, 0)
                                        )
                                    except Exception:
                                        # Fallback if font issues occur
                                        new_page.insert_text(
                                            point=(span["bbox"][0], span["bbox"][3]),
                                            text=text,
                                            fontname="helv",
                                            fontsize=11,
                                            color=(0, 0, 0)
                                        )
            
            # Copy any images or other non-text content
            new_page.show_pdf_page(
                new_page.rect,
                doc,
                page_num,
                clip=new_page.rect
            )
        
        # Save the redacted document
        output_doc.save(
            output_path,
            garbage=4,
            deflate=True,
            clean=True
        )
        output_doc.close()
        doc.close()
        
        return {
            "total_words": total_words,
            "redacted_words": redacted_words
        }
        
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return {"total_words": 0, "redacted_words": 0}

# [Rest of the Streamlit app code remains the same...]

# Sidebar for configuration
st.sidebar.header("Configuration")
st.sidebar.write("JobBERT model directory and settings.")

model_directory = st.sidebar.text_input("JobBERT Model Directory", value=str(MODEL_DIRECTORY))

if st.sidebar.button("Load JobBERT Model"):
    try:
        st.session_state.tokenizer, st.session_state.model = load_jobbert_model(model_directory)
        st.sidebar.success("JobBERT model loaded successfully.")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# Directory Selection Section
st.header("Step 1: Select Directories")

# Input Directory
input_dir = st.text_input("Input Directory", value=str(INPUT_DIRECTORY))
if st.button("Verify Input Directory"):
    if os.path.exists(input_dir) and os.path.isdir(input_dir):
        st.success(f"Valid input directory: {input_dir}")
        files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
        st.write(f"Found {len(files)} PDF files in the directory.")
    else:
        st.error("Invalid input directory. Please check the path.")

# Output Directory
output_dir = st.text_input("Output Directory", value=str(OUTPUT_DIRECTORY))
if st.button("Verify Output Directory"):
    if os.path.exists(output_dir) and os.path.isdir(output_dir):
        st.success(f"Valid output directory: {output_dir}")
    else:
        st.error("Invalid output directory. Please check the path.")

# Processing Section
st.header("Step 2: Process Files")
if st.button("Start Redaction"):
    if not input_dir or not output_dir:
        st.error("Please provide valid input and output directories.")
    elif st.session_state.model is None or st.session_state.tokenizer is None:
        st.error("Please load the JobBERT model before starting.")
    else:
        pdf_files = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]

        if not pdf_files:
            st.warning("No PDF files found in the input directory.")
        else:
            # Progress bar
            st.write("Processing files...")
            progress_bar = st.progress(0)
            
            stats = {
                "total_files": len(pdf_files),
                "processed_files": 0,
                "total_words": 0,
                "total_redacted": 0
            }

            for idx, pdf_file in enumerate(pdf_files):
                input_path = os.path.join(input_dir, pdf_file)
                output_file = f"{os.path.splitext(pdf_file)[0]}_redacted.pdf"
                output_path = os.path.join(output_dir, output_file)

                # Process the PDF
                try:
                    results = redact_pdf(input_path, output_path, 
                                      st.session_state.analyzer, 
                                      st.session_state.anonymizer)
                    
                    stats["processed_files"] += 1
                    stats["total_words"] += results["total_words"]
                    stats["total_redacted"] += results["redacted_words"]
                    
                except Exception as e:
                    st.error(f"Error processing {pdf_file}: {e}")
                    continue

                # Update progress bar
                progress_bar.progress((idx + 1) / len(pdf_files))

            # Display summary
            st.success("Redaction complete!")
            st.write(f"Processed {stats['processed_files']} of {stats['total_files']} files")
            st.write(f"Total words processed: {stats['total_words']}")
            st.write(f"Total words redacted: {stats['total_redacted']}")
            if stats['total_words'] > 0:
                redaction_rate = (stats['total_redacted'] / stats['total_words'] * 100)
                st.write(f"Overall redaction rate: {redaction_rate:.2f}%")