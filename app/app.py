import streamlit as st
from pathlib import Path
import fitz
import torch
from transformers import AutoModel, AutoTokenizer
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from presidio_anonymizer import AnonymizerEngine
from typing import Dict, Any
import os
import yaml  # Make sure this import is here
from collections import defaultdict

# Define the project root dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load configuration from config.yaml
def load_config():
    config_path = PROJECT_ROOT / "config.yaml"
    if not config_path.exists():
        return {
            "test_mode": False,
            "confidence_threshold": 0.75,
            "pii_types": [
                "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION",
                "DATE", "ORGANIZATION", "US_SSN", "US_PASSPORT"
            ]
        }
    
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# Load config immediately after the function definition
config = load_config()

class ModelManager:
    def __init__(self, model_directory: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.analyzer = AnalyzerEngine()
        self.word_cache = defaultdict(dict)
        self.custom_recognizers = self.setup_custom_recognizers()
        self.load_models(model_directory)

    def load_models(self, model_directory: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModel.from_pretrained(model_directory).to(self.device)
        for recognizer in self.custom_recognizers:
            self.analyzer.registry.add_recognizer(recognizer)

    def setup_custom_recognizers(self):
        date_patterns = [
            Pattern(
                name="date_numeric",
                regex=r"(\d{1,2}/\d{4}|\d{4})–?(Present|\d{1,2}/\d{4})",
                score=0.85
            ),
            Pattern(
                name="date_alphanumeric",
                regex=r"(?i)(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}",
                score=0.85
            )
        ]
        recognizers = [
            PatternRecognizer(
                supported_entity="DATE",
                patterns=date_patterns,
                context=["date", "period", "timeline"]
            )
        ]
        return recognizers

    def detect_pii_in_word(self, word_text: str, pii_types: list) -> bool:
        if not word_text.strip():
            return False
        
        results = self.analyzer.analyze(
            text=word_text,
            language="en"
        )
        
        return any(
            result.entity_type in pii_types
            and result.score >= 0.65
            for result in results
        )

    def validate_with_jobbert(self, texts: list, confidence_threshold: float) -> list:
        if not texts:
            return []

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        confidence_scores = [torch.norm(emb).item() for emb in embeddings]
        return [score > confidence_threshold for score in confidence_scores]

class PDFProcessor:
    def __init__(self, model_manager: ModelManager, pii_types: list, confidence_threshold: float, test_mode: bool = False):
        self.model_manager = model_manager
        self.pii_types = pii_types
        self.confidence_threshold = confidence_threshold
        self.test_mode = test_mode
        # Define colors for different modes
        self.redact_color = (0.8, 0.8, 0.8) if test_mode else (0, 0, 0)  # Grey in test mode, Black in production
        self.redact_opacity = 0.1 if test_mode else 1.0  # Semi-transparent in test mode, Opaque in production

    def process_page(self, page: fitz.Page) -> Dict[str, Any]:
        wordlist = page.get_text("words")
        redact_areas = []
        batch_texts = []
        batch_coords = []

        for word in wordlist:
            text = word[4]
            if self.model_manager.detect_pii_in_word(text, self.pii_types):
                batch_texts.append(text)
                batch_coords.append((word[0], word[1], word[2], word[3]))

        if batch_texts:
            validation_results = self.model_manager.validate_with_jobbert(
                batch_texts, self.confidence_threshold
            )
            for (coords, should_redact) in zip(batch_coords, validation_results):
                if should_redact:
                    redact_areas.append(fitz.Rect(*coords))

        return {"redact_areas": redact_areas, "word_count": len(wordlist)}

    def redact_pdf(self, input_path: str, output_path: str) -> Dict[str, int]:
        try:
            doc = fitz.open(input_path)
            total_words = 0
            redacted_words = 0

            for page_index in range(len(doc)):
                page = doc[page_index]
                result = self.process_page(page)
                total_words += result["word_count"]
                redacted_words += len(result["redact_areas"])

               # Apply redactions with test mode settings
                for rect in result["redact_areas"]:
                    annot = page.add_redact_annot(rect)
                    # Set the fill color and opacity based on test mode
                    if self.test_mode:
                        annot.set_colors(fill=self.redact_color, stroke=self.redact_color)
                        annot.update(opacity=self.redact_opacity)  # allow visual check of data 
                    else:
                        annot.set_colors(fill=self.redact_color)
                    
                page.apply_redactions()

            doc.save(output_path, garbage=4, deflate=True)
            doc.close()
            return {"total_words": total_words, "redacted_words": redacted_words}

        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            return {"total_words": 0, "redacted_words": 0}

# Streamlit UI
st.set_page_config(
    page_title="Resume Redaction App",
    page_icon="📄",
    layout="wide",
)

st.title("Resume Redaction Web App")
st.write("Use this tool to redact sensitive information from resumes.")

# Initialize session state
if "model_manager" not in st.session_state:
    st.session_state.model_manager = None

# Sidebar for configuration
st.sidebar.header("Configuration")

model_directory = st.sidebar.text_input(
    "JobBERT Model Directory",
    value=str(PROJECT_ROOT / "models" / "jobbert_model")
)

if st.sidebar.button("Load Model"):
    try:
        st.session_state.model_manager = ModelManager(model_directory)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Error loading model: {e}")

# Directory Selection
st.header("Step 1: Select Directories")

input_dir = st.text_input(
    "Input Directory",
    value=str(PROJECT_ROOT / "data" / "redact_input")
)

output_dir = st.text_input(
    "Output Directory",
    value=str(PROJECT_ROOT / "data" / "redact_output")
)

# Get PII types from config
pii_types = config["pii_types"]

# Process Files
st.header("Step 2: Process Files")

if st.button("Start Redaction"):
    if st.session_state.model_manager is None:
        st.error("Please load the model first.")
    else:
        if not os.path.exists(input_dir):
            st.error("Input directory does not exist.")
        else:
            os.makedirs(output_dir, exist_ok=True)
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            
            if not pdf_files:
                st.warning("No PDF files found in the input directory.")
            else:
                processor = PDFProcessor(
                    st.session_state.model_manager,
                    pii_types,
                    confidence_threshold=config.get("confidence_threshold", 0.75),
                    test_mode=config.get("test_mode", False)
                )
                
                progress_bar = st.progress(0)
                stats = {"total_files": len(pdf_files), "processed": 0, "total_words": 0, "total_redacted": 0}
                
                for idx, pdf_file in enumerate(pdf_files):
                    input_path = os.path.join(input_dir, pdf_file)
                    output_path = os.path.join(output_dir, f"candidate_{idx + 1}_redacted.pdf")
                    
                    results = processor.redact_pdf(input_path, output_path)
                    
                    stats["processed"] += 1
                    stats["total_words"] += results["total_words"]
                    stats["total_redacted"] += results["redacted_words"]
                    
                    progress_bar.progress((idx + 1) / len(pdf_files))
                
                st.success("Redaction complete!")
                st.write(f"Processed {stats['processed']} of {stats['total_files']} files")
                st.write(f"Total words processed: {stats['total_words']}")
                st.write(f"Total words redacted: {stats['total_redacted']}")
                
                if stats['total_words'] > 0:
                    redaction_rate = (stats['total_redacted'] / stats['total_words'] * 100)
                    st.write(f"Overall redaction rate: {redaction_rate:.2f}%")