import re
import fitz
from pytesseract import image_to_string
from pdf2image import convert_from_path
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from typing import Dict, Any, List, Tuple
import yaml
from redactor_file_processing import RedactFileProcessor

def _split_text_into_sentences(text: str, max_tokens: int = None) -> List[str]:
    """
    Split text into meaningful sentences for analysis, capping length for JobBERT.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())  # Split by punctuation
    split_sentences = []

    for sentence in sentences:
        tokens = sentence.split()
        if max_tokens is not None and len(tokens) > max_tokens:
            # Split overly long sentences into smaller chunks
            for i in range(0, len(tokens), max_tokens):
                split_sentences.append(" ".join(tokens[i:i + max_tokens]))
        else:
            split_sentences.append(sentence)

    return split_sentences

def setup_presidio_analyzer(custom_patterns: List[Dict]) -> AnalyzerEngine:
    """Set up Presidio Analyzer with custom patterns."""
    analyzer = AnalyzerEngine()

    for pattern in custom_patterns:
        recognizer = PatternRecognizer(
            supported_entity=pattern["entity_type"],
            patterns=[Pattern(
                name=pattern["name"],
                regex=pattern["regex"],
                score=pattern["score"]
            )]
        )
        analyzer.registry.add_recognizer(recognizer)
    return analyzer

class RedactionProcessor:
    def __init__(self, custom_patterns: List[Dict], keep_words_path: str = None,
                 confidence_threshold: float = None, model_directory: str = None,
                 detection_patterns_path: str = "detection_patterns.yaml"):

        # Basic configuration
        self.analyzer = setup_presidio_analyzer(custom_patterns)
        self.confidence_threshold = confidence_threshold
        self.custom_terms = []
        self.keep_words = set()
        self.discard_words = set()

        # PDF processing settings
        self.validate_pdfs = False
        self.ocr_enabled = False

        # Load JobBERT model and tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModel.from_pretrained(model_directory).to(self.device)

        # Load detection patterns and medical config
        self.detection_config = self._load_detection_config(detection_patterns_path)
        self.medical_config = self.detection_config.get('medical_detection', {}).get('jobbert', {})

        # Set JobBERT sensitivity
        self.jobbert_sensitivity = self.detection_config.get('jobbbert_sensitivity', 0.85)

        # Initialize reference embeddings for medical detection
        self._initialize_medical_reference_embeddings()

        # Load custom word filters if provided
        if keep_words_path:
            self._load_custom_word_filters(keep_words_path)

    def _load_custom_word_filters(self, path: str):
        """Load custom word filters for keep and discard lists."""
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)

            self.keep_words = set(data.get('custom_word_filters', {}).get('keep_words', []))
            self.discard_words = set(data.get('custom_word_filters', {}).get('discard_words', []))

        except Exception as e:
            print(f"Error loading custom word filters: {e}")
            self.keep_words = set()
            self.discard_words = set()

    def _load_detection_config(self, path: str) -> Dict[str, Any]:
        """Load all detection patterns and configurations from YAML file."""
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading detection patterns from {path}: {e}")
            return {}

    def _initialize_medical_reference_embeddings(self):
        """Initialize reference embeddings for medical context detection."""
        if not self.medical_config.get('reference_phrases'):
            return

        max_length = self.medical_config.get('max_length', 64)  # Fetch from config, default to 64

        inputs = self.tokenizer(
            self.medical_config['reference_phrases'],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
        self._reference_embeddings = outputs.last_hidden_state.mean(dim=1)

    def _validate_with_jobbert(self, texts: List[str]) -> List[float]:
        """Validate text using JobBERT for general sensitivity detection."""
        if not texts:
            return []

        max_length = self.medical_config.get('max_length', 64)  # Fetch from config

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        confidence_scores = [torch.norm(emb).item() for emb in embeddings]

        return confidence_scores

    def _detect_redactions(self, wordlist) -> List[fitz.Rect]:
        """Enhanced redaction detection with JobBERT for all contexts."""
        redact_areas = []

        # Get text for context
        page_text = " ".join(word["text"] if isinstance(word, dict) else word[4]
                            for word in wordlist)

        # Split text into manageable sentences for JobBERT
        max_length = self.medical_config.get('max_length', 64)  # From config
        sentences = _split_text_into_sentences(page_text, max_tokens=max_length)

        for sentence in sentences:
            # Presidio pattern matching
            presidio_results = self.analyzer.analyze(
                text=sentence,
                language="en",
                score_threshold=self.confidence_threshold
            )

            for result in presidio_results:
                matched_text = sentence[result.start:result.end]

                # Validate with JobBERT
                jobbert_confidence = self._validate_with_jobbert([matched_text])[0]
                if jobbert_confidence >= self.jobbert_sensitivity:
                    for word in wordlist:
                        raw_text = word["text"] if isinstance(word, dict) else word[4]
                        normalized_text = raw_text.lower().strip()
                        matched_normalized = matched_text.lower().strip()

                        if matched_normalized in normalized_text:
                            bbox = word["bbox"] if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])
                            redact_areas.append(bbox)

        return redact_areas

    def process_pdf(self, input_path: str, output_path: str, redact_color: Tuple[float, float, float] = (0, 0, 0), redact_opacity: float = 1.0) -> Dict[str, Any]:
        """
        Process a single PDF file with specified redaction settings.

        :param input_path: Path to the input PDF.
        :param output_path: Path to save the processed PDF.
        :param redact_color: The color for redaction/highlighting.
        :param redact_opacity: The opacity for highlighting.
        :return: Statistics about the processing.
        """
        file_processor = RedactFileProcessor(
            redact_color=redact_color,
            redact_opacity=redact_opacity,
            ocr_enabled=self.ocr_enabled,
            validate_pdfs=self.validate_pdfs
        )
        return file_processor.process_pdf(
            input_path=input_path,
            output_path=output_path,
            detect_redactions=self._detect_redactions
        )
