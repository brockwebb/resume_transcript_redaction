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
            for i in range(0, len(tokens), max_tokens):
                split_sentences.append(" ".join(tokens[i:i + max_tokens]))
        else:
            split_sentences.append(sentence)

    return split_sentences

def setup_presidio_analyzer(custom_patterns: List[Dict], detection_config: Dict) -> AnalyzerEngine:
    """Set up Presidio Analyzer with all patterns dynamically."""
    analyzer = AnalyzerEngine()
    all_patterns = []

    # Dynamically load all patterns from detection_config
    for key, patterns in detection_config.items():
        if isinstance(patterns, list):  # Ensure the value is a list of patterns
            all_patterns.extend(patterns)

    # Add custom patterns if provided
    if custom_patterns:
        all_patterns.extend(custom_patterns)

    # Register all patterns with the analyzer
    for pattern in all_patterns:
        recognizer = PatternRecognizer(
            supported_entity=pattern["entity_type"],
            patterns=[Pattern(
                name=pattern["name"],
                regex=pattern["regex"],
                score=pattern.get("score", 0.7)
            )]
        )
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


class RedactionProcessor:
    def __init__(self, custom_patterns: List[Dict], keep_words_path: str = None,
                 confidence_threshold: float = None, model_directory: str = None,
                 detection_patterns_path: str = "detection_patterns.yaml"):

        self.detection_config = self._load_detection_config(detection_patterns_path)
        self.medical_config = self.detection_config.get('medical_detection', {}).get('jobbert', {})
        self.analyzer = setup_presidio_analyzer(custom_patterns, self.detection_config)
        self.confidence_threshold = confidence_threshold
        self.custom_terms = []
        self.keep_words = set()
        self.discard_words = set()
        self.validate_pdfs = False
        self.ocr_enabled = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModel.from_pretrained(model_directory).to(self.device)
        self.jobbert_sensitivity = self.detection_config.get('jobbert', {}).get('jobbert_sensitivity')
        if self.jobbert_sensitivity is None:
            self.jobbert_sensitivity = 0.65  # Default value
        self._initialize_medical_reference_embeddings()
        if keep_words_path:
            self._load_custom_word_filters(keep_words_path)

    def _load_custom_word_filters(self, path: str):
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
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading detection patterns from {path}: {e}")
            return {}

    def _initialize_medical_reference_embeddings(self):
        if not self.medical_config.get('reference_phrases'):
            return
        max_length = self.medical_config.get('max_length', 64)
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
        if not texts:
            return []
        max_length = self.medical_config.get('max_length', 64)
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
        return [torch.norm(emb).item() for emb in embeddings]

    def _detect_redactions(self, wordlist) -> List[fitz.Rect]:
        redact_areas = []
        page_text = " ".join(word["text"] if isinstance(word, dict) else word[4] for word in wordlist)
        max_length = self.medical_config.get('max_length', 64)
        sentences = _split_text_into_sentences(page_text, max_tokens=max_length)

        for sentence in sentences:
            presidio_results = self.analyzer.analyze(
                text=sentence,
                language="en",
                score_threshold=self.confidence_threshold
            )

            for result in presidio_results:
                matched_text = sentence[result.start:result.end]
                if result.entity_type in {"PHONE_NUMBER", "EMAIL_ADDRESS", "SOCIAL_MEDIA", "DATE", "PERSON_NAME", "SCHOOL"}:
                    bbox = self._get_bbox_for_match(matched_text, wordlist)
                    if bbox:
                        redact_areas.append(bbox)
                    continue
                jobbert_confidence = self._validate_with_jobbert([matched_text])[0]
                if jobbert_confidence >= self.jobbert_sensitivity:
                    bbox = self._get_bbox_for_match(matched_text, wordlist)
                    if bbox:
                        redact_areas.append(bbox)

        return redact_areas

    def _get_bbox_for_match(self, matched_text: str, wordlist) -> fitz.Rect:
        for word in wordlist:
            raw_text = word["text"] if isinstance(word, dict) else word[4]
            normalized_text = raw_text.lower().strip()
            matched_normalized = matched_text.lower().strip()
            if matched_normalized in normalized_text:
                return word["bbox"] if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])
        return None

    def process_pdf(self, input_path: str, output_path: str, redact_color: Tuple[float, float, float] = (0, 0, 0), redact_opacity: float = 1.0) -> Dict[str, Any]:
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
