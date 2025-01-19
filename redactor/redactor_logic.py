import re
import spacy
import fitz
from pytesseract import image_to_string
from pdf2image import convert_from_path
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from typing import Dict, Any, List, Tuple
import yaml
from redactor_file_processing import RedactFileProcessor
from typing import Optional
import string
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Entity:
    """Entity class to store detected entities with their metadata."""
    text: str
    entity_type: str
    confidence: float
    start: int
    end: int

def _split_text_into_sentences(text: str) -> List[str]:
    """Split text into meaningful sentences for analysis."""
    return re.split(r'(?<=[.!?])\s+', text.strip())

def setup_presidio_analyzer(custom_patterns: List[Dict], detection_config: Dict) -> AnalyzerEngine:
    """Set up Presidio Analyzer with all patterns dynamically."""
    analyzer = AnalyzerEngine()
    all_patterns = []

    # Dynamically load all patterns from detection_config
    for key, patterns in detection_config.items():
        if isinstance(patterns, list):
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
                 confidence_threshold: float = None,
                 detection_patterns_path: str = "detection_patterns.yaml",
                 trigger_words_path: str = "custom_word_filters.yaml"):
        print("\n=== Initializing RedactionProcessor ===")
        
        # Load detection patterns once
        print("Loading detection patterns...")
        self.detection_config = self._load_detection_config(detection_patterns_path)
        
        # Initialize Presidio analyzer once
        print("Setting up Presidio analyzer...")
        self.analyzer = self._setup_presidio_analyzer(custom_patterns)
        
        # Set basic configurations
        self.confidence_threshold = confidence_threshold or 0.7
        self.custom_terms = []
        self.keep_words = set()
        self.trigger_words = set()
        
        # PDF processing settings
        self.validate_pdfs = False
        self.ocr_enabled = False
        
        # Load spaCy model
        print("Loading spaCy model...")
        self.nlp = spacy.load("en_core_web_lg")
        print("spaCy model loaded successfully")
        
        # Load word filters
        if keep_words_path:
            print("Loading keep words...")
            self._load_custom_word_filters(keep_words_path)
        
        if trigger_words_path:
            print("Loading trigger words...")
            self._load_trigger_words(trigger_words_path)
        
        print("=== Initialization Complete ===\n")
    
    def _load_trigger_words(self, path: str):
        """Load trigger words from a YAML file."""
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            self.trigger_words = set(data.get("trigger_words", []))
            print(f"Loaded trigger words: {self.trigger_words}")
        except Exception as e:
            print(f"Error loading trigger words: {e}")
            self.trigger_words = set()

    def _load_detection_config(self, path: str) -> Dict[str, Any]:
        """Load detection patterns configuration."""
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading detection patterns: {e}")
            return {}

    def _setup_presidio_analyzer(self, custom_patterns: List[Dict]) -> AnalyzerEngine:
        """Initialize and configure Presidio analyzer."""
        analyzer = AnalyzerEngine()
        all_patterns = []

        for key, patterns in self.detection_config.items():
            if isinstance(patterns, list):
                all_patterns.extend(patterns)

        if custom_patterns:
            all_patterns.extend(custom_patterns)

        for pattern in all_patterns:
            try:
                print(f"Loading pattern: {pattern['name']} | Regex: {pattern['regex']}")
                recognizer = PatternRecognizer(
                    supported_entity=pattern["entity_type"],
                    patterns=[Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern.get("score", 0.7)
                    )]
                )
                analyzer.registry.add_recognizer(recognizer)
            except Exception as e:
                print(f"Error loading pattern '{pattern.get('name', 'unknown')}': {e}")

        return analyzer

    def _load_custom_word_filters(self, path: str):
        """Load custom word filters from YAML file."""
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            self.keep_words = set(data.get('custom_word_filters', {}).get('keep_words', []))
            print(f"Loaded keep words: {self.keep_words}")
        except Exception as e:
            print(f"Error loading keep words: {e}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """Normalize text by stripping punctuation and converting to lowercase."""
        return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    @staticmethod
    def normalize_domain_specific_text(text: str) -> str:
        """Normalize text while retaining domain-specific characters."""
        return re.sub(r'[^a-zA-Z0-9@./:_-]', '', text).strip().lower()

    @staticmethod
    def _get_line_bbox(matched_text: str, wordlist: list, entity_type: str = None) -> List[fitz.Rect]:
        """Get bounding boxes for matched text within wordlist."""
        if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
            normalized_matched_text = RedactionProcessor.normalize_domain_specific_text(matched_text)
        else:
            normalized_matched_text = RedactionProcessor._normalize_text(matched_text)
    
        print(f"Looking for: '{normalized_matched_text}'")
    
        words = []
        for word in wordlist:
            if isinstance(word, dict):
                text = word.get('text', '')
                bbox = word.get('bbox')
            else:
                text = word[4]
                bbox = fitz.Rect(word[0], word[1], word[2], word[3])
    
            if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
                text = RedactionProcessor.normalize_domain_specific_text(text)
            else:
                text = RedactionProcessor._normalize_text(text)
    
            words.append({'text': text, 'bbox': bbox})
    
        match_words = normalized_matched_text.split()
        matching_bboxes = []
        current_match_bboxes = []
    
        for word in words:
            if match_words and word['text'] == match_words[0]:
                current_match_bboxes.append(word['bbox'])
                match_words.pop(0)
                if not match_words:
                    matching_bboxes.append(RedactionProcessor._combine_bboxes(current_match_bboxes))
                    current_match_bboxes = []
            else:
                current_match_bboxes = []
    
        if not matching_bboxes:
            print(f"No BBox found for: {matched_text}")
        return matching_bboxes

    @staticmethod
    def _combine_bboxes(bboxes: list) -> fitz.Rect:
        """Combine multiple bounding boxes into one unified rectangle."""
        x0 = min(bbox.x0 for bbox in bboxes if bbox)
        y0 = min(bbox.y0 for bbox in bboxes if bbox)
        x1 = max(bbox.x1 for bbox in bboxes if bbox)
        y1 = max(bbox.y1 for bbox in bboxes if bbox)
        return fitz.Rect(x0, y0, x1, y1)
    
    def _detect_redactions(self, wordlist: list) -> List[fitz.Rect]:
        """Detect areas requiring redaction using spaCy and pattern matching."""
        redact_areas = []
        
        print("\n--- Starting Detection for New Page ---")
        print(f"Processing {len(wordlist)} words")
        
        # Group words by line using y-coordinates
        lines = []
        current_line = []
        prev_y = None
        
        for word in wordlist:
            bbox = word.get('bbox') if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])
            text = word.get('text', '') if isinstance(word, dict) else word[4]
            
            if prev_y is not None and abs(bbox.y0 - prev_y) > 5:
                lines.append(current_line)
                current_line = []
            
            current_line.append({'text': text, 'bbox': bbox})
            prev_y = bbox.y0
        
        if current_line:
            lines.append(current_line)
        
        print(f"Grouped into {len(lines)} lines")
        
        # Process each line
        for line_idx, line in enumerate(lines, 1):
            context_text = " ".join(word['text'] for word in line)
            print(f"\nProcessing line {line_idx}: {context_text[:50]}...")
            
            # Use spaCy for entity detection
            doc = self.nlp(context_text)
            for ent in doc.ents:
                if ent.label_ in {'PERSON', 'ORG', 'GPE', 'LOC'}:
                    print(f"spaCy found entity: {ent.label_} - '{ent.text}'")
                    bboxes = self._get_line_bbox(ent.text, line)
                    if bboxes:
                        redact_areas.extend(bboxes)
            
            # Check trigger words
            for word in line:
                if any(trigger in word['text'].lower() for trigger in self.trigger_words):
                    print(f"Trigger word found in: {word['text']}")
                    redact_areas.append(word['bbox'])
                    continue
            
            # Use pattern detection for specific patterns
            try:
                results = self.analyzer.analyze(
                    text=context_text,
                    language="en",
                    score_threshold=self.confidence_threshold
                )
                for result in results:
                    matched_text = context_text[result.start:result.end]  # Back to using start/end indices
                    if not any(word.lower() in matched_text.lower() for word in self.keep_words):
                        print(f"Pattern matched: {result.entity_type} - '{matched_text}'")
                        bboxes = self._get_line_bbox(matched_text, line, result.entity_type)
                        if bboxes:
                            redact_areas.extend(bboxes)
            except Exception as e:
                print(f"Error in pattern analysis: {e}")
        
        print(f"\nFound {len(redact_areas)} areas to redact")
        return redact_areas

    def process_pdf(self, input_path: str, output_path: str, redact_color: Tuple[float, float, float] = (0, 0, 0), redact_opacity: float = 1.0):
        """Process a PDF file and apply redactions."""
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