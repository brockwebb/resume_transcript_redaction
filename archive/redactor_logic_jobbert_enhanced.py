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
from typing import Optional
import string

#INSERT _split_text_into_sentences
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

#INSERT setup_presidio_analyzer
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
                 confidence_threshold: float = None, model_directory: str = None,
                 detection_patterns_path: str = "detection_patterns.yaml",
                 trigger_words_path: str = "custom_word_filters.yaml",
                 debug: bool = True):
        # Load detection patterns
        self.detection_config = self._load_detection_config(detection_patterns_path)
        
        # Initialize Presidio analyzer
        self.analyzer = self._setup_presidio_analyzer(custom_patterns)
        
        # Basic configurations
        self.confidence_threshold = confidence_threshold or 0.7
        self.debug = debug
        self.custom_terms = []
        self.keep_words = set()
        self.trigger_words = set()
        
        # PDF processing settings
        self.validate_pdfs = False
        self.ocr_enabled = False
        
        # Load NLP model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_directory:
            self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
            self.model = AutoModel.from_pretrained(model_directory).to(self.device)
            self.jobbert = JobBERTProcessor(model_directory)
        else:
            self.jobbert = None
        
        # Load word filters and trigger words
        if keep_words_path:
            self._load_custom_word_filters(keep_words_path)
        if trigger_words_path:
            self._load_trigger_words(trigger_words_path)
            
        # Entity configuration
        self.entity_config = {
            'use_jobbert_for': {
                'PERSON_NAME': True,
                'ORGANIZATION': True,
                'JOB_TITLE': True,
                'EDUCATION': True,
                'SKILLS': True,
                'LOCATION': True
            },
            'max_tokens_per_chunk': 64,
            'thresholds': {
                'PERSON_NAME': 0.85,
                'ORGANIZATION': 0.80,
                'JOB_TITLE': 0.75,
                'EDUCATION': 0.80,
                'LOCATION': 0.75,
                'SKILLS': 0.70
            }
        }

    def _debug_print(self, message: str):
        """Print debug messages if debug mode is enabled."""
        if self.debug:
            print(f"[DEBUG] {message}")

    def _load_trigger_words(self, path: str):
        """
        Load trigger words from a YAML file with proper structure handling.
        """
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            self.trigger_words = set(data.get("trigger_words", []))
            print(f"Loaded trigger words: {self.trigger_words}")  # Debug output
        except Exception as e:
            print(f"Error loading trigger words: {e}")
            self.trigger_words = set()


    def _load_detection_config(self, path: str) -> Dict[str, Any]:
        try:
            with open(path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Error loading detection patterns: {e}")
            return {}

    def _setup_presidio_analyzer(self, custom_patterns: List[Dict]) -> AnalyzerEngine:
        analyzer = AnalyzerEngine()
        all_patterns = []

        for key, patterns in self.detection_config.items():
            if isinstance(patterns, list):
                all_patterns.extend(patterns)

        if custom_patterns:
            all_patterns.extend(custom_patterns)

        for pattern in all_patterns:
            try:
                print(f"Loading pattern: {pattern['name']} | Regex: {pattern['regex']}")  # Debug output
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
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            self.keep_words = set(data.get('custom_word_filters', {}).get('keep_words', []))
            print(f"Loaded keep words: {self.keep_words}")  # Debugging output
        except Exception as e:
            print(f"Error loading keep words: {e}")

    @staticmethod
    def _normalize_text(text: str) -> str:
        """
        Normalize text by stripping punctuation and converting to lowercase.
    
        Args:
            text (str): Input text.
    
        Returns:
            str: Normalized text.
        """
        return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

    @staticmethod
    def normalize_domain_specific_text(text: str) -> str:
        """
        Normalize text while retaining domain-specific characters like '.', '/', '@', and ':'.
        
        Args:
            text (str): Input text.
            
        Returns:
            str: Normalized text for domain-specific use cases.
        """
        return re.sub(r'[^a-zA-Z0-9@./:_-]', '', text).strip().lower()

    
    @staticmethod
    def _get_line_bbox(matched_text: str, wordlist: list, entity_type: str = None) -> List[fitz.Rect]:
        """
        Identifies bounding boxes for matched text within wordlist, with domain-specific handling.
    
        Args:
            matched_text (str): Text to find bounding boxes for.
            wordlist (list): List of words with their positions.
            entity_type (str): Optional entity type to apply domain-specific normalization.
    
        Returns:
            List[fitz.Rect]: List of combined bounding boxes for matched text.
        """
        # Normalize matched text
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
    
            # Normalize text
            if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
                text = RedactionProcessor.normalize_domain_specific_text(text)
            else:
                text = RedactionProcessor._normalize_text(text)
    
            words.append({'text': text, 'bbox': bbox})
    
        # Split matched text into words and match line-by-line
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
        """
        Combines multiple bounding boxes into one unified rectangle.
    
        Args:
            bboxes (list): List of bounding boxes to combine.
    
        Returns:
            fitz.Rect: Combined bounding box.
        """
        x0 = min(bbox.x0 for bbox in bboxes if bbox)
        y0 = min(bbox.y0 for bbox in bboxes if bbox)
        x1 = max(bbox.x1 for bbox in bboxes if bbox)
        y1 = max(bbox.y1 for bbox in bboxes if bbox)
        return fitz.Rect(x0, y0, x1, y1)

    
    @staticmethod
    def _aggregate_line_bboxes(bboxes: list) -> fitz.Rect:
        """
        Combines bounding boxes for a single line into one rectangle.
        
        Args:
            bboxes (list): List of bounding boxes to combine.
        
        Returns:
            fitz.Rect: Combined bounding box for the line.
        """
        x0 = min(bbox.x0 for bbox in bboxes if bbox)
        y0 = min(bbox.y0 for bbox in bboxes if bbox)
        x1 = max(bbox.x1 for bbox in bboxes if bbox)
        y1 = max(bbox.y1 for bbox in bboxes if bbox)
        return fitz.Rect(x0, y0, x1, y1)

    def _detect_redactions(self, wordlist: list) -> List[fitz.Rect]:
        """Main redaction detection method with detailed debug output."""
        redact_areas = []
        lines = []
        current_line = []
        prev_y = None
        
        # Group words into lines
        for word in wordlist:
            bbox = word.get('bbox') if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])
            text = word.get('text', '') if isinstance(word, dict) else word[4]
            
            if prev_y is not None and abs(bbox.y0 - prev_y) > 5:
                self._debug_print(f"Completed line: {' '.join(w['text'] for w in current_line)}")
                lines.append(current_line)
                current_line = []
            
            current_line.append({'text': text, 'bbox': bbox})
            prev_y = bbox.y0
        
        if current_line:
            lines.append(current_line)
        
        # Process each line
        for line_idx, line in enumerate(lines):
            self._debug_print(f"\nProcessing line {line_idx + 1}")
            context_words = line
            context_text = " ".join(word['text'] for word in context_words)
            self._debug_print(f"Line text: {context_text}")
            line_should_be_redacted = False
            
            # Step 1: Check trigger words
            self._debug_print("Checking trigger words...")
            for word in context_words:
                word_text = word['text']
                word_bbox = word['bbox']
                
                if any(trigger in word_text.lower() for trigger in self.trigger_words):
                    self._debug_print(f"Trigger word '{word_text}' detected.")
                    redact_areas.append(word_bbox)
                    line_should_be_redacted = True
                    continue
            
            if line_should_be_redacted:
                continue
            
            # Step 2: Run JobBERT for appropriate length text
            if self.jobbert and len(context_text.split()) <= self.entity_config['max_tokens_per_chunk']:
                self._debug_print("Running JobBERT analysis...")
                try:
                    entities = self.jobbert.identify_entities(context_text)
                    for entity in entities:
                        self._debug_print(
                            f"JobBERT detected: '{entity.text}' as {entity.entity_type} "
                            f"(confidence: {entity.confidence:.3f})"
                        )
                        
                        threshold = self.entity_config['thresholds'].get(entity.entity_type, 0.8)
                        if (self.entity_config['use_jobbert_for'].get(entity.entity_type, False) and 
                            entity.confidence >= threshold):
                            self._debug_print(f"Processing entity (above threshold {threshold})")
                            bboxes = self._get_line_bbox(entity.text, context_words, entity.entity_type)
                            if bboxes:
                                redact_areas.extend(bboxes)
                except Exception as e:
                    self._debug_print(f"Error in JobBERT processing: {e}")
            
            # Step 3: Pattern-based detection
            self._debug_print("Running pattern-based detection...")
            try:
                results = self.analyzer.analyze(
                    text=context_text,
                    language="en",
                    score_threshold=self.confidence_threshold
                )
                for result in results:
                    # Skip if this entity type is handled by JobBERT
                    if not self.entity_config['use_jobbert_for'].get(result.entity_type, False):
                        matched_text = context_text[result.start:result.end]
                        entity_type = result.entity_type
                        bboxes = self._get_line_bbox(matched_text, context_words, entity_type)
                        if bboxes:
                            redact_areas.extend(bboxes)
            except Exception as e:
                self._debug_print(f"Error in pattern analysis: {e}")
        
        self._debug_print(f"\nTotal areas to redact: {len(redact_areas)}")
        return redact_areas

    def process_pdf(self, input_path: str, output_path: str, redact_color: Tuple[float, float, float] = (0, 0, 0), redact_opacity: float = 1.0):
        """Process PDF file for redaction."""
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
