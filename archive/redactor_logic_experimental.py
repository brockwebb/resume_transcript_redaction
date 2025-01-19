import re
import fitz
import torch
from transformers import AutoTokenizer, AutoModel
from presidio_analyzer import AnalyzerEngine
from typing import Dict, Any, List, Tuple

from redactor_base import RedactorBase
from redactor_patterns import RedactorPatterns
from redactor_jobbert import JobBERTProcessor
from redactor_file_processing import RedactFileProcessor
from redactor_utils import normalize_text, combine_bboxes
from redactor_config import load_detection_patterns, load_trigger_words

class RedactionProcessor(RedactorBase):
    def __init__(self, custom_patterns: List[Dict], keep_words_path: str = None,
                 confidence_threshold: float = None, model_directory: str = None,
                 detection_patterns_path: str = "detection_patterns.yaml",
                 trigger_words_path: str = "custom_word_filters.yaml"):
        """Initialize RedactionProcessor with all necessary components."""
        super().__init__(config_path=detection_patterns_path)
        
        # Initialize components
        self.detection_config = load_detection_patterns(detection_patterns_path)
        self.pattern_processor = RedactorPatterns(self.detection_config)
        self.file_processor = RedactFileProcessor()
        
        # Basic configurations
        self.confidence_threshold = confidence_threshold or 0.7
        self.keep_words = set()
        self.trigger_words = load_trigger_words(trigger_words_path)
        
        # PDF processing settings
        self.validate_pdfs = False
        self.ocr_enabled = False
        
        # Initialize JobBERT if model directory provided
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if model_directory:
            self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
            self.model = AutoModel.from_pretrained(model_directory).to(self.device)
            self.jobbert_processor = JobBERTProcessor(model_directory)
        else:
            self.tokenizer = None
            self.model = None
            self.jobbert_processor = None

        # Initialize analyzer with custom patterns
        self.analyzer = AnalyzerEngine()
        if custom_patterns:
            self.pattern_processor = RedactorPatterns({'custom': custom_patterns})
            
        # Load custom word filters if provided
        if keep_words_path:
            self._load_custom_word_filters(keep_words_path)

    def _load_custom_word_filters(self, path: str):
        """Load custom word filters from YAML file."""
        config = self._load_config(path)
        self.keep_words = set(config.get('custom_word_filters', {}).get('keep_words', []))
        self.log_info(f"Loaded keep words: {self.keep_words}")

    def _detect_redactions(self, wordlist: list) -> List[fitz.Rect]:
        """Detect areas requiring redaction."""
        redact_areas = []
        
        # Group words by line
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
            
        # Process each line
        for line in lines:
            context_text = " ".join(word['text'] for word in line)
            
            # Check for trigger words
            for word in line:
                if any(trigger in word['text'].lower() for trigger in self.trigger_words):
                    redact_areas.append(word['bbox'])
                    
            # Use pattern processor for detection
            detections = self.pattern_processor.detect_patterns(context_text)
            for detection in detections:
                start, end = detection['start'], detection['end']
                matched_text = context_text[start:end]
                bboxes = self._get_line_bbox(matched_text, line, detection['entity_type'])
                if bboxes:
                    redact_areas.extend(bboxes)
                    
        return redact_areas

    def _get_line_bbox(self, matched_text: str, wordlist: list, entity_type: str = None) -> List[fitz.Rect]:
        """Get bounding boxes for matched text."""
        if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
            normalized_matched_text = self.normalize_text(matched_text, domain_specific=True)
        else:
            normalized_matched_text = self.normalize_text(matched_text)
            
        words = []
        for word in wordlist:
            if isinstance(word, dict):
                text = word['text']
                bbox = word['bbox']
            else:
                text = word[4]
                bbox = fitz.Rect(word[0], word[1], word[2], word[3])
                
            if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
                text = self.normalize_text(text, domain_specific=True)
            else:
                text = self.normalize_text(text)
                
            words.append({'text': text, 'bbox': bbox})
            
        # Match words and collect bounding boxes
        match_words = normalized_matched_text.split()
        matching_bboxes = []
        current_match_bboxes = []
        
        for word in words:
            if match_words and word['text'] == match_words[0]:
                current_match_bboxes.append(word['bbox'])
                match_words.pop(0)
                if not match_words:
                    matching_bboxes.append(combine_bboxes(current_match_bboxes))
                    current_match_bboxes = []
            else:
                current_match_bboxes = []
                
        return matching_bboxes

    def process_pdf(self, input_path: str, output_path: str, redact_color: Tuple[float, float, float] = (0, 0, 0), redact_opacity: float = 1.0):
        """Process PDF for redaction."""
        self.file_processor.set_redact_color(redact_color)
        self.file_processor.set_redact_opacity(redact_opacity)
        return self.file_processor.process_pdf(
            input_path=input_path,
            output_path=output_path,
            detect_redactions=self._detect_redactions
        )