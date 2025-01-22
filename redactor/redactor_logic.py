# redactor/redactor_logic.py
import re
import yaml
import fitz
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import string
from pathlib import Path

# Keep pattern matcher for fallback
from .detectors.pattern_matcher import PatternMatcher
# Add new detection system
from .detectors.ensemble_coordinator import EnsembleCoordinator
from .detectors.base_detector import Entity
from .redactor_file_processing import RedactFileProcessor
from app.utils.logger import RedactionLogger
from app.utils.config_loader import ConfigLoader

class RedactionProcessor:
    """Processor for detecting and applying redactions to documents."""

    def __init__(self, 
                 custom_patterns: Dict, 
                 keep_words_path: Optional[str] = None,
                 confidence_threshold: Optional[float] = None,
                 detection_patterns_path: str = "detection_patterns.yaml",
                 trigger_words_path: str = "custom_word_filters.yaml",
                 logger: Optional[RedactionLogger] = None):
        """Initialize RedactionProcessor with both detection systems."""
        
        # Initialize logger
        self.logger = logger if logger else RedactionLogger(
            name="redaction_processor",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )

        self.logger.info("=== Initializing RedactionProcessor ===")
        
        # Initialize config loader
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()

        # Set basic configurations
        self.confidence_threshold = confidence_threshold or 0.7
        self.keep_words = set()
        self.trigger_words = set()

        # PDF processing settings
        self.validate_pdfs = False
        self.ocr_enabled = False

        # Initialize pattern matcher (fallback detection)
        try:
            # Flatten the patterns structure
            flattened_patterns = []
            for category, patterns in custom_patterns.items():
                if isinstance(patterns, list):
                    flattened_patterns.extend(patterns)

            # Initialize pattern matcher with flattened list
            self.pattern_matcher = PatternMatcher(
                detection_config={'patterns': flattened_patterns},
                confidence_threshold=self.confidence_threshold,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Error initializing pattern matcher: {str(e)}")
            raise

        # Initialize ensemble detection system
        try:
            self.ensemble = EnsembleCoordinator(
                config_loader=self.config_loader,
                logger=self.logger
            )
        except Exception as e:
            self.logger.error(f"Error initializing ensemble detection: {str(e)}")
            raise

        # Load word filters if provided
        if keep_words_path:
            self.logger.info("Loading keep words...")
            self._load_custom_word_filters(keep_words_path)

        if trigger_words_path:
            self.logger.info("Loading trigger words...")
            self._load_trigger_words(trigger_words_path)

        self.logger.info("=== Initialization Complete ===")

    def _detect_redactions(self, wordlist: list) -> List[fitz.Rect]:
        """Detect areas requiring redaction using ensemble detection with pattern matcher fallback."""
        redact_areas = []
        
        self.logger.debug(f"Processing wordlist with {len(wordlist)} items")
        
        text = self._get_text_from_wordlist(wordlist)
        
        # First try ensemble detection
        try:
            entities = self.ensemble.detect_entities(text)
            self.logger.debug(f"Found {len(entities)} entities using ensemble detection")
        except Exception as e:
            self.logger.error(f"Ensemble detection failed: {str(e)}")
            entities = []

        # If ensemble detection fails or finds nothing, fall back to pattern matcher
        if not entities:
            self.logger.debug("Falling back to pattern matcher")
            pattern_entities = self.pattern_matcher.detect_entities(text)
            entities.extend([
                Entity(
                    text=ent["text"],
                    entity_type=ent["entity_type"],
                    confidence=ent["confidence"],
                    start=ent["start"],
                    end=ent["end"],
                    source="pattern_matcher",  # Add source parameter
                    metadata=ent.get("metadata", {})  # Optional metadata
                ) for ent in pattern_entities
            ])
        
        self.logger.debug(f"Found {len(entities)} total entities before keep_words filtering")
        
        # Convert entities to redaction areas
        for entity in entities:
            self.logger.debug(f"Processing entity: {entity}")
            
            # Check if this is a keep_word before redacting
            if self._should_keep(entity.text):
                self.logger.info(f"Preserving term '{entity.text}' matched in keep_words")
                continue
                
            # Also check normalized version
            normalized_text = self._normalize_text(entity.text)
            if self._should_keep(normalized_text):
                self.logger.info(f"Preserving term '{entity.text}' (normalized: '{normalized_text}') matched in keep_words")
                continue
                
            # If we get here, term should be redacted
            self.logger.debug(f"Term will be redacted: '{entity.text}' (type: {entity.entity_type})")
            
            # Get bounding boxes for the entity
            bboxes = self._get_entity_bbox(entity.text, wordlist, entity.entity_type)
            
            if bboxes:
                redact_areas.extend(bboxes)
            else:
                self.logger.warning(f"No bounding boxes found for entity: {entity.text}")

        self.logger.info(f"Final redaction count: {len(redact_areas)} areas after keep_words filtering")
        return redact_areas

    def _normalize_text(self, text: str) -> str:
        """Normalize text while preserving certain special characters."""
        if not text:
            return ""
            
        # Convert to lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove all punctuation except .-_ and space
        normalized = re.sub(r'[^\w\s.\-_]', '', normalized)
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        self.logger.debug(f"Normalized text: '{text}' -> '{normalized}'")
        return normalized
    
    @staticmethod
    def normalize_domain_specific_text(text: str) -> str:
        """Normalize text while retaining domain-specific characters."""
        return re.sub(r'[^a-zA-Z0-9@./:_-]', '', text).strip().lower()

    def _should_keep(self, text: str) -> bool:
        """Check if text should be kept (not redacted)."""
        if not text:
            return False
        
        normalized_text = self._normalize_text(text)
        normalized_keep_words = {self._normalize_text(k) for k in self.keep_words}
        
        # Direct match check
        if normalized_text in normalized_keep_words:
            self.logger.debug(f"Direct keep word match: '{text}' -> '{normalized_text}'")
            return True
            
        # Multi-word term checks
        text_parts = set(normalized_text.split())
        for keep_word in normalized_keep_words:
            keep_word_parts = set(keep_word.split())
            
            # Check if all parts of the keep_word appear in the text
            if len(keep_word_parts) > 1 and keep_word_parts.issubset(text_parts):
                self.logger.debug(f"Multi-word keep match: '{text}' contains '{keep_word}'")
                return True
                
            # Check if text is part of a keep_word
            if len(text_parts) == 1 and text_parts.issubset(keep_word_parts):
                self.logger.debug(f"Partial keep word match: '{text}' is part of '{keep_word}'")
                return True
        
        return False

    def _load_custom_word_filters(self, path: str):
        """Load custom word filters from YAML file."""
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
                
            # Load keep_words from custom_word_filters section
            if isinstance(data, dict) and 'custom_word_filters' in data:
                self.keep_words = set(data['custom_word_filters'].get('keep_words', []))
            else:
                self.keep_words = set()
                
            self.logger.debug(f"Loaded keep words: {self.keep_words}")
            
        except Exception as e:
            self.logger.error(f"Error loading keep words: {e}")
            self.keep_words = set()

    def _load_trigger_words(self, path: str):
        """Load trigger words from a YAML file."""
        try:
            with open(path, 'r') as file:
                data = yaml.safe_load(file)
            self.trigger_words = set(data.get('trigger_words', []))
            self.logger.debug(f"Loaded trigger words: {self.trigger_words}")
        except Exception as e:
            self.logger.error(f"Error loading trigger words: {e}")
            self.trigger_words = set()

    def _get_entity_bbox(self, matched_text: str, wordlist: list, entity_type: str = None) -> List[fitz.Rect]:
        """Get bounding boxes for matched text within wordlist."""
        if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
            normalized_matched_text = self.normalize_domain_specific_text(matched_text)
        else:
            normalized_matched_text = self._normalize_text(matched_text)

        self.logger.debug(f"Looking for: '{normalized_matched_text}'")

        words = []
        for word in wordlist:
            if isinstance(word, dict):
                text = word.get('text', '')
                bbox = word.get('bbox')
            else:
                text = word[4]
                bbox = fitz.Rect(word[0], word[1], word[2], word[3])

            if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
                text = self.normalize_domain_specific_text(text)
            else:
                text = self._normalize_text(text)

            words.append({'text': text, 'bbox': bbox})

        match_words = normalized_matched_text.split()
        matching_bboxes = []
        current_match_bboxes = []

        for word in words:
            if match_words and word['text'] == match_words[0]:
                current_match_bboxes.append(word['bbox'])
                match_words.pop(0)
                if not match_words:
                    matching_bboxes.append(self._combine_bboxes(current_match_bboxes))
                    current_match_bboxes = []
            else:
                current_match_bboxes = []

        if not matching_bboxes:
            self.logger.debug(f"No BBox found for: {matched_text}")
        return matching_bboxes

    @staticmethod
    def _combine_bboxes(bboxes: list) -> fitz.Rect:
        """Combine multiple bounding boxes into one unified rectangle."""
        if not bboxes:
            return None
            
        x0 = min(bbox.x0 for bbox in bboxes if bbox)
        y0 = min(bbox.y0 for bbox in bboxes if bbox)
        x1 = max(bbox.x1 for bbox in bboxes if bbox)
        y1 = max(bbox.y1 for bbox in bboxes if bbox)
        return fitz.Rect(x0, y0, x1, y1)

    def _get_text_from_wordlist(self, wordlist: list) -> str:
        """Extract text from word list maintaining proper spacing."""
        text_parts = []
        last_y = None
        
        for word in wordlist:
            # Handle both dict and tuple formats from PDF extraction
            if isinstance(word, dict):
                text = word.get('text', '')
                y_pos = word.get('bbox', {}).get('y0')
            else:  # Tuple format
                text = word[4]
                y_pos = word[1]
            
            # Add newline if y position changes significantly
            if last_y is not None and abs(y_pos - last_y) > 5:
                text_parts.append('\n')
            
            text_parts.append(text)
            last_y = y_pos
        
        return ' '.join(text_parts)

    def process_pdf(self, 
                   input_path: str, 
                   output_path: str, 
                   redact_color: Tuple[float, float, float] = (0, 0, 0), 
                   redact_opacity: float = 1.0) -> Dict[str, int]:
        """Process a PDF file and apply redactions."""
        self.logger.info(f"Processing PDF: {input_path}")
        try:
            file_processor = RedactFileProcessor(
                redact_color=redact_color,
                redact_opacity=redact_opacity,
                ocr_enabled=self.ocr_enabled,
                validate_pdfs=self.validate_pdfs
            )

            stats = file_processor.process_pdf(
                input_path=input_path,
                output_path=output_path,
                detect_redactions=self._detect_redactions
            )

            self.logger.info(f"PDF processing complete: {stats}")
            return stats

        except Exception as e:
            self.logger.error(f"Error processing PDF {input_path}: {str(e)}")
            raise