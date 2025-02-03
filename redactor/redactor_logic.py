# redactor/redactor_logic.py

###############################################################################
# SECTION 1: IMPORTS AND TYPING
###############################################################################

import re
import yaml
import fitz
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

# Base components
from .detectors.base_detector import Entity
from .detectors.ensemble_coordinator import EnsembleCoordinator
from .redactor_file_processing import RedactFileProcessor

# Utilities
from app.utils.logger import RedactionLogger
from app.utils.config_loader import ConfigLoader


###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class RedactionProcessor:
    """
    Processor for detecting and applying redactions to documents.
    Uses configuration-driven validation patterns and ensemble detection.
    """

    def __init__(self, 
                 config_loader: Optional[ConfigLoader] = None,
                 logger: Optional[RedactionLogger] = None,
                 debug_mode: bool = False):
        """
        Initialize RedactionProcessor with validation-based detection systems.

        Args:
            config_loader: Optional configuration loader instance
            logger: Optional logger instance  
            debug_mode: Enable detailed debug logging
        """
        self._init_logging(logger, debug_mode)
        self._init_config(config_loader)
        self._init_core_components()
        self._init_word_filters()
        
        self.logger.info("=== RedactionProcessor Initialization Complete ===")

    def _init_logging(self, logger: Optional[RedactionLogger], debug_mode: bool) -> None:
        """Initialize logging with optional debug mode."""
        self.debug_mode = debug_mode
        log_level = "DEBUG" if debug_mode else "INFO"
        
        self.logger = logger if logger else RedactionLogger(
            name="redaction_processor",
            log_level=log_level,
            log_dir="logs",
            console_output=debug_mode
        )
        
        self.logger.info("=== Initializing RedactionProcessor ===")
        if debug_mode:
            self.logger.debug("Debug mode enabled - verbose logging active")

    def _init_config(self, config_loader: Optional[ConfigLoader]) -> None:
        """Initialize configuration loader and load configs."""
        try:
            self.config_loader = config_loader if config_loader else ConfigLoader()
            self.config_loader.load_all_configs()
            
            # Load core settings
            core_config = self.config_loader.get_config("core_settings")
            if not core_config:
                raise ValueError("Missing core_settings configuration")
                
            self.confidence_threshold = core_config.get("confidence_threshold", 0.7)
            self.validate_pdfs = core_config.get("validate_pdfs", False)
            self.ocr_enabled = core_config.get("ocr_enabled", False)
            
            if self.debug_mode:
                self.logger.debug(f"Loaded core settings:")
                self.logger.debug(f"- Confidence threshold: {self.confidence_threshold}")
                self.logger.debug(f"- Validate PDFs: {self.validate_pdfs}")
                self.logger.debug(f"- OCR enabled: {self.ocr_enabled}")
                
        except Exception as e:
            self.logger.error(f"Error initializing configuration: {str(e)}")
            raise

    def _init_core_components(self) -> None:
            """Initialize detection systems using ensemble detection."""
            try:
                # Initialize ensemble detection system
                self.ensemble = EnsembleCoordinator(
                    config_loader=self.config_loader,
                    logger=self.logger
                )
                
                if self.debug_mode:
                    self.logger.debug("EnsembleCoordinator initialized")
                    
            except Exception as e:
                self.logger.error(f"Error initializing core components: {str(e)}")
                raise
    
    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities in text using ensemble detection.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        if not text:
            self.logger.warning("Empty text provided for entity detection")
            return []

        try:
            # Get detections using ensemble
            detected = self.ensemble.detect_entities(text)
            
            if self.debug_mode:
                self.logger.debug(f"Detected {len(detected)} entities")
                for entity in detected:
                    self.logger.debug(
                        f"{entity.entity_type}: '{entity.text}' "
                        f"({entity.confidence:.2f})"
                    )

            # Filter using keep words
            filtered = []
            for entity in detected:
                if self._should_keep(entity.text):
                    if self.debug_mode:
                        self.logger.debug(f"Keeping term '{entity.text}' matched in keep_words")
                    continue
                
                filtered.append(entity)

            if self.debug_mode:
                self.logger.debug(f"After filtering: {len(filtered)} entities")

            return filtered

        except Exception as e:
            self.logger.error(f"Error in entity detection: {str(e)}")
            return []

    def _init_word_filters(self) -> None:
        """Initialize word filter lists from configuration."""
        try:
            filter_config = self.config_loader.get_config("word_filters")
            if not filter_config:
                self.logger.warning("No word filter configuration found")
                self.keep_words = set()
                self.trigger_words = set()
                return
                
            self.keep_words = set(filter_config.get("keep_words", []))
            self.trigger_words = set(filter_config.get("trigger_words", []))
            
            if self.debug_mode:
                self.logger.debug(f"Loaded {len(self.keep_words)} keep words")
                self.logger.debug(f"Loaded {len(self.trigger_words)} trigger words")
                
        except Exception as e:
            self.logger.error(f"Error loading word filters: {str(e)}")
            self.keep_words = set()
            self.trigger_words = set()

###############################################################################
    # SECTION 3: TEXT NORMALIZATION AND FILTERING
    ###############################################################################

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text while preserving domain-specific special characters.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Load normalization config
        norm_config = self.config_loader.get_config("text_normalization")
        if not norm_config:
            if self.debug_mode:
                self.logger.debug("No normalization config found - using defaults")
            # Default normalization - convert to lowercase and strip whitespace
            normalized = text.lower().strip()
            # Remove all punctuation except .-_ and space
            normalized = re.sub(r'[^\w\s.\-_]', '', normalized)
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            return normalized
            
        try:
            # Get preservation patterns
            preserve_patterns = norm_config.get("preserve_patterns", [r'[.\-_]'])
            
            # Convert to lowercase and strip whitespace
            normalized = text.lower().strip()
            
            # Build regex pattern that excludes preserved characters
            pattern_parts = []
            for pattern in preserve_patterns:
                pattern_parts.append(f"(?!{pattern})")
            exclude_pattern = f"[^\\w\\s{''.join(pattern_parts)}]"
            
            # Remove punctuation except preserved patterns
            normalized = re.sub(exclude_pattern, '', normalized)
            
            # Normalize whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            if self.debug_mode:
                self.logger.debug(f"Normalized text: '{text}' -> '{normalized}'")
                
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error in text normalization: {str(e)}")
            return text.lower().strip()

    def _normalize_domain_text(self, text: str, domain_type: str) -> str:
        """
        Normalize text using domain-specific rules.
        
        Args:
            text: Text to normalize
            domain_type: Type of domain (e.g., 'email', 'url', 'phone')
            
        Returns:
            Domain-normalized text
        """
        if not text:
            return ""
            
        try:
            # Get domain normalization rules
            norm_config = self.config_loader.get_config("text_normalization")
            if not norm_config or "domain_rules" not in norm_config:
                if self.debug_mode:
                    self.logger.debug(f"No domain rules found for {domain_type}")
                return self._normalize_text(text)
                
            domain_rules = norm_config["domain_rules"].get(domain_type, {})
            
            # Get preserved characters for this domain
            preserved_chars = domain_rules.get("preserve_chars", "")
            if not preserved_chars:
                return self._normalize_text(text)
                
            # Build regex pattern
            pattern = f"[^a-zA-Z0-9{re.escape(preserved_chars)}]"
            
            # Apply normalization
            normalized = re.sub(pattern, '', text).strip().lower()
            
            if self.debug_mode:
                self.logger.debug(
                    f"Domain ({domain_type}) normalization: '{text}' -> '{normalized}'"
                )
                
            return normalized
            
        except Exception as e:
            self.logger.error(f"Error in domain normalization: {str(e)}")
            return self._normalize_text(text)

    def _should_keep(self, text: str) -> bool:
        """
        Check if text should be kept (not redacted).
        
        Args:
            text: Text to check
            
        Returns:
            Boolean indicating if text should be kept
        """
        if not text:
            return False
        
        try:
            normalized_text = self._normalize_text(text)
            normalized_keep_words = {self._normalize_text(k) for k in self.keep_words}
            
            # Direct match check
            if normalized_text in normalized_keep_words:
                if self.debug_mode:
                    self.logger.debug(f"Direct keep word match: '{text}' -> '{normalized_text}'")
                return True
                
            # Multi-word term checks
            text_parts = set(normalized_text.split())
            for keep_word in normalized_keep_words:
                keep_word_parts = set(keep_word.split())
                
                # Check if all parts of the keep_word appear in the text
                if len(keep_word_parts) > 1 and keep_word_parts.issubset(text_parts):
                    if self.debug_mode:
                        self.logger.debug(f"Multi-word keep match: '{text}' contains '{keep_word}'")
                    return True
                    
                # Check if text is part of a keep_word
                if len(text_parts) == 1 and text_parts.issubset(keep_word_parts):
                    if self.debug_mode:
                        self.logger.debug(f"Partial keep word match: '{text}' is part of '{keep_word}'")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking keep words: {str(e)}")
            return False

    ###############################################################################
    # SECTION 4: ENTITY TEXT EXTRACTION 
    ###############################################################################

    def _get_text_from_wordlist(self, wordlist: list) -> str:
        """
        Extract text from word list maintaining proper spacing.
        
        Args:
            wordlist: List of word objects from PDF
            
        Returns:
            Extracted text with spacing preserved
        """
        text_parts = []
        last_y = None
        
        try:
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
            
            if self.debug_mode:
                self.logger.debug(f"Extracted {len(text_parts)} text segments from wordlist")
                
            return ' '.join(text_parts)
            
        except Exception as e:
            self.logger.error(f"Error extracting text from wordlist: {str(e)}")
            return ' '.join(str(w[4]) if isinstance(w, tuple) else str(w.get('text', '')) 
                          for w in wordlist)

###############################################################################
    # SECTION 5: ENTITY DETECTION AND BOUNDING BOXES
    ###############################################################################

    def _detect_redactions(self, wordlist: list) -> List[fitz.Rect]:
        """
        Detect areas requiring redaction using ensemble detection with pattern matcher fallback.
        
        Args:
            wordlist: List of words from PDF
            
        Returns:
            List of bounding boxes for redaction
        """
        redact_areas = []
        
        if self.debug_mode:
            self.logger.debug(f"Processing wordlist with {len(wordlist)} items")
        
        text = self._get_text_from_wordlist(wordlist)
        
        # First try ensemble detection
        try:
            entities = self.ensemble.detect_entities(text)
            if self.debug_mode:
                self.logger.debug(f"Found {len(entities)} entities using ensemble detection")
                for entity in entities:
                    self.logger.debug(
                        f"Ensemble entity: {entity.entity_type} "
                        f"'{entity.text}' ({entity.confidence:.2f})"
                    )
        except Exception as e:
            self.logger.error(f"Ensemble detection failed: {str(e)}")
            entities = []

        # If ensemble detection fails or finds nothing, fall back to pattern matcher
        if not entities:
            if self.debug_mode:
                self.logger.debug("Falling back to validation pattern matcher")
            try:
                pattern_entities = self.pattern_matcher.detect_entities(text)
                if self.debug_mode:
                    self.logger.debug(f"Found {len(pattern_entities)} entities using pattern matcher")
                entities.extend(pattern_entities)
            except Exception as e:
                self.logger.error(f"Pattern matcher detection failed: {str(e)}")
        
        if self.debug_mode:
            self.logger.debug(f"Found {len(entities)} total entities before keep_words filtering")
        
        # Convert entities to redaction areas
        for entity in entities:
            if self.debug_mode:
                self.logger.debug(f"Processing entity: {entity}")
            
            # Check if this is a keep_word before redacting
            if self._should_keep(entity.text):
                self.logger.info(f"Preserving term '{entity.text}' matched in keep_words")
                continue
                
            # Also check normalized version
            normalized_text = self._normalize_text(entity.text)
            if self._should_keep(normalized_text):
                self.logger.info(
                    f"Preserving term '{entity.text}' "
                    f"(normalized: '{normalized_text}') matched in keep_words"
                )
                continue
                
            # If we get here, term should be redacted
            if self.debug_mode:
                self.logger.debug(f"Term will be redacted: '{entity.text}' (type: {entity.entity_type})")
            
            # Get bounding boxes for the entity
            bboxes = self._get_entity_bbox(entity.text, wordlist, entity.entity_type)
            
            if bboxes:
                redact_areas.extend(bboxes)
            else:
                self.logger.warning(f"No bounding boxes found for entity: {entity.text}")

        if self.debug_mode:
            self.logger.debug(f"Final redaction count: {len(redact_areas)} areas after filtering")
        return redact_areas

    def _get_entity_bbox(self, matched_text: str, wordlist: list, entity_type: str) -> List[fitz.Rect]:
        """
        Get bounding boxes for matched text within wordlist.
        
        Args:
            matched_text: Text to find bounding boxes for
            wordlist: List of words from PDF
            entity_type: Type of entity for specialized handling
            
        Returns:
            List of bounding box rectangles
        """
        # Apply appropriate normalization based on entity type
        if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
            normalized_matched_text = self._normalize_domain_text(
                matched_text, 
                domain_type="url" if entity_type == "WEBSITE" else "email"
            )
        else:
            normalized_matched_text = self._normalize_text(matched_text)

        if self.debug_mode:
            self.logger.debug(f"Looking for: '{normalized_matched_text}'")

        # Process word list
        words = []
        for word in wordlist:
            if isinstance(word, dict):
                text = word.get('text', '')
                bbox = word.get('bbox')
            else:
                text = word[4]
                bbox = fitz.Rect(word[0], word[1], word[2], word[3])

            # Apply same normalization to word text
            if entity_type in {"WEBSITE", "EMAIL_ADDRESS", "SOCIAL_MEDIA"}:
                text = self._normalize_domain_text(
                    text,
                    domain_type="url" if entity_type == "WEBSITE" else "email"
                )
            else:
                text = self._normalize_text(text)

            words.append({'text': text, 'bbox': bbox})

        # Split into words for matching
        match_words = normalized_matched_text.split()
        matching_bboxes = []
        current_match_bboxes = []

        for word in words:
            if match_words and word['text'] == match_words[0]:
                current_match_bboxes.append(word['bbox'])
                match_words.pop(0)
                if not match_words:  # Found all words in sequence
                    matching_bboxes.append(self._combine_bboxes(current_match_bboxes))
                    current_match_bboxes = []
            else:
                current_match_bboxes = []  # Reset on mismatch

        if not matching_bboxes and self.debug_mode:
            self.logger.debug(f"No BBox found for: {matched_text}")
            
        return matching_bboxes

    @staticmethod
    def _combine_bboxes(bboxes: list) -> Optional[fitz.Rect]:
        """
        Combine multiple bounding boxes into one unified rectangle.
        
        Args:
            bboxes: List of bounding boxes to combine
            
        Returns:
            Combined bounding box or None if no valid boxes
        """
        if not bboxes:
            return None
            
        valid_boxes = [bbox for bbox in bboxes if bbox]
        if not valid_boxes:
            return None
            
        x0 = min(bbox.x0 for bbox in valid_boxes)
        y0 = min(bbox.y0 for bbox in valid_boxes)
        x1 = max(bbox.x1 for bbox in valid_boxes)
        y1 = max(bbox.y1 for bbox in valid_boxes)
        
        return fitz.Rect(x0, y0, x1, y1)

    ###############################################################################
    # SECTION 6: PUBLIC INTERFACE
    ###############################################################################

    def process_pdf(self, 
                   input_path: str, 
                   output_path: str, 
                   redact_color: Tuple[float, float, float] = (0, 0, 0), 
                   redact_opacity: float = 1.0) -> Dict[str, int]:
        """
        Process a PDF file and apply redactions.

        Args:
            input_path: Path to input PDF
            output_path: Path to save redacted PDF
            redact_color: RGB color tuple for redactions
            redact_opacity: Opacity level for redactions (0.0 to 1.0)

        Returns:
            Dict with processing statistics
        """
        self.logger.info(f"Processing PDF: {input_path}")
        try:
            file_processor = RedactFileProcessor(
                redact_color=redact_color,
                redact_opacity=redact_opacity,
                ocr_enabled=self.ocr_enabled,
                validate_pdfs=self.validate_pdfs,
                logger=self.logger
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