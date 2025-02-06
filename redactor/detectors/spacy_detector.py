# redactor/detectors/spacy_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set
import spacy
import re
import json
from .base_detector import BaseDetector, Entity
from redactor.validation import ValidationCoordinator

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class SpacyDetector(BaseDetector):
    """
    Uses spaCy for context-aware named entity recognition.
    Focuses on linguistic analysis for entity detection.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize SpacyDetector with configuration and NLP model.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            confidence_threshold: Minimum confidence threshold
        """
        # Initialize base detector
        super().__init__(config_loader, logger, confidence_threshold)
        
        self.nlp = None
        self.entity_types = set()
        self.entity_mappings = {}
        self.confidence_thresholds = {}

        # Initialize core components
        self._validate_config()
        self._load_spacy_model()
        
        # Initialize validator
        self.validator = ValidationCoordinator(config_loader, logger)

    def _load_spacy_model(self) -> None:
        """Load the spaCy NLP model specified in configuration."""
        try:
            # Try to load large model first, fall back to small if needed
            model_name = self.config_loader.get_config("main").get(
                "language_model", 
                "en_core_web_lg"
            )
            try:
                if self.debug_mode:
                    self.logger.debug(f"Loading spaCy model: {model_name}")
                self.nlp = spacy.load(model_name)
            except OSError:
                if self.debug_mode:
                    self.logger.debug("Large model failed, falling back to en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            if self.debug_mode:
                self.logger.debug(f"SpaCy model {self.nlp.meta['name']} loaded successfully")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading SpaCy model: {e}")
            raise

    def _validate_config(self) -> bool:
        """
        Validate routing configuration for SpacyDetector.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                if self.logger:
                    self.logger.error("Missing routing configuration for SpacyDetector")
                return False
    
            spacy_config = routing_config["routing"].get("spacy_primary", {})
            
            # Validate required sections
            if not spacy_config:
                if self.logger:
                    self.logger.error("Missing 'spacy_primary' configuration")
                return False

            # Load entity types and mappings
            self.entity_types = set(spacy_config.get("entities", []))
            self.entity_mappings = spacy_config.get("mappings", {})
            
            # Get confidence thresholds
            self.confidence_thresholds = spacy_config.get("thresholds", {})

            if self.debug_mode:
                if self.logger:
                    self.logger.debug(f"Loaded spaCy entity types: {self.entity_types}")
                    self.logger.debug(f"Entity mappings: {self.entity_mappings}")
                    self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating spaCy config: {str(e)}")
            return False

###############################################################################
# SECTION 3: ENTITY DETECTION
###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities in the provided text using SpaCy NLP.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        if not text:
            return []

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    mapped_type = self.entity_mappings.get(ent.label_, ent.label_)
                    
                    # Apply type-specific threshold
                    confidence = self.confidence_thresholds.get(
                        ent.label_, 
                        self.confidence_threshold
                    )

                    entity = Entity(
                        text=ent.text,
                        entity_type=mapped_type,
                        confidence=confidence,
                        start=ent.start_char,
                        end=ent.end_char,
                        source="spacy",
                    )

                    entities.append(entity)
                    
                    if self.debug_mode and self.logger:
                        self.logger.debug(
                            f"spaCy entity: {entity.entity_type} "
                            f"'{entity.text}' (Confidence: {entity.confidence:.2f})"
                        )

            return entities

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error detecting entities with SpaCy: {e}")
            return []

###############################################################################
# SECTION 4: VALIDATION
###############################################################################

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate entity using ValidationCoordinator.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool indicating if entity is valid
        """
        result = self.validator.validate_entity(
            entity=entity,
            text=text,
            source="spacy"
        )
        
        if result.confidence_adjustment:
            entity.confidence += result.confidence_adjustment
            
        self.last_validation_reasons = result.reasons
        
        return result.is_valid