# redactor/detectors/spacy_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set
import spacy
import re
import json
from .base_detector import BaseDetector
from evaluation.models import Entity, adjust_entity_confidence
from redactor.validation import ValidationCoordinator
from dataclasses import replace


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
    
            # Dynamically load entity types from routing configuration
            self.entity_types = set(spacy_config.get("entities", []))
            
            # Default mapping and fallback to routing configuration
            self.entity_mappings = spacy_config.get("mappings", {
                # Person mappings
                "PERSON": "PERSON",
                
                # Date mappings
                "DATE": "DATE_TIME",
                "TIME": "DATE_TIME",
                
                # Educational institution mappings
                "ORG": "EDUCATIONAL_INSTITUTION",  # Universities, colleges
                
                # PHI/Medical mappings - enhanced
                "DISEASES": "PHI",        # Medical conditions
                "PROBLEM": "PHI",         # Medical issues/conditions
                "TREATMENT": "PHI",       # Medical procedures/treatments
                "CONDITION": "PHI",       # Medical conditions 
                "DISORDER": "PHI",        # Medical disorders
                "SYMPTOM": "PHI",         # Medical symptoms
                "HOSPITAL": "PHI",        # Medical facilities
                "MEDICALRECORD": "PHI",   # Medical records
                "MEDICATION": "PHI",      # Medications
                "PROCEDURE": "PHI",       # Medical procedures
                "DIAGNOSIS": "PHI",       # Medical diagnoses
                
                # Protected class mappings - enhanced
                "NORP": "PROTECTED_CLASS",      # Nationalities, religious, political groups
                "RELIGION": "PROTECTED_CLASS",  # Religious groups
                "NATIONALITY": "PROTECTED_CLASS", # National origin
                "ETHNICITY": "PROTECTED_CLASS",   # Ethnic groups
                "RACE": "PROTECTED_CLASS",        # Racial identifiers
                "GENDER": "PROTECTED_CLASS",      # Gender identities
                "ORIENTATION": "PROTECTED_CLASS", # Sexual orientation
                "IDEOLOGY": "PROTECTED_CLASS",    # Political/religious ideology
                "BELIEF": "PROTECTED_CLASS",      # Belief systems
            })
            
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

    def _process_custom_entities(self, doc, text: str) -> List[Entity]:
        """
        Process custom entity types that need special handling beyond spaCy's built-in NER.
        
        Args:
            doc: Processed spaCy Doc object
            text: Original text string
            
        Returns:
            List of additional detected entities
        """
        custom_entities = []
        
        # PHI custom patterns
        phi_patterns = [
            (r"\b\d{1,3}%\s+(?:service[\s-]*connected|VA|disabled|disability)", "PHI"),
            (r"\bservice[\s-]*connected\s+(?:disability|condition|rating)\s+(?:of\s+)?\d{1,3}%\b", "PHI"),
            (r"\bVA\s+(?:disability|rating)\s+(?:of\s+)?\d{1,3}%\b", "PHI"),
            (r"\b(?:Type\s*[12]\s*diabetes|insulin\s+dependent)\b", "PHI"),
            (r"\b(?:permanent\s+and\s+total|P&T)\s+disability\b", "PHI"),
            (r"\b(?:reasonable\s+accommodation|medical\s+leave|ADA\s+request)\b", "PHI")
        ]
        
        # Protected class custom patterns
        pc_patterns = [
            (r"\bprounou?ns[:;]\s*(?:he/him|she/her|they/them)", "PROTECTED_CLASS"),
            (r"\b(?:gender|sexual)\s+(?:identity|orientation)\b", "PROTECTED_CLASS"),
            (r"\b(?:Women|Men|Gender|Trans|Nonbinary)(?:'s)?\s+(?:Association|Society|Network|Group|Organization)\b", "PROTECTED_CLASS"),
            (r"\b(?:Muslim|Jewish|Christian|Buddhist|Hindu|Sikh)\s+(?:Student|Community|Association)\b", "PROTECTED_CLASS"),
            (r"\b(?:LGBTQ(?:\+|IA)?|lesbian|gay|bisexual|transgender|queer)\b", "PROTECTED_CLASS"),
            (r"\b(?:African\s*American|Latino|Hispanic|Asian\s*American|Pacific\s*Islander|Native\s*American)\b", "PROTECTED_CLASS")
        ]
        
        # Process all patterns
        all_patterns = phi_patterns + pc_patterns
        for pattern, entity_type in all_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                # Calculate character positions
                start_char = match.start()
                end_char = match.end()
                matched_text = match.group()
                
                # Create entity with appropriate confidence
                base_confidence = self.confidence_thresholds.get(
                    entity_type, 
                    self.confidence_threshold
                )
                
                entity = Entity(
                    text=matched_text,
                    entity_type=entity_type,
                    start_char=start_char,
                    end_char=end_char,
                    confidence=base_confidence,
                    detector_source="spacy",
                    page=None,
                    sensitivity=None,
                    validation_rules=[],
                    original_confidence=base_confidence
                )
                
                custom_entities.append(entity)
                
                if self.debug_mode and self.logger:
                    self.logger.debug(
                        f"Custom entity: {entity_type} "
                        f"'{matched_text}' (Confidence: {base_confidence:.2f})"
                    )
                    
        return custom_entities

    def detect_entities(self, text: str, target_entity_type: Optional[str] = None) -> List[Entity]:
        """Detect entities in the provided text using SpaCy NLP."""
        if not text:
            return []
    
        try:
            doc = self.nlp(text)
            entities = []
    
            for ent in doc.ents:
                if (ent.label_ in self.entity_types or 
                    ent.label_ in self.entity_mappings):
                    
                    mapped_type = self.entity_mappings.get(
                        ent.label_, 
                        ent.label_
                    )
                    
                    # Skip if we have a target type and this doesn't match
                    if target_entity_type and mapped_type != target_entity_type:
                        continue
                    
                    # Just use the threshold directly - don't clamp
                    base_confidence = self.confidence_thresholds.get(
                        mapped_type, 
                        self.confidence_threshold
                    )
    
                    entity = Entity(
                        text=ent.text,
                        entity_type=mapped_type,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                        confidence=base_confidence,  # Let validation adjust this freely
                        detector_source="spacy",
                        page=None,
                        sensitivity=None,
                        validation_rules=[],
                        original_confidence=base_confidence
                    )
    
                    entities.append(entity)
                    
                    if self.debug_mode and self.logger:
                        self.logger.debug(
                            f"spaCy entity: {entity.entity_type} "
                            f"'{entity.text}' (Confidence: {entity.confidence:.2f})"
                        )
            
            # Add custom entity processing
            custom_entities = self._process_custom_entities(doc, text)
            entities.extend(custom_entities)
    
            return entities
    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error detecting entities with SpaCy: {e}")
            return []

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate entity using ValidationCoordinator.
        Lets confidence adjustments happen freely during validation.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool indicating if entity is valid
        """
        try:
            result = self.validator.validate_entity(
                entity=entity,
                text=text,
                source="spacy"
            )
            
            # Apply confidence adjustment without clamping
            if result.confidence_adjustment:
                entity = adjust_entity_confidence(
                    entity, 
                    result.confidence_adjustment, 
                    logger=self.logger
                )
            
            # Track validation info
            self.last_validation_reasons = result.reasons
            if hasattr(result, 'metadata') and result.metadata.get('applied_rules'):
                entity = replace(entity, 
                    validation_rules=list(entity.validation_rules or []) + 
                    result.metadata['applied_rules']
                )
            
            return True  # Let validation happen, filter confidences later
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Validation error in SpaCy detection: {str(e)}")
            return False