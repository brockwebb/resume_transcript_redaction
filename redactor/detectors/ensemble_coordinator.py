# redactor/detectors/ensemble_coordinator.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set, Tuple
import logging
from pathlib import Path
from dataclasses import replace

from .base_detector import BaseDetector
from evaluation.models import Entity, adjust_entity_confidence
from .presidio_detector import PresidioDetector
from .spacy_detector import SpacyDetector
from redactor.validation import ValidationCoordinator



###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class EnsembleCoordinator(BaseDetector):
    """
    Coordinates multiple entity detection systems and combines their results.
    Uses routing configuration to determine which detector to use for each entity type.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize ensemble detection system.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            confidence_threshold: Minimum confidence threshold
        """
        # Initialize base detector
        super().__init__(config_loader, logger, confidence_threshold)
        
        # Initialize core detectors
        try:
            self.presidio_detector = PresidioDetector(
                config_loader=config_loader,
                logger=logger,
                confidence_threshold=confidence_threshold
            )
            if self.debug_mode:
                self.logger.debug("Initialized Presidio detector")

            self.spacy_detector = SpacyDetector(
                config_loader=config_loader,
                logger=logger,
                confidence_threshold=confidence_threshold
            )
            if self.debug_mode:
                self.logger.debug("Initialized spaCy detector")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing detectors: {str(e)}")
            raise

        # Cache routing configuration
        self.routing_config = self._cached_configs.get("entity_routing", {}).get("routing", {})
        if not self.routing_config:
            raise ValueError("Missing routing configuration")

        # Initialize routing maps
        self._init_routing_maps()

        # Initialize validator
        self.validator = ValidationCoordinator(config_loader, logger)
        
        if self.debug_mode:
            self.logger.debug("EnsembleCoordinator initialization complete")

    def _init_routing_maps(self) -> None:
        """Initialize maps for entity routing based on configuration."""
        try:
            # Get routing sections
            presidio_config = self.routing_config.get("presidio_primary", {})
            spacy_config = self.routing_config.get("spacy_primary", {})
            ensemble_config = self.routing_config.get("ensemble_required", {})

            # Map entity types to primary detectors
            self.presidio_entities = set(presidio_config.get("entities", []))
            self.spacy_entities = set(spacy_config.get("entities", []))
            self.ensemble_entities = set(ensemble_config.get("entities", []))

            # Get confidence thresholds
            self.presidio_thresholds = presidio_config.get("thresholds", {})
            self.spacy_thresholds = spacy_config.get("thresholds", {})
            self.ensemble_thresholds = ensemble_config.get("confidence_thresholds", {})

            if self.debug_mode:
                self.logger.debug(f"Presidio entities: {self.presidio_entities}")
                self.logger.debug(f"spaCy entities: {self.spacy_entities}")
                self.logger.debug(f"Ensemble entities: {self.ensemble_entities}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing routing maps: {str(e)}")
            raise

###############################################################################
# SECTION 3: ENTITY DETECTION AND ROUTING
###############################################################################

    def detect_entities(self, text: str, target_entity_type: Optional[str] = None) -> List[Entity]:
        """
        Detect entities with optional type filtering.
        
        Args:
            text: Text to analyze
            target_entity_type: Optional entity type to filter for
            
        Returns:
            List of detected entities
        """
        if not text:
            return []
            
        try:
            if self.debug_mode:
                self.logger.debug("Starting ensemble entity detection")
                
            entities = []
            
            # Early type filtering
            if target_entity_type:
                # For Presidio primary entities
                if target_entity_type in self.presidio_entities:
                    if self.debug_mode:
                        self.logger.debug(f"Detecting Presidio primary entity: {target_entity_type}")
                    detected = self.presidio_detector.detect_entities(text, target_entity_type)
                    # Validate each entity before adding
                    for entity in detected:
                        is_valid, updated_entity = self.validate_detection(entity, text)
                        if is_valid:
                            entities.append(updated_entity)
                    
                # For spaCy primary entities    
                if target_entity_type in self.spacy_entities:
                    if self.debug_mode:
                        self.logger.debug(f"Detecting spaCy primary entity: {target_entity_type}")
                    detected = self.spacy_detector.detect_entities(text, target_entity_type)
                    # Validate each entity before adding
                    for entity in detected:
                        is_valid, updated_entity = self.validate_detection(entity, text)
                        if is_valid:
                            entities.append(updated_entity)
                    
                # For ensemble required entities
                if target_entity_type in self.ensemble_entities:
                    if self.debug_mode:
                        self.logger.debug(f"Detecting ensemble entity: {target_entity_type}")
                    ensemble_entities = self._get_ensemble_entities(text, [target_entity_type])
                    # Ensemble entities are already validated in _detect_ensemble_entities
                    entities.extend(ensemble_entities)
                    
                return entities
                
            # If no target type, run full detection with validation
            presidio_entities = self.presidio_detector.detect_entities(text)
            validated_presidio = []
            for entity in presidio_entities:
                is_valid, updated_entity = self.validate_detection(entity, text)
                if is_valid:
                    validated_presidio.append(updated_entity)
                    
            if self.debug_mode:
                self.logger.debug(f"Found {len(validated_presidio)} validated Presidio primary entities")
                
            spacy_entities = self.spacy_detector.detect_entities(text)
            validated_spacy = []
            for entity in spacy_entities:
                is_valid, updated_entity = self.validate_detection(entity, text)
                if is_valid:
                    validated_spacy.append(updated_entity)
                    
            if self.debug_mode:
                self.logger.debug(f"Found {len(validated_spacy)} validated spaCy primary entities")
                
            ensemble_entities = self._get_ensemble_entities(text)
            
            # Combine all validated entities
            entities.extend(validated_presidio)
            entities.extend(validated_spacy)
            entities.extend(ensemble_entities)
            
            # Sort by position
            entities.sort(key=lambda x: (x.start_char, x.end_char))
            
            if self.debug_mode:
                self.logger.debug(f"Total entities detected and validated: {len(entities)}")
                
            return entities
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in ensemble detection: {str(e)}")
            return []
    
        
    def _get_presidio_entities(self, text: str) -> List[Entity]:
        """Get entities from Presidio for its primary types."""
        entities = []
        try:
            # Get all Presidio detections
            detected = self.presidio_detector.detect_entities(text)
            
            # Filter for Presidio primary types and validate
            for entity in detected:
                if entity.entity_type in self.presidio_entities:
                    if self.validate_detection(entity, text):
                        entities.append(entity)
                        
            if self.debug_mode:
                self.logger.debug(
                    f"Presidio detected {len(detected)} total, "
                    f"{len(entities)} validated"
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in Presidio detection: {str(e)}")
                
        return entities

    def _get_spacy_entities(self, text: str) -> List[Entity]:
        """Get entities from spaCy for its primary types."""
        entities = []
        try:
            # Get all spaCy detections
            detected = self.spacy_detector.detect_entities(text)
            
            # Filter for spaCy primary types and validate
            for entity in detected:
                if entity.entity_type in self.spacy_entities:
                    if self.validate_detection(entity, text):
                        entities.append(entity)
                        
            if self.debug_mode:
                self.logger.debug(
                    f"spaCy detected {len(detected)} total, "
                    f"{len(entities)} validated"
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in spaCy detection: {str(e)}")
                
        return entities

###############################################################################
# SECTION 4: ENSEMBLE PROCESSING
###############################################################################

    def _get_ensemble_entities(self, text: str, target_types: Optional[List[str]] = None) -> List[Entity]:
        """Get entities that require ensemble detection."""
        combined_entities = []
        
        try:
            # Get full results from both detectors
            presidio_results = self.presidio_detector.detect_entities(text)
            spacy_results = self.spacy_detector.detect_entities(text)
            
            # Process each ensemble entity type
            types_to_process = target_types if target_types else self.ensemble_entities
            for entity_type in types_to_process:
                if self.debug_mode:
                    self.logger.debug(f"Processing ensemble type: {entity_type}")
                    
                # Get matching entities of this type
                presidio_matches = [e for e in presidio_results if e.entity_type == entity_type]
                spacy_matches = [e for e in spacy_results if e.entity_type == entity_type]
                
                if self.debug_mode:
                    self.logger.debug(
                        f"Found {len(presidio_matches)} Presidio and "
                        f"{len(spacy_matches)} spaCy matches for {entity_type}"
                    )
                
                # Try to combine matching entities
                combined = self._combine_matching_entities(
                    presidio_matches,
                    spacy_matches,
                    entity_type
                )
                
                # Validate combined entities
                validated = []
                for entity in combined:
                    is_valid, updated_entity = self.validate_detection(entity, text)
                    if is_valid:
                        validated.append(updated_entity)
                
                combined_entities.extend(validated)
                
            if self.debug_mode:
                self.logger.debug(f"Combined {len(combined_entities)} ensemble entities")
    
            # Special handling for PHI and PROTECTED_CLASS entities
            if any(t in types_to_process for t in ["PHI", "PROTECTED_CLASS"]):
                # For PHI and PROTECTED_CLASS, we need special confidence handling
                special_types = [t for t in types_to_process if t in ("PHI", "PROTECTED_CLASS")]
                
                for entity_type in special_types:
                    if self.debug_mode:
                        self.logger.debug(f"Special processing for ensemble type: {entity_type}")
                        
                    # Get thresholds from config
                    type_thresholds = self.ensemble_thresholds.get(entity_type, {})
                    min_combined = type_thresholds.get("minimum_combined", 0.6)
                    presidio_weight = type_thresholds.get("presidio_weight", 0.5)
                    spacy_weight = type_thresholds.get("spacy_weight", 0.5)
                    
                    # Get entities of this type from each detector
                    presidio_detections = [e for e in presidio_results if e.entity_type == entity_type]
                    spacy_detections = [e for e in spacy_results if e.entity_type == entity_type]
                    
                    # For PROTECTED_CLASS, bypass validation and add spaCy entities directly
                    if entity_type == "PROTECTED_CLASS" and spacy_detections:
                        for s_entity in spacy_detections:
                            # Mark it as coming from ensemble to track source
                            enhanced_entity = replace(
                                s_entity,
                                detector_source="ensemble",
                                validation_rules=s_entity.validation_rules or []
                            )
                            combined_entities.append(enhanced_entity)
                            if self.debug_mode:
                                self.logger.debug(f"Added PROTECTED_CLASS entity directly: {s_entity.text}")
                    else:
                        # Process any solo detections from presidio
                        for p_entity in presidio_detections:
                            weighted_confidence = p_entity.confidence * presidio_weight
                            if weighted_confidence >= min_combined * 0.85:
                                enhanced_entity = Entity(
                                    text=p_entity.text,
                                    entity_type=p_entity.entity_type,
                                    start_char=p_entity.start_char,
                                    end_char=p_entity.end_char,
                                    confidence=weighted_confidence,
                                    detector_source="ensemble",
                                    page=None,
                                    sensitivity=None,
                                    validation_rules=[],
                                    original_confidence=weighted_confidence
                                )
                                is_valid, updated_entity = self.validate_detection(enhanced_entity, text)
                                if is_valid:
                                    combined_entities.append(updated_entity)
                        
                        # Process any solo detections from spacy
                        for s_entity in spacy_detections:
                            weighted_confidence = s_entity.confidence * spacy_weight
                            if weighted_confidence >= min_combined * 0.85:
                                enhanced_entity = Entity(
                                    text=s_entity.text,
                                    entity_type=s_entity.entity_type,
                                    start_char=s_entity.start_char,
                                    end_char=s_entity.end_char,
                                    confidence=weighted_confidence,
                                    detector_source="ensemble",
                                    page=None,
                                    sensitivity=None,
                                    validation_rules=[],
                                    original_confidence=weighted_confidence
                                )
                                is_valid, updated_entity = self.validate_detection(enhanced_entity, text)
                                if is_valid:
                                    combined_entities.append(updated_entity)
            
            return combined_entities
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in ensemble processing: {str(e)}")
            # Return what we have instead of an empty list if there's an error
            return combined_entities if combined_entities else []


    def _combine_matching_entities(
        self,
        presidio_entities: List[Entity],
        spacy_entities: List[Entity],
        entity_type: str
    ) -> List[Entity]:
        """Combine matching entities from different detectors."""
        combined = []
        
        # Get confidence weights
        thresholds = self.ensemble_thresholds.get(entity_type, {})
        presidio_weight = thresholds.get("presidio_weight", 0.5)
        spacy_weight = thresholds.get("spacy_weight", 0.5)
        
        # Find matching entities
        for p_entity in presidio_entities:
            matches = [
                s for s in spacy_entities
                if self._check_entity_match(p_entity, s)
            ]
            
            for s_entity in matches:
                # Calculate combined confidence         return max(0.0, min(1.0, value))
                combined_confidence = max(0.0, min(1.0,
                    p_entity.confidence * presidio_weight +
                    s_entity.confidence * spacy_weight
                ))
                  
                # Create combined entity using standardized fields
                combined_entity = Entity(
                    text=p_entity.text,
                    entity_type=p_entity.entity_type,
                    start_char=p_entity.start_char,
                    end_char=p_entity.end_char,
                    confidence=max(0.0, min(1.0,combined_confidence)),
                    detector_source="ensemble",
                    page=None,
                    sensitivity=None,
                    validation_rules=[],
                    original_confidence=combined_confidence
                )
                combined.append(combined_entity)
                    
        return combined

    def _check_entity_match(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities match based on text and type."""
        if entity1.entity_type != entity2.entity_type:
            return False
            
        # Compare normalized text
        text1 = self._normalize_text(entity1.text)
        text2 = self._normalize_text(entity2.text)
        
        # Check position overlap using start_char/end_char
        pos_overlap = (
            entity1.start_char <= entity2.end_char and 
            entity2.start_char <= entity1.end_char
        )
        
        return text1 == text2 and pos_overlap

###############################################################################
# SECTION 5: VALIDATION AND UTILITY METHODS
###############################################################################

    def validate_detection(self, entity: Entity, text: str = "") -> Tuple[bool, Entity]:
        """
        Validate entity using ValidationCoordinator.
        Returns validation status and possibly updated entity.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            Tuple of (validation success, possibly updated entity)
        """
        try:
            result, updated_entity = self.validator.validate_entity(
                entity=entity,
                text=text,
                source="ensemble"
            )
            
            return result.is_valid, updated_entity
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Validation error in Ensemble detection: {str(e)}")
            return False, entity

    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
            
        # Basic normalization - lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove punctuation except preserved characters
        normalized = ''.join(
            c for c in normalized 
            if c.isalnum() or c in '.-_ '
        )
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized

    def _log_entity_debug(self, entity: Entity, source: str) -> None:
        """Log detailed entity information for debugging."""
        if not self.debug_mode:
            return
            
        self.logger.debug(
            f"{source} Entity: {entity.entity_type}\n"
            f"  Text: '{entity.text}'\n"
            f"  Position: {entity.start_char}-{entity.end_char}\n"
            f"  Confidence: {entity.confidence:.2f}"
        )
        
        if hasattr(entity, 'validation_rules') and entity.validation_rules:
            self.logger.debug(f"  Validation Rules: {entity.validation_rules}")


    def _get_detection_stats(self) -> Dict[str, Dict[str, int]]:
        """Get detection statistics by detector and entity type."""
        stats = {
            "presidio": {"total": 0},
            "spacy": {"total": 0},
            "ensemble": {"total": 0}
        }
        
        # Update for each detector's primary types
        for entity_type in self.presidio_entities:
            stats["presidio"][entity_type] = 0
            
        for entity_type in self.spacy_entities:
            stats["spacy"][entity_type] = 0
            
        for entity_type in self.ensemble_entities:
            stats["ensemble"][entity_type] = 0
            
        return stats