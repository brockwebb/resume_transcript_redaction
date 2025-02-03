# redactor/detectors/ensemble_coordinator.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set, Tuple
import logging
from pathlib import Path

from .base_detector import BaseDetector, Entity
from .presidio_detector import PresidioDetector
from .spacy_detector import SpacyDetector

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

    def _validate_config(self) -> bool:
        """
        Validate ensemble configuration.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            routing_config = self._cached_configs.get("entity_routing", {}).get("routing", {})
            
            if not routing_config:
                if self.logger:
                    self.logger.error("Missing routing configuration")
                return False

            # Verify required sections exist
            required_sections = ["presidio_primary", "spacy_primary", "ensemble_required"]
            for section in required_sections:
                if section not in routing_config:
                    if self.logger:
                        self.logger.error(f"Missing required routing section: {section}")
                    return False

                # Verify each section has required fields
                section_config = routing_config[section]
                if "entities" not in section_config:
                    if self.logger:
                        self.logger.error(f"Missing entities list in {section}")
                    return False

                # Verify thresholds for ensemble section
                if section == "ensemble_required":
                    if "confidence_thresholds" not in section_config:
                        if self.logger:
                            self.logger.error("Missing confidence thresholds in ensemble config")
                        return False
                    
                    # Verify threshold structure for each ensemble entity
                    for entity_type in section_config["entities"]:
                        thresholds = section_config["confidence_thresholds"].get(entity_type, {})
                        if not all(k in thresholds for k in ["minimum_combined", "presidio_weight", "spacy_weight"]):
                            if self.logger:
                                self.logger.error(f"Invalid threshold config for {entity_type}")
                            return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating config: {str(e)}")
            return False

###############################################################################
# SECTION 3: ENTITY DETECTION AND ROUTING
###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities using configured detection strategy.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        if not text:
            return []
            
        try:
            if self.debug_mode:
                self.logger.debug("Starting ensemble entity detection")
                
            all_entities = []
            
            # Get Presidio primary detections
            presidio_entities = self._get_presidio_entities(text)
            if self.debug_mode:
                self.logger.debug(f"Found {len(presidio_entities)} Presidio primary entities")
            
            # Get spaCy primary detections
            spacy_entities = self._get_spacy_entities(text)
            if self.debug_mode:
                self.logger.debug(f"Found {len(spacy_entities)} spaCy primary entities")
            
            # Get ensemble detections that require both
            ensemble_entities = self._get_ensemble_entities(text)
            if self.debug_mode:
                self.logger.debug(f"Found {len(ensemble_entities)} ensemble entities")
            
            # Combine all detections
            all_entities.extend(presidio_entities)
            all_entities.extend(spacy_entities)
            all_entities.extend(ensemble_entities)
            
            # Sort by position
            all_entities.sort(key=lambda x: (x.start, x.end))
            
            if self.debug_mode:
                self.logger.debug(f"Total entities detected: {len(all_entities)}")
                
            return all_entities
            
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
            
            # Filter for Presidio primary types and check thresholds
            for entity in detected:
                if entity.entity_type in self.presidio_entities:
                    threshold = self.presidio_thresholds.get(
                        entity.entity_type, 
                        self.confidence_threshold
                    )
                    if entity.confidence >= threshold:
                        entities.append(entity)
                        
            if self.debug_mode:
                self.logger.debug(
                    f"Presidio detected {len(detected)} total, "
                    f"{len(entities)} passed thresholds"
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
            
            # Filter for spaCy primary types and check thresholds
            for entity in detected:
                if entity.entity_type in self.spacy_entities:
                    threshold = self.spacy_thresholds.get(
                        entity.entity_type, 
                        self.confidence_threshold
                    )
                    if entity.confidence >= threshold:
                        entities.append(entity)
                        
            if self.debug_mode:
                self.logger.debug(
                    f"spaCy detected {len(detected)} total, "
                    f"{len(entities)} passed thresholds"
                )
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in spaCy detection: {str(e)}")
                
        return entities

###############################################################################
# SECTION 4: ENSEMBLE PROCESSING AND COMBINATION
###############################################################################

    def _get_ensemble_entities(self, text: str) -> List[Entity]:
        """
        Get entities that require ensemble detection.
        Combines and validates results from both detectors.
        """
        combined_entities = []
        
        try:
            # Get full results from both detectors
            presidio_results = self.presidio_detector.detect_entities(text)
            spacy_results = self.spacy_detector.detect_entities(text)
            
            # Process each ensemble entity type
            for entity_type in self.ensemble_entities:
                if self.debug_mode:
                    self.logger.debug(f"Processing ensemble type: {entity_type}")
                    
                # Get confidence thresholds for this type
                thresholds = self.ensemble_thresholds.get(entity_type, {})
                if not thresholds:
                    if self.debug_mode:
                        self.logger.debug(f"No threshold config for {entity_type}")
                    continue
                    
                # Get matching entities of this type
                presidio_matches = [e for e in presidio_results if e.entity_type == entity_type]
                spacy_matches = [e for e in spacy_results if e.entity_type == entity_type]
                
                if self.debug_mode:
                    self.logger.debug(
                        f"Found {len(presidio_matches)} Presidio and "
                        f"{len(spacy_matches)} spaCy matches for {entity_type}"
                    )
                
                # Try to combine matching entities
                combined = self._combine_entity_detections(
                    presidio_matches,
                    spacy_matches,
                    thresholds
                )
                
                combined_entities.extend(combined)
                
            if self.debug_mode:
                self.logger.debug(f"Combined {len(combined_entities)} ensemble entities")
                
            return combined_entities
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in ensemble processing: {str(e)}")
            return []

    def _combine_entity_detections(
        self,
        presidio_entities: List[Entity],
        spacy_entities: List[Entity],
        thresholds: Dict[str, float]
    ) -> List[Entity]:
        """
        Combine matching entities from different detectors.
        
        Args:
            presidio_entities: Entities from Presidio
            spacy_entities: Entities from spaCy
            thresholds: Confidence thresholds for combination
            
        Returns:
            List of combined entities that meet threshold requirements
        """
        combined = []
        
        # Get required thresholds
        min_combined = thresholds.get("minimum_combined", 0.8)
        presidio_weight = thresholds.get("presidio_weight", 0.5)
        spacy_weight = thresholds.get("spacy_weight", 0.5)
        
        # Find matching entities
        for p_entity in presidio_entities:
            matches = [
                s for s in spacy_entities
                if self._check_entity_match(p_entity, s)
            ]
            
            for s_entity in matches:
                # Calculate combined confidence
                combined_confidence = (
                    p_entity.confidence * presidio_weight +
                    s_entity.confidence * spacy_weight
                )
                
                if combined_confidence >= min_combined:
                    # Create combined entity using Presidio's span
                    combined_entity = Entity(
                        text=p_entity.text,
                        entity_type=p_entity.entity_type,
                        confidence=combined_confidence,
                        start=p_entity.start,
                        end=p_entity.end,
                        source="ensemble",
                        metadata={
                            "presidio_confidence": p_entity.confidence,
                            "spacy_confidence": s_entity.confidence,
                            "combined_confidence": combined_confidence
                        }
                    )
                    combined.append(combined_entity)
                    
        return combined

    def _check_entity_match(self, entity1: Entity, entity2: Entity) -> bool:
        """
        Check if two entities match based on text and type.
        
        Args:
            entity1: First entity to compare
            entity2: Second entity to compare
            
        Returns:
            bool indicating if entities match
        """
        # Must be same entity type
        if entity1.entity_type != entity2.entity_type:
            return False
            
        # Compare normalized text
        text1 = self._normalize_entity_text(entity1.text)
        text2 = self._normalize_entity_text(entity2.text)
        
        return text1 == text2

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text string
        """
        if not text:
            return ""
            
        # Basic normalization - lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove all punctuation except .-_ and space
        normalized = ''.join(
            c for c in normalized 
            if c.isalnum() or c in '.-_ '
        )
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized

###############################################################################
# SECTION 5: UTILITY METHODS
###############################################################################

    def _get_entity_confidence(self, entity: Entity, entity_type: str) -> float:
        """
        Get appropriate confidence threshold for entity type.
        
        Args:
            entity: Entity to check
            entity_type: Type of entity
            
        Returns:
            float: Confidence threshold
        """
        if entity_type in self.presidio_entities:
            return self.presidio_thresholds.get(entity_type, self.confidence_threshold)
        elif entity_type in self.spacy_entities:
            return self.spacy_thresholds.get(entity_type, self.confidence_threshold)
        else:
            thresholds = self.ensemble_thresholds.get(entity_type, {})
            return thresholds.get("minimum_combined", self.confidence_threshold)

    def _log_entity_debug(self, entity: Entity, source: str) -> None:
        """
        Log detailed entity information for debugging.
        
        Args:
            entity: Entity to log
            source: Source of the entity detection
        """
        if not self.debug_mode:
            return
            
        self.logger.debug(
            f"{source} Entity: {entity.entity_type}\n"
            f"  Text: '{entity.text}'\n"
            f"  Position: {entity.start}-{entity.end}\n"
            f"  Confidence: {entity.confidence:.2f}"
        )
        
        if entity.metadata:
            self.logger.debug(f"  Metadata: {entity.metadata}")

    def _get_detection_stats(self) -> Dict[str, Dict[str, int]]:
        """
        Get detection statistics by detector and entity type.
        
        Returns:
            Dict containing detection statistics
        """
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

###############################################################################
# SECTION 6: VALIDATION AND POST-PROCESSING
###############################################################################

    def _validate_entity(self, entity: Entity) -> bool:
        """
        Validate entity based on its type and source.
        
        Args:
            entity: Entity to validate
            
        Returns:
            bool: Whether entity is valid
        """
        try:
            # Get appropriate threshold
            threshold = self._get_entity_confidence(entity, entity.entity_type)
            
            # Check confidence threshold
            if entity.confidence < threshold:
                if self.debug_mode:
                    self.logger.debug(
                        f"Entity failed confidence check: {entity.confidence} < {threshold}"
                    )
                return False
            
            # Validate by source
            if entity.source == "presidio":
                return self.presidio_detector.validate_detection(entity)
            elif entity.source == "spacy":
                return self.spacy_detector.validate_detection(entity)
            elif entity.source == "ensemble":
                # For ensemble entities, check both detectors
                return (self.presidio_detector.validate_detection(entity) and 
                       self.spacy_detector.validate_detection(entity))
            else:
                if self.debug_mode:
                    self.logger.debug(f"Unknown entity source: {entity.source}")
                return False
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating entity: {str(e)}")
            return False

    def _post_process_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Apply post-processing to detected entities.
        
        Args:
            entities: List of entities to process
            
        Returns:
            List of processed entities
        """
        if not entities:
            return []
            
        try:
            processed = []
            
            # Sort by position
            entities.sort(key=lambda x: (x.start, x.end))
            
            # Remove duplicates and overlaps
            current = entities[0]
            
            for next_entity in entities[1:]:
                # Check for overlap
                if current.end <= next_entity.start:
                    # No overlap - add current and move to next
                    if self._validate_entity(current):
                        processed.append(current)
                    current = next_entity
                else:
                    # Overlap - keep entity with higher confidence
                    if next_entity.confidence > current.confidence:
                        current = next_entity
            
            # Add final entity
            if self._validate_entity(current):
                processed.append(current)
                
            if self.debug_mode:
                self.logger.debug(
                    f"Post-processing: {len(entities)} input, {len(processed)} output"
                )
                
            return processed
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in post-processing: {str(e)}")
            return entities  # Return original list if processing fails