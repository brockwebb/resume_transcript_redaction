# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Optional, Any, Dict
from presidio_analyzer import AnalyzerEngine
from .base_detector import BaseDetector, Entity
from .validation_pattern_matcher import ValidationPatternMatcher
import re

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class PresidioDetector(BaseDetector):
    """
    Presidio-based detector that uses ValidationPatternMatcher for consistency.
    Combines Presidio's NLP capabilities with validated patterns from a single source.
    """
    
    def __init__(self, config_loader: Any, logger: Optional[Any] = None, language: str = "en"):
        """
        Initialize Presidio detector with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            language: Language code for Presidio analyzer
        """
        # Initialize base class
        super().__init__(config_loader, logger)
        
        self.language = language
        self.analyzer = None
        self.entity_types = {}
        self.last_validation_reasons = []
        
        # Initialize pattern matcher for consistent pattern handling
        self.pattern_matcher = ValidationPatternMatcher(config_loader, logger)
        
        # Initialize components
        self._init_presidio()
        self._load_entity_types()
        
        if self.debug_mode:
            self.logger.debug("PresidioDetector initialized with ValidationPatternMatcher")

###############################################################################
# SECTION 3: CONFIGURATION AND SETUP
###############################################################################

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer with basic configuration."""
        try:
            self.analyzer = AnalyzerEngine()
            
            # Use only core Presidio recognizers that don't overlap with our patterns
            for recognizer in [
                "PhoneRecognizer", 
                "EmailRecognizer", 
                "DateRecognizer",
                "UsSsnRecognizer", 
                "UrlRecognizer"
            ]:
                self.analyzer.registry.remove_recognizer(recognizer)
            
            if self.debug_mode:
                recognizers = self.analyzer.get_recognizers()
                self.logger.debug(
                    f"Initialized Presidio with core recognizers: "
                    f"{[r.name for r in recognizers]}"
                )
        except Exception as e:
            self.logger.error(f"Error initializing Presidio analyzer: {str(e)}")
            raise

    def _load_entity_types(self) -> None:
        """Load entity types and their thresholds from configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            ensemble_config = routing_config.get("routing", {}).get("ensemble_required", {})
            
            # Load standard entity types and thresholds
            base_threshold = presidio_config.get("confidence_threshold", 0.75)
            for entity in presidio_config.get("entities", []):
                threshold = presidio_config.get("thresholds", {}).get(entity, base_threshold)
                self.entity_types[entity] = threshold
            
            # Load ensemble settings
            self.ensemble_settings = {}
            for entity_type, config in ensemble_config.get("confidence_thresholds", {}).items():
                self.ensemble_settings[entity_type] = {
                    "minimum_combined": config.get("minimum_combined", 0.7),
                    "presidio_weight": config.get("presidio_weight", 0.65)
                }
            
            if self.debug_mode:
                self.logger.debug("Loaded entity configurations:")
                self.logger.debug(f"Entity types: {self.entity_types}")
                self.logger.debug(f"Ensemble settings: {self.ensemble_settings}")
                
        except Exception as e:
            self.logger.error(f"Error loading entity types: {str(e)}")
            raise

###############################################################################
    # SECTION 4: ENTITY DETECTION
    ###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities using both Presidio and ValidationPatternMatcher.
        
        Args:
            text: Text to analyze
            
        Returns:
            Combined list of detected entities
        """
        if not text or not self.analyzer or not self.entity_types:
            return []
            
        entities = []
        
        try:
            # Get pattern-based detections first
            pattern_entities = self.pattern_matcher.detect_entities(text)
            
            if self.debug_mode:
                self.logger.debug(
                    f"Found {len(pattern_entities)} entities using ValidationPatternMatcher"
                )
            
            # Get Presidio detections for remaining entity types
            presidio_results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=list(self.entity_types.keys())
            )
            
            if self.debug_mode:
                self.logger.debug(
                    f"Found {len(presidio_results)} entities using Presidio core recognizers"
                )
            
            # Convert Presidio results to our Entity format
            for result in presidio_results:
                entity = Entity(
                    text=text[result.start:result.end],
                    entity_type=result.entity_type,
                    confidence=result.score,
                    start=result.start,
                    end=result.end,
                    source="presidio",
                    metadata={
                        "recognition_metadata": result.recognition_metadata,
                        "analysis_explanation": result.analysis_explanation
                    }
                )
                
                # Validate using our consistent rules
                if self.validate_detection(entity, text):
                    entities.append(entity)
                    if self.debug_mode:
                        self.log_detection(entity)
                        
            # Combine results, preferring pattern matches for overlaps
            combined = self._combine_results(pattern_entities, entities)
            
            if self.debug_mode:
                self.logger.debug(
                    f"Final detection count: {len(combined)} "
                    f"(Pattern: {len(pattern_entities)}, Presidio: {len(entities)})"
                )
                
            return combined
            
        except Exception as e:
            self.logger.error(f"Error during entity detection: {str(e)}")
            return []

    ###############################################################################
    # SECTION 5: RESULT PROCESSING AND COMBINATION
    ###############################################################################

    def _combine_results(self, pattern_entities: List[Entity], 
                        presidio_entities: List[Entity]) -> List[Entity]:
        """
        Combine results from both detection methods, handling overlaps.
        Prefers pattern matches over Presidio matches for overlapping entities.
        
        Args:
            pattern_entities: Entities from ValidationPatternMatcher
            presidio_entities: Entities from Presidio
            
        Returns:
            Combined list of entities
        """
        if not pattern_entities:
            return presidio_entities
        if not presidio_entities:
            return pattern_entities
            
        try:
            # Index pattern entities by span
            pattern_spans = {(e.start, e.end): e for e in pattern_entities}
            
            # Add non-overlapping Presidio entities
            combined = list(pattern_entities)
            overlap_count = 0
            
            for presidio_ent in presidio_entities:
                span = (presidio_ent.start, presidio_ent.end)
                # Only add if no pattern match exists for this span
                if span not in pattern_spans:
                    combined.append(presidio_ent)
                else:
                    overlap_count += 1
                    
            if self.debug_mode and overlap_count > 0:
                self.logger.debug(
                    f"Resolved {overlap_count} overlaps, preferring pattern matches"
                )
                
            return sorted(combined, key=lambda e: (e.start, e.end))
            
        except Exception as e:
            self.logger.error(f"Error combining results: {str(e)}")
            # Return pattern entities as fallback in case of error
            return pattern_entities

###############################################################################
    # SECTION 6: ENTITY VALIDATION
    ###############################################################################

    def validate_detection(self, entity: Entity, text: str) -> bool:
        """
        Validate entity detection against configured thresholds.
        Uses ValidationPatternMatcher's validation rules for consistency.
        
        Args:
            entity: Entity to validate
            text: Full text context
            
        Returns:
            bool indicating if entity is valid
        """
        try:
            # Initialize validation tracking
            self.last_validation_reasons = []
            
            if self.debug_mode:
                self.logger.debug(
                    f"Validating {entity.entity_type} entity: '{entity.text}'"
                )
            
            # Check confidence threshold
            threshold = self.entity_types.get(entity.entity_type, self.confidence_threshold)
            if entity.confidence < threshold:
                reason = (
                    f"Entity '{entity.text}' confidence below threshold: "
                    f"{entity.confidence:.2f} < {threshold}"
                )
                if self.debug_mode:
                    self.logger.debug(reason)
                self.last_validation_reasons.append(reason)
                return False
            
            # Empty text check
            if not entity.text.strip():
                reason = f"Entity has invalid text: '{entity.text}'"
                if self.debug_mode:
                    self.logger.debug(reason)
                self.last_validation_reasons.append(reason)
                return False
            
            # Delegate to ValidationPatternMatcher's validation rules
            return self.pattern_matcher.validate_detection(entity, text)
            
        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False

    ###############################################################################
    # SECTION 7: CONFIG VALIDATION AND MANAGEMENT
    ###############################################################################

    def _validate_config(self) -> bool:
        """
        Validate the configuration for PresidioDetector.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                self.logger.error("Missing routing configuration for PresidioDetector")
                return False

            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            if not presidio_config:
                self.logger.error("Missing 'presidio_primary' configuration in routing")
                return False

            if "entities" not in presidio_config:
                self.logger.error("'entities' is missing in 'presidio_primary' configuration")
                return False

            if self.debug_mode:
                self.logger.debug("PresidioDetector configuration validated successfully")
            return True

        except Exception as e:
            self.logger.error(f"Error validating PresidioDetector configuration: {str(e)}")
            return False