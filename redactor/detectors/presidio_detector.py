# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Optional, Any, Dict
from presidio_analyzer import AnalyzerEngine
from .base_detector import BaseDetector, Entity
#from .validation_pattern_matcher import ValidationPatternMatcher
from redactor.validation.validation_rules import EntityValidationRules
import re

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class PresidioDetector(BaseDetector):
    """
    Uses Presidio for enhanced NLP-based entity detection.
    Focuses on structured data patterns and validation rules.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize Presidio detector with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            confidence_threshold: Minimum confidence threshold
        """
        # Initialize base detector first
        super().__init__(config_loader, logger, confidence_threshold)
        
        # Initialize core attributes
        self.analyzer = None
        self.entity_types = set()
        self.entity_mappings = {}
        self.confidence_thresholds = {}

        # Initialize validation rules
        try:
            self.validation_rules = EntityValidationRules(logger)
            if self.debug_mode:
                self.logger.debug("Validation rules initialized successfully")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to initialize validation rules: {e}. "
                    "Will continue with default validation."
                )
            self.validation_rules = None

        # Initialize Presidio components
        self._init_presidio()
        self._validate_config()

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer and components."""
        try:
            from presidio_analyzer import AnalyzerEngine
            self.analyzer = AnalyzerEngine()
            
            if self.debug_mode:
                self.logger.debug("Initialized Presidio analyzer")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize Presidio analyzer: {e}")
            raise

    def _validate_config(self) -> bool:
        """
        Validate Presidio configuration and setup entity mappings.
        
        Returns:
            bool: Whether configuration is valid
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                if self.logger:
                    self.logger.error("Missing routing configuration for PresidioDetector")
                return False
    
            presidio_config = routing_config["routing"].get("presidio_primary", {})
            ensemble_config = routing_config["routing"].get("ensemble_required", {})
            
            # Load basic entities from presidio_primary
            self.entity_types = set(presidio_config.get("entities", []))
            self.entity_mappings = presidio_config.get("mappings", {})
            
            # Get base thresholds
            base_thresholds = presidio_config.get("thresholds", {})
            ensemble_thresholds = ensemble_config.get("confidence_thresholds", {})
            
            # Initialize and populate thresholds
            self.confidence_thresholds = {}
            
            # Add base thresholds
            for entity, threshold in base_thresholds.items():
                self.confidence_thresholds[entity] = threshold
                
            # Add ensemble thresholds - using minimum_combined as base
            for entity, config in ensemble_thresholds.items():
                if entity in self.entity_types:
                    self.confidence_thresholds[entity] = config.get("minimum_combined", self.confidence_threshold)
    
            if self.debug_mode:
                self.logger.debug(f"Loaded Presidio entity types: {self.entity_types}")
                self.logger.debug(f"Entity mappings: {self.entity_mappings}")
                self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")
    
            return True
    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating Presidio config: {str(e)}")
            return False

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
        Uses EntityValidationRules for validation.
        
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
            
            # Use EntityValidationRules for validation
            if self.validation_rules:
                validation_result = self.validation_rules.validate_entity(entity, text)
                
                if self.debug_mode:
                    self.logger.debug(
                        f"Validation result for {entity.entity_type}: "
                        f"Valid: {validation_result.is_valid}, "
                        f"Reason: {validation_result.reason}"
                    )
                
                return validation_result.is_valid
            
            # Fallback if no validation rules
            return True
            
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