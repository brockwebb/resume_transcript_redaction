# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Optional, Any, Dict
from presidio_analyzer import AnalyzerEngine
from .base_detector import BaseDetector, Entity
# Initialize validator
from redactor.validation import ValidationCoordinator


###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class PresidioDetector(BaseDetector):
    """
    Uses Presidio for enhanced NLP-based entity detection.
    Focuses on structured data patterns and detection.
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

        # Initialize Presidio components
        self._init_presidio()
        self._validate_config()
        self.validator = ValidationCoordinator(config_loader, logger)

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer with basic configuration."""
        try:
            self.analyzer = AnalyzerEngine()
            
            # Remove specific recognizers that might overlap with our patterns
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
            if self.logger:
                self.logger.error(f"Error initializing Presidio analyzer: {str(e)}")
            raise

    def _validate_config(self) -> bool:
        """
        Validate the configuration for PresidioDetector.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                if self.logger:
                    self.logger.error("Missing routing configuration for PresidioDetector")
                return False

            presidio_config = routing_config["routing"].get("presidio_primary", {})
            
            # Validate required sections
            if not presidio_config:
                if self.logger:
                    self.logger.error("Missing 'presidio_primary' configuration")
                return False

            # Load entity types and mappings
            self.entity_types = set(presidio_config.get("entities", []))
            self.entity_mappings = presidio_config.get("mappings", {})
            
            # Get confidence thresholds
            self.confidence_thresholds = presidio_config.get("thresholds", {})

            if self.debug_mode:
                if self.logger:
                    self.logger.debug(f"Loaded Presidio entity types: {self.entity_types}")
                    self.logger.debug(f"Entity mappings: {self.entity_mappings}")
                    self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating PresidioDetector configuration: {str(e)}")
            return False

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities using Presidio.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities
        """
        if not text or not self.analyzer or not self.entity_types:
            return []
            
        entities = []
        
        try:
            # Get Presidio detections
            presidio_results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=list(self.entity_types)
            )
            
            if self.debug_mode and self.logger:
                self.logger.debug(
                    f"Presidio detected {len(presidio_results)} potential entities"
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
                
                # Apply type-specific thresholds
                type_threshold = self.confidence_thresholds.get(
                    entity.entity_type, 
                    self.confidence_threshold
                )
                
                # Validate confidence against threshold
                if entity.confidence >= type_threshold:
                    entities.append(entity)
                    
                    if self.debug_mode and self.logger:
                        self.logger.debug(
                            f"Presidio entity: {entity.entity_type} "
                            f"'{entity.text}' (Confidence: {entity.confidence:.2f})"
                        )
            
            return entities
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during Presidio entity detection: {str(e)}")
            return []

###############################################################################
# SECTION 3: VALIDATION METHODS
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
        # Use validation coordinator
        result = self.validator.validate_entity(
            entity=entity,
            text=text,
            source="presidio"
        )
        
        # Update confidence and store reasons
        if result.confidence_adjustment:
            entity.confidence += result.confidence_adjustment
            
        self.last_validation_reasons = result.reasons
        
        return result.is_valid