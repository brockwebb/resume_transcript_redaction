# redactor/detectors/presidio_detector.py

from typing import List, Optional, Any, Dict, Set, Union
from pathlib import Path

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.recognizer_result import RecognizerResult
from presidio_analyzer.pattern_recognizer import PatternRecognizer

from .base_detector import BaseDetector, Entity
from .custom_recognizers import ConfigDrivenPhoneRecognizer, ConfigDrivenSsnRecognizer

class PresidioDetector(BaseDetector):
    """Detector implementation using Presidio's built-in recognizers."""
    
    def __init__(self,
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 language: str = "en"):
        """Initialize Presidio detector with configuration."""
        self.language = language
        self.analyzer = None
        self.entity_types = {}
        
        # Initialize base class to set up logger and config
        super().__init__(config_loader, logger)
        
        # Now do our initialization using the logger
        self._init_presidio()
        self._load_entity_types()

    def _load_entity_types(self) -> None:
        """Load entity types from configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                self.logger.error("Missing entity routing configuration")
                return

            presidio_config = routing_config["routing"].get("presidio_primary", {})
            base_threshold = presidio_config.get("confidence_threshold", 0.75)
            entities = presidio_config.get("entities", [])

            for entity in entities:
                threshold = presidio_config.get("thresholds", {}).get(entity, base_threshold)
                self.entity_types[entity] = threshold

            self.logger.debug("Loaded Presidio entity types with thresholds:")
            for entity, threshold in self.entity_types.items():
                self.logger.debug(f"  {entity}: {threshold}")

        except Exception as e:
            self.logger.error(f"Error loading entity types: {str(e)}")
            raise

    def _validate_config(self) -> bool:
        """Validate Presidio-specific configuration."""
        return True  # Initial validation always passes, we'll load config after

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer with default recognizers and our custom ones."""
        try:
            self.analyzer = AnalyzerEngine()
            
            # Remove default recognizers that we're replacing with custom ones
            self.analyzer.registry.remove_recognizer("PhoneRecognizer")
            self.analyzer.registry.remove_recognizer("UsSsnRecognizer")
            
            # Add our custom phone recognizer
            custom_phone = ConfigDrivenPhoneRecognizer(
                config_loader=self.config_loader,
                logger=self.logger
            )
            self.analyzer.registry.add_recognizer(custom_phone)
            
            # Add our custom SSN recognizer
            custom_ssn = ConfigDrivenSsnRecognizer(
                config_loader=self.config_loader,
                logger=self.logger
            )
            self.analyzer.registry.add_recognizer(custom_ssn)
            
            # Debug recognizer information
            recognizers = self.analyzer.get_recognizers()
            self.logger.debug(
                f"Initialized Presidio with recognizers: "
                f"{[r.name for r in recognizers]}"
            )
            
        except Exception as e:
            self.logger.error(f"Error initializing Presidio analyzer: {str(e)}")
            raise

    def detect_entities(self, text: str) -> List[Entity]:
        """Detect entities using Presidio's built-in recognizers."""
        if not text or not self.analyzer or not self.entity_types:
            return []

        try:
            self.logger.debug(f"Analyzing text with Presidio: {text[:100]}...")

            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=list(self.entity_types.keys())
            )

            self.logger.debug(
                f"Raw Presidio results before filtering:"
            )
            for result in results:
                self.logger.debug(
                    f"  Found: '{text[result.start:result.end]}' "
                    f"Type: {result.entity_type} "
                    f"Score: {result.score:.3f} "
                    f"Start: {result.start} End: {result.end} "
                    f"Recognition data: {result.recognition_metadata}"
                )

            entities = []
            for result in results:
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
                
                if self.validate_detection(entity):
                    entities.append(entity)
                    self.log_detection(entity)
                else:
                    self.logger.debug(
                        f"Entity failed validation: {entity.entity_type} "
                        f"({entity.confidence:.3f} < "
                        f"{self.entity_types.get(entity.entity_type, self.confidence_threshold)}) "
                        f"Text: '{entity.text}'"
                    )

            return entities

        except Exception as e:
            self.logger.error(f"Error during Presidio detection: {str(e)}")
            return []

    def validate_detection(self, entity: Entity) -> bool:
        """Validate detection against configured thresholds."""
        try:
            # Get entity-specific threshold
            threshold = self.entity_types.get(
                entity.entity_type, 
                self.confidence_threshold
            )
            
            # Check confidence threshold
            if entity.confidence < threshold:
                self.logger.debug(
                    f"Entity '{entity.text}' confidence below threshold: "
                    f"{entity.confidence:.2f} < {threshold}"
                )
                return False

            # Check for empty or whitespace-only text
            if not entity.text or not entity.text.strip():
                self.logger.debug(f"Entity has invalid text: '{entity.text}'")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False