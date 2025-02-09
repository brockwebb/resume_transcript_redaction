# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Optional, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern 
from .base_detector import BaseDetector, Entity
from redactor.validation import ValidationCoordinator

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class PresidioDetector(BaseDetector):
    """Uses Presidio for base entity detection."""
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """Initialize detector with configuration."""
        super().__init__(config_loader, logger, confidence_threshold)
        
        self.analyzer = None
        self.entity_types = set()
        self.confidence_thresholds = {}

        self._init_presidio()
        self._load_config()
        self.validator = ValidationCoordinator(config_loader, logger)

    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer."""
        try:
            self.analyzer = AnalyzerEngine()
            
            # Load validation params for patterns
            validation_params = self.config_loader.get_config("validation_params")
            if validation_params:
                # GPA custom recognizer (no native support)
                if "GPA" in validation_params:
                    gpa_config = validation_params["GPA"]
                    patterns = [
                        Pattern(
                            name="gpa_standard",
                            regex=gpa_config["common_formats"]["standard"],
                            score=0.85
                        ),
                        Pattern(
                            name="gpa_with_scale",
                            regex=gpa_config["common_formats"]["with_scale"],
                            score=0.9
                        )
                    ]
                    
                    gpa_recognizer = PatternRecognizer(
                        supported_entity="GPA",
                        patterns=patterns,
                        context=gpa_config.get("context_words", [])
                    )
                    
                    self.analyzer.registry.add_recognizer(gpa_recognizer)

                # PHONE_NUMBER patterns to enhance native detector
                if "PHONE_NUMBER" in validation_params:
                    phone_config = validation_params["PHONE_NUMBER"]
                    phone_patterns = []
                    for pattern_name, regex in phone_config["patterns"].items():
                        phone_patterns.append(
                            Pattern(
                                name=f"phone_{pattern_name}",
                                regex=regex,
                                score=0.85
                            )
                        )
                    
                    phone_recognizer = PatternRecognizer(
                        supported_entity="PHONE_NUMBER",
                        patterns=phone_patterns,
                        context=phone_config.get("context_words", [])
                    )
                    
                    self.analyzer.registry.add_recognizer(phone_recognizer)

                # ADDRESS patterns to enhance native detector
                if "ADDRESS" in validation_params:
                    addr_config = validation_params["ADDRESS"]
                    addr_patterns = []
                    
                    # Add patterns for full street types
                    for street_type in addr_config["street_types"]["full"]:
                        addr_patterns.append(
                            Pattern(
                                name=f"addr_full_{street_type}",
                                regex=f"\\b\\d+\\s+[A-Za-z\\s]+{street_type}\\b",
                                score=0.85
                            )
                        )
                    
                    # Add patterns for abbreviated street types
                    for abbrev in addr_config["street_types"]["abbrev"]:
                        addr_patterns.append(
                            Pattern(
                                name=f"addr_abbrev_{abbrev}",
                                regex=f"\\b\\d+\\s+[A-Za-z\\s]+{abbrev}\\b",
                                score=0.85
                            )
                        )

                    # Add apartment/unit patterns
                    for unit_type in addr_config["unit_types"]:
                        addr_patterns.append(
                            Pattern(
                                name=f"addr_unit_{unit_type}",
                                regex=f"\\b{unit_type}\\s*\\d+\\b",
                                score=0.85
                            )
                        )
                    
                    addr_recognizer = PatternRecognizer(
                        supported_entity="ADDRESS",
                        patterns=addr_patterns,
                        context=addr_config.get("context_words", [])
                    )
                    
                    self.analyzer.registry.add_recognizer(addr_recognizer)

                # OTHER_ID - use native recognizers + our patterns
                if "OTHER_ID" in validation_params:
                    id_config = validation_params["OTHER_ID"]
                    id_patterns = []
                    for pattern_name, pattern_info in id_config["patterns"].items():
                        id_patterns.append(
                            Pattern(
                                name=f"other_id_{pattern_name}",
                                regex=pattern_info["regex"],
                                score=pattern_info["confidence"]
                            )
                        )
                    
                    id_recognizer = PatternRecognizer(
                        supported_entity="OTHER_ID",
                        patterns=id_patterns,
                        context=id_config.get("context_words", [])
                    )
                    
                    self.analyzer.registry.add_recognizer(id_recognizer)
            
            # Keep all recognizers - let ValidationCoordinator handle the complex validation
            if self.debug_mode:
                recognizers = self.analyzer.get_recognizers()
                self.logger.debug(f"Loaded recognizers: {[r.name for r in recognizers]}")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error initializing Presidio: {str(e)}")
            raise

    
###############################################################################
# SECTION 3: CONFIGURATION MANAGEMENT
###############################################################################

    def _load_config(self) -> None:
        """Load basic configuration."""
        routing_config = self.config_loader.get_config("entity_routing")
        if routing_config and "routing" in routing_config:
            presidio_config = routing_config["routing"].get("presidio_primary", {})
            # Ensure DATE_TIME is included
            self.entity_types = set(presidio_config.get("entities", []))
            if "DATE_TIME" not in self.entity_types:
                self.entity_types.add("DATE_TIME")
            
            self.confidence_thresholds = presidio_config.get("thresholds", {})
            
            if self.debug_mode:
                self.logger.debug(f"Loaded entity types: {self.entity_types}")
                self.logger.debug(f"Loaded thresholds: {self.confidence_thresholds}")

###############################################################################
# SECTION 4: ENTITY DETECTION
###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """Detect entities using Presidio."""
        if not text or not self.analyzer:
            return []
            
        try:
            results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=list(self.entity_types)
            )
            
            entities = []
            for result in results:
                # Apply threshold and create entity
                threshold = self.confidence_thresholds.get(
                    result.entity_type, 
                    self.confidence_threshold
                )
                
                if result.score >= threshold:
                    entity = Entity(
                        text=text[result.start:result.end],
                        entity_type=result.entity_type,
                        confidence=result.score,
                        start=result.start,
                        end=result.end,
                        source="presidio"
                    )
                    entities.append(entity)
                    
                    if self.debug_mode:
                        self.logger.debug(
                            f"Detected {entity.entity_type}: '{entity.text}' "
                            f"(confidence: {entity.confidence:.2f})"
                        )
            
            return entities
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in detection: {str(e)}")
            return []

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate detected entity using ValidationCoordinator.
        Required implementation of abstract method from BaseDetector.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool indicating if entity is valid
        """
        result = self.validator.validate_entity(entity, text, source="presidio")
        return result.is_valid

###############################################################################
# END OF FILE
###############################################################################