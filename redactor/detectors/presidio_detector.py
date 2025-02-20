# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Optional, Any
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern 
from .base_detector import BaseDetector
from evaluation.models import Entity, adjust_entity_confidence
from redactor.validation import ValidationCoordinator
from dataclasses import replace


###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################
    
# redactor/detectors/presidio_detector.py

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################
    
class PresidioDetector(BaseDetector):
    """Uses Presidio for pattern-based entity recognition."""
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """Initialize PresidioDetector with configuration."""
        super().__init__(config_loader, logger, confidence_threshold)
        
        self.analyzer = None
        self.entity_types = set()
        self.entity_mappings = {}
        self.confidence_thresholds = {}
        
        # Initialize core components
        self._validate_config()
        self._init_analyzer()
        if self.debug_mode:
            self.logger.debug("Initialized Presidio detector")

    def _validate_config(self) -> bool:
        """Validate routing configuration for PresidioDetector."""
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
    
            # Load entity types from routing configuration
            self.entity_types = set(presidio_config.get("entities", []))
            
            # Initialize entity mappings (1:1 for Presidio)
            self.entity_mappings = {
                entity_type: entity_type 
                for entity_type in self.entity_types
            }
            
            # Get confidence thresholds
            self.confidence_thresholds = presidio_config.get("thresholds", {})
    
            if self.debug_mode:
                self.logger.debug(f"Loaded entity types: {self.entity_types}")
                self.logger.debug(f"Loaded thresholds: {self.confidence_thresholds}")
    
            return True
    
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating Presidio config: {str(e)}")
            return False

    def _init_analyzer(self) -> None:
        """Initialize Presidio analyzer with custom recognizers."""
        try:
            from presidio_analyzer import AnalyzerEngine
            
            # Initialize analyzer
            self.analyzer = AnalyzerEngine()
            
            # Load custom recognizers
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

 

                # INTERNET_REFERENCE patterns
                if "INTERNET_REFERENCE" in validation_params:
                    web_config = validation_params["INTERNET_REFERENCE"]
                    web_patterns = []
                    
                    if "validation_types" in web_config:
                        # URL patterns
                        url_config = web_config["validation_types"].get("url", {})
                        if "patterns" in url_config:
                            for pattern_name, pattern in url_config["patterns"].items():
                                web_patterns.append(
                                    Pattern(
                                        name=f"url_{pattern_name}",
                                        regex=pattern,
                                        score=0.85
                                    )
                                )

                        # Social handle patterns  
                        social_config = web_config["validation_types"].get("social_handle", {})
                        if "patterns" in social_config:
                            for pattern_name, pattern in social_config["patterns"].items():
                                web_patterns.append(
                                    Pattern(
                                        name=f"social_{pattern_name}", 
                                        regex=pattern,
                                        score=0.85
                                    )
                                )

                    # Get context words from known platforms
                    context_words = []
                    if "known_platforms" in web_config:
                        for platform_list in web_config["known_platforms"].values():
                            context_words.extend([p.split('.')[0] for p in platform_list])

                    web_recognizer = PatternRecognizer(
                        supported_entity="INTERNET_REFERENCE",
                        patterns=web_patterns,
                        context=context_words
                    )
                    self.analyzer.registry.add_recognizer(web_recognizer)

                    if self.debug_mode:
                        self.logger.debug(f"Added INTERNET_REFERENCE recognizer with {len(web_patterns)} patterns")

                
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

                if "EDUCATIONAL_INSTITUTION" in validation_params:
                    edu_config = validation_params["EDUCATIONAL_INSTITUTION"]
                    patterns = []
                    
                    # Create pattern for each core term
                    for term in edu_config.get("core_terms", []):
                        patterns.append(
                            Pattern(
                                name=f"edu_{term}",
                                regex=f"\\b[A-Z][A-Za-z\\s]*{term}\\b",
                                score=0.85
                            )
                        )
                    
                    # Add pattern for known acronyms
                    acronyms = "|".join(edu_config.get("known_acronyms", []))
                    if acronyms:
                        patterns.append(
                            Pattern(
                                name="edu_acronym",
                                regex=f"\\b({acronyms})\\b",
                                score=0.9
                            )
                        )
                    
                    edu_recognizer = PatternRecognizer(
                        supported_entity="EDUCATIONAL_INSTITUTION",
                        patterns=patterns,
                        context=edu_config.get("core_terms", [])
                    )
                    
                    self.analyzer.registry.add_recognizer(edu_recognizer)


                # PHI (Protected Health Information) recognizer
                if "PHI" in validation_params:
                    phi_config = validation_params["PHI"]
                    phi_patterns = []
                    
                    # Add patterns for specific medical conditions
                    if "patterns" in phi_config:
                        specific_patterns = phi_config["patterns"].get("specific_conditions", {}).get("regex", [])
                        for i, pattern in enumerate(specific_patterns):
                            phi_patterns.append(
                                Pattern(
                                    name=f"phi_specific_{i}",
                                    regex=pattern,
                                    score=0.85
                                )
                            )
                    
                    # Add medication patterns
                    if "patterns" in phi_config and "medication_patterns" in phi_config["patterns"]:
                        med_patterns = phi_config["patterns"]["medication_patterns"].get("regex", [])
                        for i, pattern in enumerate(med_patterns):
                            phi_patterns.append(
                                Pattern(
                                    name=f"phi_medication_{i}",
                                    regex=pattern,
                                    score=0.80
                                )
                            )
                    
                    # Add treatment/procedure patterns
                    if "patterns" in phi_config and "procedure_context" in phi_config["patterns"]:
                        proc_patterns = phi_config["patterns"]["procedure_context"].get("regex", [])
                        for i, pattern in enumerate(proc_patterns):
                            phi_patterns.append(
                                Pattern(
                                    name=f"phi_procedure_{i}",
                                    regex=pattern,
                                    score=0.80
                                )
                            )
                    
                    # Add medical context patterns
                    if "patterns" in phi_config and "medical_context_patterns" in phi_config["patterns"]:
                        context_patterns = phi_config["patterns"]["medical_context_patterns"].get("regex", [])
                        for i, pattern in enumerate(context_patterns):
                            phi_patterns.append(
                                Pattern(
                                    name=f"phi_context_{i}",
                                    regex=pattern,
                                    score=0.70
                                )
                            )
                    
                    # Add VA/disability patterns
                    va_patterns = [
                        r"\b(?:\d{1,3}%\s+(?:service[\s-]*connected|VA|disabled|disability))",
                        r"\b(?:service[\s-]*connected\s+(?:disability|condition|rating)\s+(?:of\s+)?\d{1,3}%)",
                        r"\b(?:VA\s+(?:disability|rating)\s+(?:of\s+)?\d{1,3}%)",
                        r"\b(?:permanent\s+and\s+total\s+disability)"
                    ]
                    for i, pattern in enumerate(va_patterns):
                        phi_patterns.append(
                            Pattern(
                                name=f"phi_va_{i}",
                                regex=pattern,
                                score=0.85
                            )
                        )
                    
                    # Create context from categories and common terms
                    context_words = phi_config.get("context_words", [])
                    if "categories" in phi_config:
                        # Add medical conditions
                        for condition_list in phi_config["categories"].get("medical_conditions", {}).values():
                            context_words.extend(condition_list)
                        
                        # Add treatments
                        for treatment_list in phi_config["categories"].get("treatments", {}).values():
                            context_words.extend(treatment_list)
                    
                    phi_recognizer = PatternRecognizer(
                        supported_entity="PHI",
                        patterns=phi_patterns,
                        context=context_words
                    )
                    
                    self.analyzer.registry.add_recognizer(phi_recognizer)
                    if self.debug_mode:
                        self.logger.debug(f"Added PHI recognizer with {len(phi_patterns)} patterns")

                # PROTECTED_CLASS recognizer
                if "PROTECTED_CLASS" in validation_params:
                    pc_config = validation_params["PROTECTED_CLASS"]
                    pc_patterns = []
                    
                    # Gender patterns
                    if "categories" in pc_config and "gender_patterns" in pc_config["categories"]:
                        gender_conf = pc_config["categories"]["gender_patterns"]
                        
                        # Organization membership patterns
                        if "org_membership" in gender_conf:
                            pc_patterns.append(
                                Pattern(
                                    name="pc_gender_org",
                                    regex=gender_conf["org_membership"],
                                    score=0.85
                                )
                            )
                        
                        # Identity regex
                        if "identity_regex" in gender_conf:
                            pc_patterns.append(
                                Pattern(
                                    name="pc_gender_identity",
                                    regex=gender_conf["identity_regex"],
                                    score=0.80
                                )
                            )
                    
                    # Religious patterns
                    if "categories" in pc_config and "religious_patterns" in pc_config["categories"]:
                        rel_conf = pc_config["categories"]["religious_patterns"]
                        
                        # Organization membership patterns
                        if "org_membership" in rel_conf:
                            pc_patterns.append(
                                Pattern(
                                    name="pc_religious_org",
                                    regex=rel_conf["org_membership"],
                                    score=0.85
                                )
                            )
                        
                        # Identity regex
                        if "identity_regex" in rel_conf:
                            pc_patterns.append(
                                Pattern(
                                    name="pc_religious_identity",
                                    regex=rel_conf["identity_regex"],
                                    score=0.80
                                )
                            )
                    
                    # Pronoun patterns
                    if "categories" in pc_config and "pronoun_patterns" in pc_config["categories"]:
                        pronoun_conf = pc_config["categories"]["pronoun_patterns"]
                        for pattern_name, pattern in pronoun_conf.items():
                            pc_patterns.append(
                                Pattern(
                                    name=f"pc_pronoun_{pattern_name}",
                                    regex=pattern,
                                    score=0.90
                                )
                            )
                    
                    # Build context words
                    context_words = []
                    
                    # Add gender terms
                    if "categories" in pc_config and "gender_patterns" in pc_config["categories"]:
                        gender_terms = pc_config["categories"]["gender_patterns"].get("gender_terms", [])
                        context_words.extend(gender_terms)
                        
                        # Add identity context
                        context_words.extend(pc_config["categories"]["gender_patterns"].get("identity_context", []))
                        context_words.extend(pc_config["categories"]["gender_patterns"].get("org_context", []))
                    
                    # Add religious terms
                    if "categories" in pc_config and "religious_patterns" in pc_config["categories"]:
                        religious_terms = pc_config["categories"]["religious_patterns"].get("religious_terms", [])
                        context_words.extend(religious_terms)
                        
                        # Add identity context
                        context_words.extend(pc_config["categories"]["religious_patterns"].get("identity_context", []))
                        context_words.extend(pc_config["categories"]["religious_patterns"].get("org_context", []))
                    
                    # Add race/ethnicity terms
                    if "categories" in pc_config and "race_ethnicity" in pc_config["categories"]:
                        context_words.extend(pc_config["categories"]["race_ethnicity"])
                    
                    # Add orientation terms
                    if "categories" in pc_config and "orientation" in pc_config["categories"]:
                        context_words.extend(pc_config["categories"]["orientation"])
                    
                    # Remove duplicates while preserving order
                    context_words = list(dict.fromkeys(context_words))
                    
                    pc_recognizer = PatternRecognizer(
                        supported_entity="PROTECTED_CLASS",
                        patterns=pc_patterns,
                        context=context_words
                    )
                    
                    self.analyzer.registry.add_recognizer(pc_recognizer)
                    if self.debug_mode:
                        self.logger.debug(f"Added PROTECTED_CLASS recognizer with {len(pc_patterns)} patterns")



            
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
    
        
    def detect_entities(self, text: str, target_entity_type: Optional[str] = None) -> List[Entity]:
        """Detect entities using Presidio analyzer."""
        if not text:
            return []
            
        try:
            # If target type specified, only detect that type
            entity_types = {target_entity_type} if target_entity_type else self.entity_types
            
            analyzer_results = self.analyzer.analyze(
                text=text,
                language='en',
                entities=entity_types,
                correlation_id=None
            )
            
            entities = []
            for result in analyzer_results:
                # Create initial entity
                mapped_type = self.entity_mappings.get(result.entity_type, result.entity_type)
                
                # Skip if we have a target type and this doesn't match
                if target_entity_type and mapped_type != target_entity_type:
                    continue
                    
                entity = Entity(
                    text=text[result.start:result.end],
                    entity_type=mapped_type,
                    start_char=result.start,
                    end_char=result.end,
                    confidence=max(0.0, min(1.0, result.score)),
                    detector_source="presidio",
                    page=None,
                    sensitivity=None,
                    validation_rules=[],
                    original_confidence=result.score
                )
                
                entities.append(entity)
                
                if self.debug_mode:
                    self.logger.debug(
                        f"Presidio entity: {entity.entity_type} "
                        f"'{entity.text}' (Confidence: {entity.confidence:.2f})"
                    )
                    
            return entities
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in detection: {str(e)}")
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
                source="presidio"
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
                self.logger.error(f"Validation error in Presidio detection: {str(e)}")
            return False

###############################################################################
# END OF FILE
###############################################################################