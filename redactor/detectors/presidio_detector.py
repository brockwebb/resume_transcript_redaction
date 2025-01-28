from typing import List, Optional, Any, Dict
from presidio_analyzer import AnalyzerEngine
from .base_detector import BaseDetector, Entity
from .custom_recognizers import (
    ConfigDrivenPhoneRecognizer,
    ConfigDrivenSsnRecognizer,
    ConfigDrivenAddressRecognizer,
    ConfigDrivenEducationalInstitutionRecognizer,
    ConfigDrivenGpaRecognizer,
    ConfigDrivenLocationRecognizer,
)
import re
from redactor.validation.validation_rules import EntityValidationRules 


class PresidioDetector(BaseDetector):
    def __init__(self, config_loader: Any, logger: Optional[Any] = None, language: str = "en"):
        """Initialize Presidio detector with configuration."""
        self.last_validation_reasons = []
        self.language = language
        self.analyzer = None
        self.entity_types = {}
        self.ignore_date_words = []
        self.ignore_phrases = []
        
        # Initialize base class
        super().__init__(config_loader, logger)
    
        # Add validation rules initialization
        try:
            from redactor.validation.validation_rules import EntityValidationRules
            self.validation_rules = EntityValidationRules(logger)
            if logger:
                logger.debug("Validation rules initialized successfully")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to initialize validation rules: {e}. Will continue with default validation.")
            self.validation_rules = None
    
        # Initialize Presidio analyzer and load entity types
        self._init_presidio()
        self._load_entity_types()
    
        # Load configurations
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            presidio_primary = routing_config.get("routing", {}).get("presidio_primary", {})
            validation_config = presidio_primary.get("validation", {})
            
            # General configurations
            self.ignore_phrases = validation_config.get("ignore_phrases", [])
            self.ignore_date_words = validation_config.get("date_ignore_words", [])
            
            # Social media patterns from core_patterns.yaml
            core_patterns = self.config_loader.get_config("core_patterns")
            self.social_media_patterns = core_patterns.get("social_media_patterns", [])
            self.social_context_words = core_patterns.get("context_words", {}).get("social_media", [])
            
            self.logger.debug(f"Loaded ignore phrases: {self.ignore_phrases}")
            self.logger.debug(f"Loaded date ignore words: {self.ignore_date_words}")
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {str(e)}")

            
    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer with default and custom recognizers."""
        try:
            self.analyzer = AnalyzerEngine()

            # Remove default recognizers that are replaced by custom ones
            self.analyzer.registry.remove_recognizer("PhoneRecognizer")
            self.analyzer.registry.remove_recognizer("UsSsnRecognizer")

            # Add custom recognizers
            self.analyzer.registry.add_recognizer(ConfigDrivenAddressRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenLocationRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenPhoneRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenEducationalInstitutionRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenGpaRecognizer(self.config_loader, self.logger))
            
            self.analyzer.registry.add_recognizer(ConfigDrivenSsnRecognizer(self.config_loader, self.logger))

            recognizers = self.analyzer.get_recognizers()
            self.logger.debug(f"Initialized Presidio with recognizers: {[r.name for r in recognizers]}")
        except Exception as e:
            self.logger.error(f"Error initializing Presidio analyzer: {str(e)}")
            raise

    def _load_entity_types(self) -> None:
        """Load entity types and their thresholds from configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            ensemble_config = routing_config.get("routing", {}).get("ensemble_required", {})
            base_threshold = presidio_config.get("confidence_threshold", 0.75)
            entities = presidio_config.get("entities", [])
    
            # Load standard entity types and thresholds
            for entity in entities:
                threshold = presidio_config.get("thresholds", {}).get(entity, base_threshold)
                self.entity_types[entity] = threshold
    
            # Load ensemble-specific configurations
            self.ensemble_settings = {}
            for entity_type, config in ensemble_config.get("confidence_thresholds", {}).items():
                self.ensemble_settings[entity_type] = {
                    "minimum_combined": config.get("minimum_combined", 0.7),
                    "presidio_weight": config.get("presidio_weight", 0.65)
                }
    
            # Get date-specific ensemble settings if available
            date_ensemble_config = ensemble_config.get("confidence_thresholds", {}).get("DATE_TIME", {})
            if date_ensemble_config:
                self.date_confidence_settings = {
                    "minimum_combined": date_ensemble_config.get("minimum_combined", 0.7),
                    "presidio_weight": date_ensemble_config.get("presidio_weight", 0.65)
                }
                self.logger.debug(f"Loaded date ensemble settings: {self.date_confidence_settings}")
    
            self.logger.debug("Loaded Presidio entity types with thresholds:")
            for entity, threshold in self.entity_types.items():
                self.logger.debug(f"  {entity}: {threshold}")
                
            if self.ensemble_settings:
                self.logger.debug("Loaded ensemble configurations:")
                for entity, settings in self.ensemble_settings.items():
                    self.logger.debug(f"  {entity}: {settings}")
    
        except Exception as e:
            self.logger.error(f"Error loading entity types: {str(e)}")
            raise
    def _validate_config(self) -> bool:
        """Validate the configuration for PresidioDetector."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                self.logger.error("Missing routing configuration for PresidioDetector.")
                return False

            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            if not presidio_config:
                self.logger.error("Missing 'presidio_primary' configuration in routing.")
                return False

            if "entities" not in presidio_config:
                self.logger.error("'entities' is missing in 'presidio_primary' configuration.")
                return False

            self.logger.debug("PresidioDetector configuration validated successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error validating PresidioDetector configuration: {str(e)}")
            return False

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
    
            self.logger.debug("Raw Presidio results before filtering:")
            for result in results:
                # Add extra debug info for dates
                if result.entity_type == "DATE_TIME":
                    self.logger.debug(
                        f"Found date: '{text[result.start:result.end]}' "
                        f"Score: {result.score:.3f} "
                        f"Recognition metadata: {result.recognition_metadata}"
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
            
                # Date-specific handling
                if entity.entity_type == "DATE_TIME":
                    self.logger.debug(
                        f"Processing DATE_TIME entity: '{entity.text}' "
                        f"Initial confidence: {entity.confidence}"
                    )
                    # Adjust initial confidence for ensemble weighting
                    entity.metadata["original_confidence"] = entity.confidence
                    
                    # Check recognition metadata for specific date formats
                    if result.recognition_metadata:
                        if "pattern_name" in result.recognition_metadata:
                            entity.metadata["date_pattern"] = result.recognition_metadata["pattern_name"]
                    
                    # Adjust confidence based on ensemble settings
                    if hasattr(self, 'date_confidence_settings'):
                        presidio_weight = self.date_confidence_settings["presidio_weight"]
                        entity.confidence *= presidio_weight
                        entity.metadata["weighted_confidence"] = entity.confidence
                        self.logger.debug(
                            f"Adjusted date confidence with weight {presidio_weight}: {entity.confidence}"
                        )            
               

                # Validate detection
                if self.validate_detection(entity, text):
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

    def validate_detection(self, entity: Entity, full_text: str) -> bool:
        """Validate detection against thresholds and ignore phrases."""
        try:
            # Initialize validation tracking
            self.last_validation_reasons = []
            self.logger.debug(f"Starting validation for {entity.entity_type} entity: '{entity.text}'")
    
            # Base validation checks
            if self._should_ignore(entity.text, full_text, entity.entity_type):
                reason = f"Entity '{entity.text}' ignored due to ignore phrases."
                self.logger.debug(reason)
                self.last_validation_reasons.append(reason)
                return False
    
            threshold = self.entity_types.get(entity.entity_type, self.confidence_threshold)
            if entity.confidence < threshold:
                reason = f"Entity '{entity.text}' confidence below threshold: {entity.confidence:.2f} < {threshold}"
                self.logger.debug(reason)
                self.last_validation_reasons.append(reason)
                return False
    
            if not entity.text.strip():
                reason = f"Entity has invalid text: '{entity.text}'"
                self.logger.debug(reason)
                self.last_validation_reasons.append(reason)
                return False
    
            # Date-specific validation
            if entity.entity_type == "DATE_TIME":
                initial_confidence = entity.confidence
                if hasattr(self, 'validation_rules') and self.validation_rules:
                    try:
                        self.logger.debug(f"Applying date validation rules to: {entity.text}")
                        result = self.validation_rules.validate_date(entity, full_text)
                        if result.is_valid:
                            entity.confidence += result.confidence_adjustment
                            self.logger.debug(
                                f"Date validation passed. Confidence adjusted from {initial_confidence} "
                                f"to {entity.confidence} (adj: {result.confidence_adjustment})"
                            )
                            self.last_validation_reasons.append(result.reason)
                            return True
                        else:
                            self.logger.debug(f"Date validation failed: {result.reason}")
                            self.last_validation_reasons.append(result.reason)
                            return False
                    except Exception as e:
                        self.logger.warning(f"Date validation error, using fallback: {e}")
                        # Fall through to basic date validation
                
                # Basic date validation if rules fail
                return self._validate_basic_date(entity, full_text)
    
            # Address-specific validation
            elif entity.entity_type == "ADDRESS":
                initial_confidence = entity.confidence
                self.logger.debug(f"Applying address validation rules to: {entity.text}")
                is_valid = self._validate_address(entity, full_text)
                if not is_valid:
                    return False
                self.logger.debug(
                    f"Address validation passed. Confidence adjusted from {initial_confidence} "
                    f"to {entity.confidence}"
                )
    
            # GPA-specific validation
            elif entity.entity_type == "GPA":
                initial_confidence = entity.confidence
                self.logger.debug(f"Applying GPA validation rules to: {entity.text}")
                is_valid = self._validate_gpa(entity, full_text)
                if not is_valid:
                    self.logger.debug(f"GPA validation failed: {'; '.join(self.last_validation_reasons)}")
                    return False
                self.logger.debug(
                    f"GPA validation passed. Confidence adjusted from {initial_confidence} "
                    f"to {entity.confidence}"
                )
    
            # Phone-specific validation
            elif entity.entity_type == "PHONE_NUMBER":
                initial_confidence = entity.confidence
                self.logger.debug(f"Applying phone validation rules to: {entity.text}")
                is_valid = self._validate_phone(entity, full_text)
                if not is_valid:
                    self.logger.debug(f"Phone validation failed: {'; '.join(self.last_validation_reasons)}")
                    return False
                self.logger.debug(
                    f"Phone validation passed. Confidence adjusted from {initial_confidence} "
                    f"to {entity.confidence}"
                )

            # Email address validation
            elif entity.entity_type == "EMAIL_ADDRESS":
                initial_confidence = entity.confidence
                self.logger.debug(f"Applying email validation rules to: {entity.text}")
                is_valid = self._validate_email_address(entity, full_text)
                if not is_valid:
                    self.logger.debug(f"Email validation failed: {'; '.join(self.last_validation_reasons)}")
                    return False
                self.logger.debug(
                    f"Email validation passed. Confidence adjusted from {initial_confidence} "
                    f"to {entity.confidence}"
                )

            # Internet reference validation
            elif entity.entity_type == "INTERNET_REFERENCE":
                initial_confidence = entity.confidence
                self.logger.debug(f"Applying internet reference validation rules to: {entity.text}")
                is_valid = self._validate_internet_reference(entity, full_text)
                if not is_valid:
                    self.logger.debug(f"Internet reference validation failed: {'; '.join(self.last_validation_reasons)}")
                    return False
                self.logger.debug(
                    f"Internet reference validation passed. Confidence adjusted from {initial_confidence} "
                    f"to {entity.confidence}"
                )
            
            # All validations passed
            return True
    
        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False
    
        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False
    
    def _validate_basic_date(self, entity: Entity, text: str) -> bool:
        """
        Basic date validation when advanced rules are unavailable.
        Used as fallback for date validation with extra resume-specific checks.
        """
        try:
            # First check format
            if not self._validate_date_format(entity):
                self.last_validation_reasons.append("Failed date format validation")
                return False
    
            # Check for false positive contexts
            context_lower = text.lower()
            false_positive_terms = set([
                'hours', 'minutes', 'seconds', 'course', 'version', 
                'volume', 'vol', 'score', 'grade'
            ])
            
            # Check surrounding context (within reasonable window)
            start_idx = max(0, entity.start - 50)
            end_idx = min(len(text), entity.end + 50)
            context_window = text[start_idx:end_idx].lower()
    
            if any(term in context_window for term in false_positive_terms):
                self.last_validation_reasons.append("Date appears in false positive context")
                return False
    
            # Apply confidence boosts
            confidence_boost = self._get_date_confidence_boost(entity)
            if confidence_boost > 0:
                entity.confidence += confidence_boost
                self.last_validation_reasons.append(f"Applied confidence boost: +{confidence_boost}")
    
            # Final validation passed
            return True
    
        except Exception as e:
            self.logger.error(f"Error in basic date validation: {e}")
            return False

    def _validate_address(self, entity: Entity, text: str) -> bool:
            """
            Validate ADDRESS entities using configuration-driven rules.
            Must be called from validate_detection method.
            """
            if not entity.text:
                return False
    
            address_config = self.config_loader.get_config("validation_params").get("address", {})
            if not address_config:
                self.logger.error("Missing address validation configuration")
                return False
    
            text_lower = entity.text.lower()
            reasons = []
    
            # Load configuration
            street_types = set(
                t.lower() for t in 
                address_config.get("street_types", {}).get("full", []) +
                address_config.get("street_types", {}).get("abbrev", [])
            )
            unit_types = set(t.lower() for t in address_config.get("unit_types", []))
            directionals = set(d.lower() for d in address_config.get("directionals", []))
            validation_rules = address_config.get("validation_rules", {})
            confidence_boosts = address_config.get("confidence_boosts", {})
            confidence_penalties = address_config.get("confidence_penalties", {})
    
            import re
    
            # Basic validation
            if len(entity.text) < validation_rules.get("min_chars", 5):
                reasons.append("Address too short")
                self.last_validation_reasons = reasons
                return False
    
            # Check for required number if specified
            if validation_rules.get("require_number", True):
                if not any(c.isdigit() for c in entity.text):
                    penalty = confidence_penalties.get("no_number", -0.5)
                    entity.confidence += penalty
                    reasons.append(f"Missing street number: {penalty}")
                    self.last_validation_reasons = reasons
                    return False
    
            # Check for street type using word boundaries
            has_street_type = False
            for st in street_types:
                # Look for street type as a word, handling potential comma or end of string
                if re.search(rf'\b{st}\b(?:[,\s]|$)', text_lower):
                    has_street_type = True
                    break
            
            if validation_rules.get("require_street_type", True) and not has_street_type:
                penalty = confidence_penalties.get("no_street_type", -0.3)
                entity.confidence += penalty
                reasons.append(f"Missing street type: {penalty}")
                self.last_validation_reasons = reasons
                return False
            else:
                boost = confidence_boosts.get("street_match", 0.2)
                entity.confidence += boost
                reasons.append(f"Valid street type: +{boost}")
    
            # Check for unit/suite numbers using word boundaries
            has_unit = False
            for unit in unit_types:
                if re.search(rf'\b{unit}\b', text_lower):
                    has_unit = True
                    break
            
            if has_unit:
                boost = confidence_boosts.get("has_unit", 0.1)
                entity.confidence += boost
                reasons.append(f"Has unit/suite: +{boost}")
    
            # Check for directionals using word boundaries
            has_directional = False
            for dir in directionals:
                if re.search(rf'\b{dir}\b', text_lower):
                    has_directional = True
                    break
    
            if has_directional:
                boost = confidence_boosts.get("directional", 0.1)
                entity.confidence += boost
                reasons.append(f"Has directional: +{boost}")
    
            # Additional boost for having complete address structure
            if len(reasons) >= 2:  # Has at least two valid components
                boost = confidence_boosts.get("full_address", 0.2)
                entity.confidence += boost
                reasons.append(f"Complete address structure: +{boost}")
    
            # Store validation reasons and debug log
            self.last_validation_reasons = reasons
            self.logger.debug(f"Address validation results for '{entity.text}': {reasons}")
            return True    

    def _validate_gpa(self, entity: Entity, text: str) -> bool:
            """
            Validate GPA entities using configuration-driven rules.
            Must be called from validate_detection method.
            """
            if not entity.text:
                return False
    
            # Get GPA validation parameters
            gpa_params = self.config_loader.get_config("validation_params").get("gpa", {})
            if not gpa_params:
                self.logger.error("Missing GPA validation configuration")
                return False
    
            validation_rules = gpa_params.get("validation_rules", {})
            confidence_boosts = gpa_params.get("confidence_boosts", {})
            confidence_penalties = gpa_params.get("confidence_penalties", {})
            text_lower = entity.text.lower()
            reasons = []
    
            import re
    
            # Check for GPA indicators first
            has_indicator = any(indicator.lower() in text_lower 
                              for indicator in gpa_params.get("indicators", []))
            if not has_indicator and validation_rules.get("require_indicator", True):
                reasons.append("Missing GPA indicator")
                self.last_validation_reasons = reasons
                return False
    
            # Extract numeric value - Updated regex to handle negative numbers
            numeric_match = re.search(r'(-?\d+\.?\d*)', text_lower)
            if not numeric_match:
                reasons.append("No valid GPA value found")
                self.last_validation_reasons = reasons
                return False
    
            try:
                gpa_value = float(numeric_match.group(1))
            except ValueError:
                reasons.append("Could not parse GPA value")
                self.last_validation_reasons = reasons
                return False
    
            # Validate value range - Do this early to fail fast on invalid values
            min_value = validation_rules.get("min_value", 0.0)
            max_value = validation_rules.get("max_value", 4.0)
            
            self.logger.debug(f"Validating GPA value: {gpa_value} against range [{min_value}, {max_value}]")
            
            if gpa_value < min_value:
                reasons.append(f"GPA value {gpa_value} below minimum {min_value}")
                self.last_validation_reasons = reasons
                return False
                
            if gpa_value > max_value:
                reasons.append(f"GPA value {gpa_value} exceeds maximum {max_value}")
                self.last_validation_reasons = reasons
                return False
    
            # Now that we know we have a valid GPA, add indicator boost
            if has_indicator:
                boost = confidence_boosts.get("has_indicator", 0.2)
                entity.confidence += boost
                reasons.append(f"Found GPA indicator: +{boost}")
    
            # Check for proper value range (add boost if within ideal range)
            if 2.0 <= gpa_value <= max_value:
                boost = confidence_boosts.get("proper_value_range", 0.2)
                entity.confidence += boost
                reasons.append(f"Valid GPA range: +{boost}")
    
            # Check for scale indicator (e.g., "/4.0")
            has_scale = any(scale_ind in text_lower 
                           for scale_ind in gpa_params.get("scale_indicators", []))
            if has_scale:
                boost = confidence_boosts.get("has_scale", 0.1)
                entity.confidence += boost
                reasons.append(f"Found scale indicator: +{boost}")
    
            # Check for standard format
            format_matched = False
            for pattern_name, pattern in gpa_params.get("common_formats", {}).items():
                if re.search(pattern, text_lower):
                    format_matched = True
                    boost = confidence_boosts.get("standard_format", 0.1)
                    entity.confidence += boost
                    reasons.append(f"Matches {pattern_name} format: +{boost}")
                    break
    
            # Check decimal places
            if '.' in str(gpa_value):
                decimal_places = len(str(gpa_value).split('.')[-1])
                if decimal_places > validation_rules.get("decimal_places", 2):
                    reasons.append(f"Too many decimal places: {decimal_places}")
                    self.last_validation_reasons = reasons
                    return False
    
            # Store validation reasons and debug log
            self.last_validation_reasons = reasons
            self.logger.debug(f"GPA validation results for '{entity.text}': {reasons}")
            return True

    def _validate_phone(self, entity: Entity, text: str) -> bool:
        """
        Validate phone number entities using configuration-driven rules.
        Must be called from validate_detection method.
    
        This version:
          - Removes common phone context prefixes (e.g., "Phone:", "Tel:", "mobile:")
            before matching the actual number.
          - Awards a separate context boost if we detect 'Phone:' or other prefix.
          - Uses digit-count checks, anchored regex patterns, and extension logic.
        """
        import re
    
        if not entity.text:
            return False
    
        # Load configurations
        core_patterns = self.config_loader.get_config("core_patterns")
        phone_patterns = core_patterns.get("phone_patterns", [])
    
        # Get phone validation parameters
        phone_params = self.config_loader.get_config("validation_params").get("phone", {})
        if not phone_params:
            self.logger.error("Missing phone validation configuration")
            return False
    
        validation_rules = phone_params.get("validation_rules", {})
        confidence_boosts = phone_params.get("confidence_boosts", {})
        reasons = []
    
        # Step 1: Detect & remove leading phone-like prefixes, e.g. "Phone: ", "Ph:", "Tel:", "Cell:", etc.
        # Adjust this list or use your config if you want more or different prefixes.
        prefix_regex = r'^\s*(?:phone|ph|tel|mobile|fax)\s*:\s*'
        raw_text = entity.text.strip()
        prefix_removed = re.sub(prefix_regex, '', raw_text, flags=re.IGNORECASE)
    
        # If we removed something, treat that as a "context" found
        if len(prefix_removed) < len(raw_text):
            prefix_boost = confidence_boosts.get("phone_prefix_context", 0.1)
            entity.confidence += prefix_boost
            reasons.append(f"Detected phone prefix => +{prefix_boost}")
    
        # Step 2: Parse out any extension
        text_lower = prefix_removed.lower()
        has_extension = bool(re.search(r'(?:ext\.?|x)\s*\d+', text_lower))
        main_number = prefix_removed
        ext_digits = ""
    
        if has_extension:
            parts = re.split(r'(?:ext\.?|x)\s*', text_lower, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2:
                main_number = parts[0]
                ext_digits = re.sub(r'[^\d]', '', parts[1])
                # Example: "ext 123" => "123"
                if len(ext_digits) > 5:
                    reasons.append("Extension too long (> 5 digits)")
                    self.last_validation_reasons = reasons
                    return False
    
        # Step 3: Clean digits in main_number
        clean_number = re.sub(r'[^\d]', '', main_number)
        digit_count = len(clean_number)
    
        # Log some debug info
        self.logger.debug(f"Validating phone number: {entity.text}")
        self.logger.debug(f"After prefix removal => '{prefix_removed}'")
        self.logger.debug(f"Digit count: {digit_count}")
        self.logger.debug(f"Clean number: {clean_number}")
    
        # Step 4: Check if "international" by leading '+'
        is_international = prefix_removed.strip().startswith('+')
        self.logger.debug(f"Is international: {is_international}")
    
        # Step 5: Enforce digit counts from config
        min_digits = validation_rules.get("min_digits", 10)
        max_digits = validation_rules.get("max_digits", 15)
        if digit_count < min_digits:
            reasons.append(f"Too few digits: {digit_count}")
            self.last_validation_reasons = reasons
            return False
        if digit_count > max_digits:
            reasons.append(f"Too many digits: {digit_count}")
            self.last_validation_reasons = reasons
            return False
    
        # If exactly 11 digits in a non-international format, ensure it starts with 1, etc.
        if digit_count == 11 and not is_international:
            # e.g. "1.555.123.4567" or "1-555-123-4567" is allowed
            # If it doesn't start with "1" or "(1", fail
            if not re.match(r'^1[)\-.\s(]|^\(1\)', prefix_removed.strip()):
                reasons.append(f"Invalid 11-digit US number, missing leading '1': {digit_count} digits")
                self.last_validation_reasons = reasons
                return False
    
        # Step 6: Attempt pattern match
        matched_pattern = False
        pattern_text = main_number.strip()  # The number minus extension & prefix
    
        for pattern in phone_patterns:
            # Force anchored ^...$
            regex_str = pattern["regex"]
            if not (regex_str.startswith('^') and regex_str.endswith('$')):
                regex_str = '^' + regex_str.strip('^$') + '$'
    
            if re.match(regex_str, pattern_text):
                matched_pattern = True
                # Format match boost
                fmt_boost = confidence_boosts.get("format_match", 0.2)
                entity.confidence += fmt_boost
                reasons.append(f"Matches {pattern['name']} format: +{fmt_boost}")
    
                # If international
                if is_international:
                    intl_boost = confidence_boosts.get("intl_format", 0.1)
                    entity.confidence += intl_boost
                    reasons.append(f"International format: +{intl_boost}")
                else:
                    # US standard
                    standard_boost = confidence_boosts.get("standard_format", 0.1)
                    entity.confidence += standard_boost
                    reasons.append(f"Standard format: +{standard_boost}")
    
                    # Check grouping if 10-digit typical US
                    # e.g. (123) 456-7890 or 123-456-7890
                    grouping_regex = r'^\(?\d{3}\)?[-.\s]+\d{3}[-.\s]+\d{4}$'
                    if re.match(grouping_regex, pattern_text):
                        group_boost = confidence_boosts.get("proper_grouping", 0.1)
                        entity.confidence += group_boost
                        reasons.append(f"Proper digit grouping: +{group_boost}")
    
                break
    
        if not matched_pattern:
            reasons.append("Does not match any valid phone format")
            self.last_validation_reasons = reasons
            return False
    
        # Step 7: Extension boost
        if has_extension:
            ext_boost = confidence_boosts.get("has_extension", 0.1)
            entity.confidence += ext_boost
            reasons.append(f"Valid extension format: +{ext_boost}")
    
        # Step 8: Check for “context words” in the surrounding text
        if text:
            context_words = phone_params.get("context_words", [])
            lower_context = text.lower()
            if any(word.lower() in lower_context for word in context_words):
                ctx_boost = confidence_boosts.get("has_context", 0.1)
                entity.confidence += ctx_boost
                reasons.append(f"Found phone context in surrounding text: +{ctx_boost}")
    
        self.last_validation_reasons = reasons
        self.logger.debug(f"Phone validation results for '{entity.text}': {reasons}")
        return True

    
    def _should_ignore(self, entity_text: str, full_text: str, entity_type: str) -> bool:
        """Determine if an entity should be ignored based on ignore phrases and type-specific logic."""
        self.logger.debug(f"Checking ignore for entity: '{entity_text}' with type '{entity_type}'.")

        for phrase in self.ignore_phrases:
            if phrase.lower() in entity_text.lower():
                self.logger.debug(f"Ignored due to phrase match: '{phrase}' in '{entity_text}'.")
                return True

        if entity_type == "DATE_TIME":
            for phrase in self.ignore_date_words:
                if phrase.lower() in entity_text.lower():
                    self.logger.debug(f"Ignored DATE_TIME due to ignore word: '{phrase}' in '{entity_text}'.")
                    return True

        return False


    def _get_date_confidence_boost(self, entity: Entity) -> float:
        """Calculate confidence boost for date entities based on recognition metadata."""
        try:
            if not entity.metadata or "recognition_metadata" not in entity.metadata:
                return 0.0
    
            metadata = entity.metadata["recognition_metadata"]
            confidence_boost = 0.0
    
            # Check if it's a known date format
            if metadata.get("pattern_name"):
                confidence_boost += 0.1
                self.logger.debug(f"Date matched known pattern: +0.1 boost")
    
            # Check for year presence (important for resumes)
            text_lower = entity.text.lower()
            if re.search(r'(?:19|20)\d{2}', text_lower):
                confidence_boost += 0.1
                self.logger.debug(f"Date contains year: +0.1 boost")
    
            # Check for well-known date markers
            date_markers = {'class of', 'graduated', 'since', 'from', 'to', 'present'}
            if any(marker in text_lower for marker in date_markers):
                confidence_boost += 0.15
                self.logger.debug(f"Date contains known marker: +0.15 boost")
    
            return confidence_boost
    
        except Exception as e:
            self.logger.error(f"Error calculating date confidence boost: {e}")
            return 0.0
    
    def _validate_date_format(self, entity: Entity) -> bool:
        """Additional validation for date formats commonly found in resumes."""
        try:
            text = entity.text.lower().strip()
            
            # Valid patterns
            patterns = [
                # Year ranges (2020-2023, 2020 - present)
                r'(?:19|20)\d{2}\s*[-–—]\s*(?:(?:19|20)\d{2}|present|current|ongoing)',
                # Single years
                r'\b(?:19|20)\d{2}\b',
                # Month year formats
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s*(?:19|20)\d{2}',
                # Year with markers
                r'(?:class\s+of|graduated|expected|anticipated)\s+(?:19|20)\d{2}',
            ]
    
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    self.logger.debug(f"Date format validation passed for pattern: {pattern}")
                    return True
    
            self.logger.debug(f"Date format validation failed for: {text}")
            return False
    
        except Exception as e:
            self.logger.error(f"Error in date format validation: {e}")
            return False

    def _validate_email_address(self, entity: Entity, text: str) -> bool:
        """
        Validate email address entities using configuration-based rules.
        Must be called from validate_detection method.
        """
        if not entity.text:
            return False
    
        email_config = self.config_loader.get_config("validation_params").get("email_address", {})
        if not email_config:
            self.logger.error("Missing email validation configuration")
            return False
    
        text_lower = entity.text.lower()
        reasons = []
        
        # Get configs
        validation_rules = email_config.get("validation_rules", {})
        confidence_boosts = email_config.get("confidence_boosts", {})
        confidence_penalties = email_config.get("confidence_penalties", {})
        known_domains = set(d.lower() for d in email_config.get("known_domains", []))
        patterns = email_config.get("patterns", {})
    
        # Basic length validation
        if len(entity.text) < validation_rules.get("min_chars", 5):
            reasons.append("Email too short")
            self.last_validation_reasons = reasons
            return False
    
        if len(entity.text) > validation_rules.get("max_length", 254):
            penalty = confidence_penalties.get("excessive_length", -0.2)
            entity.confidence += penalty
            reasons.append(f"Email too long: {penalty}")
            self.last_validation_reasons = reasons
            return False
    
        # Must have exactly one @
        if text_lower.count('@') != 1:
            reasons.append("Invalid @ symbol count")
            self.last_validation_reasons = reasons
            return False
    
        # Split into local and domain parts
        try:
            local_part, domain_part = text_lower.split('@')
        except ValueError:
            reasons.append("Invalid email format")
            self.last_validation_reasons = reasons
            return False
    
        # Validate local part
        if not local_part or local_part.startswith('.') or local_part.endswith('.'):
            reasons.append("Invalid local part")
            self.last_validation_reasons = reasons
            return False
    
        # Allow plus addressing if configured
        if '+' in local_part and not validation_rules.get("allow_plus_addressing", True):
            reasons.append("Plus addressing not allowed")
            self.last_validation_reasons = reasons
            return False
    
        # Validate domain part
        if not domain_part or '.' not in domain_part:
            reasons.append("Invalid domain part")
            self.last_validation_reasons = reasons
            return False
    
        # Check against basic pattern first
        import re
        basic_pattern = patterns.get("basic", "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$")
        if not re.match(basic_pattern, entity.text):
            reasons.append("Does not match basic email pattern")
            self.last_validation_reasons = reasons
            return False
    
        # Apply confidence adjustments
        # Check for proper format
        strict_pattern = patterns.get("strict")
        if strict_pattern and re.match(strict_pattern, entity.text):
            boost = confidence_boosts.get("proper_format", 0.2)
            entity.confidence += boost
            reasons.append(f"Matches strict format: +{boost}")
    
        # Known domain boost
        domain_parts = domain_part.split('.')
        if any(part in known_domains for part in domain_parts):
            boost = confidence_boosts.get("known_domain", 0.1)
            entity.confidence += boost
            reasons.append(f"Known domain: +{boost}")
    
        # Context boost
        context_words = {"email", "contact", "mail", "e-mail", "@"}
        text_lower = text.lower()
        if any(word in text_lower for word in context_words):
            boost = confidence_boosts.get("has_context", 0.1)
            entity.confidence += boost
            reasons.append(f"Found email context: +{boost}")
    
        self.last_validation_reasons = reasons
        return True

    def _validate_internet_reference(self, entity: Entity, text: str) -> bool:
        """
        Validate internet reference entities (URLs and social media) using configuration-based rules.
        Must be called from validate_detection method.
        """
        if not entity.text:
            return False
    
        config = self.config_loader.get_config("validation_params").get("internet_reference", {})
        if not config:
            self.logger.error("Missing internet reference validation configuration")
            return False
    
        text_lower = entity.text.lower()
        reasons = []
    
        # Get configs
        validation_rules = config.get("validation_rules", {})
        confidence_boosts = config.get("confidence_boosts", {})
        confidence_penalties = config.get("confidence_penalties", {})
        known_platforms = config.get("known_platforms", {})
        validation_types = config.get("validation_types", {})
    
        # Basic length validation
        if len(entity.text) < validation_rules.get("min_chars", 4):
            reasons.append("Reference too short")
            self.last_validation_reasons = reasons
            return False
    
        if len(entity.text) > validation_rules.get("max_length", 2048):
            penalty = confidence_penalties.get("excessive_length", -0.2)
            entity.confidence += penalty
            reasons.append(f"Reference too long: {penalty}")
            self.last_validation_reasons = reasons
            return False
    
        import re
    
        # Check if it's a social media handle
        if text_lower.startswith('@'):
            # Validate handle format
            handle_rules = validation_types.get("social_handle", {})
            handle_pattern = f"^@[{handle_rules.get('allowed_chars', 'A-Za-z0-9_-')}]+$"
            max_length = handle_rules.get("max_length", 30)
    
            if not re.match(handle_pattern, entity.text):
                reasons.append("Invalid handle format")
                self.last_validation_reasons = reasons
                return False
    
            if len(entity.text) > max_length:
                penalty = confidence_penalties.get("excessive_length", -0.2)
                entity.confidence += penalty
                reasons.append(f"Handle too long: {penalty}")
                self.last_validation_reasons = reasons
                return False
    
            # Handle format boost
            boost = confidence_boosts.get("proper_format", 0.1)
            entity.confidence += boost
            reasons.append(f"Valid handle format: +{boost}")
    
        else:
            # Validate as URL
            url_rules = validation_types.get("url", {})
            
            # Basic URL pattern
            url_pattern = (
                r'^(?:https?:\/\/)?'  # Optional protocol
                r'(?:[\w-]+\.)+[a-zA-Z]{2,}'  # Domain
                r'(?:\/[^\s]*)?$'  # Optional path
            )
    
            if not re.match(url_pattern, text_lower):
                reasons.append("Invalid URL format")
                self.last_validation_reasons = reasons
                return False
    
            # Protocol boost
            if text_lower.startswith(('http://', 'https://')):
                boost = confidence_boosts.get("has_protocol", 0.1)
                entity.confidence += boost
                reasons.append(f"Has protocol: +{boost}")
    
        # Check for known platforms
        social_platforms = set(p.lower() for p in known_platforms.get("social", []))
        common_domains = set(d.lower() for d in known_platforms.get("common_domains", []))
        
        if any(platform in text_lower for platform in social_platforms):
            boost = confidence_boosts.get("known_platform", 0.2)
            entity.confidence += boost
            reasons.append(f"Known social platform: +{boost}")
        elif any(domain in text_lower for domain in common_domains):
            boost = confidence_boosts.get("known_platform", 0.1)
            entity.confidence += boost
            reasons.append(f"Known domain: +{boost}")
    
        # Context boost
        context_words = {
            "website", "link", "url", "profile", "handle", "twitter", "github",
            "linkedin", "social", "follow", "connect", "web", "visit"
        }
        text_lower = text.lower()
        if any(word in text_lower for word in context_words):
            boost = confidence_boosts.get("has_context", 0.1)
            entity.confidence += boost
            reasons.append(f"Found web context: +{boost}")
    
        self.last_validation_reasons = reasons
        return True