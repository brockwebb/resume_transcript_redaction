# redactor/validation/validation_coordinator.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set, Tuple
import re
import logging
import json
from dataclasses import dataclass
from pathlib import Path

from ..detectors.base_detector import Entity

@dataclass
class ValidationResult:
    """Result of entity validation including confidence adjustments."""
    is_valid: bool
    confidence_adjustment: float = 0.0
    reasons: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.reasons is None:
            self.reasons = []
        if self.metadata is None:
            self.metadata = {}

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class ValidationCoordinator:
    """
    Coordinates validation logic across detectors.
    Uses configuration-driven validation rules and confidence adjustments.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None):
        """
        Initialize validation coordinator with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
        """
        self.logger = logger
        self.config_loader = config_loader
        
        # Debug mode setup
        self.debug_mode = False
        if logger:
            if hasattr(logger, 'level'):
                self.debug_mode = logger.level == 10
            elif hasattr(logger, 'getEffectiveLevel'):
                self.debug_mode = logger.getEffectiveLevel() == 10
            elif hasattr(logger, 'log_level'):
                self.debug_mode = logger.log_level == "DEBUG"
        
        # Load configurations
        self.validation_params = self._load_validation_params()
        self.routing_config = self._load_routing_config()
        
        # Initialize validation methods based on config
        self.validation_methods = {}
        self._init_validation_methods()
        
        if self.debug_mode:
            self.logger.debug("Initialized ValidationCoordinator")
            self.logger.debug(f"Loaded validation methods: {list(self.validation_methods.keys())}")

    def _load_validation_params(self) -> Dict:
        """Load validation parameters from configuration."""
        try:
            validation_config = self.config_loader.get_config("validation_params")
            if not validation_config:
                if self.logger:
                    self.logger.error("Missing validation parameters configuration")
                raise ValueError("Missing validation parameters")
                
            if self.debug_mode:
                self.logger.debug("Loaded validation parameters")
                
            return validation_config
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading validation parameters: {str(e)}")
            raise

    def _load_routing_config(self) -> Dict:
        """Load entity routing configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                if self.logger:
                    self.logger.error("Missing entity routing configuration")
                raise ValueError("Missing entity routing")
                
            if self.debug_mode:
                self.logger.debug("Loaded entity routing configuration")
                
            return routing_config.get("routing", {})
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading routing configuration: {str(e)}")
            raise

    def _init_validation_methods(self) -> None:
        """Initialize validation methods based on configuration."""
        if self.debug_mode:
            self.logger.debug("Initializing validation methods")
        
        entity_types = set()
        for section in ["presidio_primary", "spacy_primary", "ensemble_required"]:
            if section in self.routing_config:
                entity_types.update(self.routing_config[section].get("entities", []))
        
        #type_to_method = self.config_loader.get_config("validation_config")
        config = self.config_loader.get_config("validation_config")
        type_to_method = config.get("validation_methods", {})
        if not type_to_method:
            if self.logger:
                self.logger.error("Missing validation method mappings config")
            raise ValueError("Missing validation config")
                
        if self.debug_mode:
            self.logger.debug(f"Entity types: {entity_types}")
            self.logger.debug(f"Method mappings: {type_to_method}")
    
        # Register validation methods
        for entity_type in entity_types:
            if entity_type in type_to_method:
                method_name = type_to_method[entity_type] 
                if hasattr(self, method_name):
                    self.validation_methods[entity_type] = getattr(self, method_name)
                    if self.debug_mode:
                        self.logger.debug(f"Mapped {entity_type} to {method_name}")
            

    def validate_entity(self, entity: Entity, text: str, source: str = None) -> ValidationResult:
        """
        Validate entity using configuration-driven rules.
        """
        if not entity or not entity.text:
            return ValidationResult(is_valid=False, reasons=["Empty entity"])
    
        if self.debug_mode:
            self.logger.debug(f"\n=== Starting Validation ===")
            self.logger.debug(f"Entity: {entity.text} ({entity.entity_type})")
            self.logger.debug(f"Initial confidence: {entity.confidence}")
    
        validator = self.validation_methods.get(entity.entity_type)
        validation_rules = self.validation_params.get(entity.entity_type, {})
    
        if not validator:
            if self.debug_mode:
                self.logger.debug("No validator found - using default")
            return ValidationResult(is_valid=True)
    
        if self.debug_mode:
            self.logger.debug(f"Using validator: {validator.__name__}")
            self.logger.debug(f"Validation rules: {validation_rules}")
    
        result = validator(entity, text, validation_rules, source)
    
        if self.debug_mode:
            self.logger.debug(f"\n=== Validation Complete ===")
            self.logger.debug(f"Valid: {result.is_valid}")
            self.logger.debug(f"Adjustment: {result.confidence_adjustment}")
            self.logger.debug(f"Reasons: {result.reasons}")
    
        if result and result.is_valid:
            if result.confidence_adjustment != 0:
                prev_conf = entity.confidence
                entity.confidence += result.confidence_adjustment
                if self.debug_mode:
                    self.logger.debug(f"\n=== Confidence Update ===")
                    self.logger.debug(f"Previous: {prev_conf}")
                    self.logger.debug(f"Adjustment: {result.confidence_adjustment}")
                    self.logger.debug(f"Final: {entity.confidence}")
    
        return result


    def _validate_educational(
        self, 
        entity: Entity, 
        text: str,
        rules: Dict, 
        source: str = None
    ) -> ValidationResult:
        """Validate educational institution using config rules."""
        if self.debug_mode:
            self.logger.debug(f"Starting educational validation: {entity.text}")
            self.logger.debug(f"Rules: {rules}")
        
        # Extract needed config
        confidence_boosts = rules.get("confidence_boosts", {})
        confidence_penalties = rules.get("confidence_penalties", {})
        core_terms = set(rules.get("core_terms", []))
        known_acronyms = set(rules.get("known_acronyms", []))
        generic_terms = set(rules.get("generic_terms", []))
        min_words = rules.get("min_words", 2)
    
        text_lower = entity.text.lower()
        words_lower = text_lower.split()        # purely lowercase words
        words_orig = entity.text.split()        # original text (for capitalization checks)
        reasons = []
        confidence_adjustment = 0.0
    
        # -------------------------------------------------------------------------
        # NEGATIVE (FAIL) CHECKS: Return is_valid=False if specific conditions fail
        # -------------------------------------------------------------------------
    
        # 1) If there's no text or all whitespace, fail fast.
        if not entity.text.strip():
            reasons.append("Empty or blank educational institution text")
            return ValidationResult(is_valid=False, reasons=reasons)
    
        # 2) If fewer than min_words, check for known acronym or capitalization
        #    e.g. single word "mit" is invalid unless recognized as "MIT".
        if len(words_lower) < min_words:
            if entity.text not in known_acronyms:
                if not entity.text[0].isupper():
                    reasons.append(
                        "Single-word institution with no capitalization "
                        "and not a known acronym"
                    )
                    return ValidationResult(is_valid=False, reasons=reasons)
    
        # 3) If the first character is not uppercase (and not in known_acronyms),
        #    fail if your test requires a capital letter for validity.
        if entity.text and not entity.text[0].isupper():
            if entity.text not in known_acronyms:
                reasons.append("Educational institution missing capitalization")
                return ValidationResult(is_valid=False, reasons=reasons)
    
        # 4) If the entire text is a "generic term" you specifically disallow,
        #    mark invalid. For example, maybe plain "school" is too vague.
        if entity.text.lower() in generic_terms:
            reasons.append(f"Matches disallowed generic term: {entity.text}")
            return ValidationResult(is_valid=False, reasons=reasons)
    
        # -------------------------------------------------------------------------
        # POSITIVE (BOOST) CHECKS: Apply confidence adjustments if it passes above
        # -------------------------------------------------------------------------
    
        # (A) Known acronym boost
        if entity.text in known_acronyms:
            boost_acr = confidence_boosts.get("known_acronym", 0.1)
            confidence_adjustment += boost_acr
            reasons.append(f"Known acronym institution: +{boost_acr}")
    
        # (B) Core term check
        has_core_term = any(term in text_lower for term in core_terms)
        if has_core_term:
            boost = confidence_boosts.get("core_term_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Core term match: +{boost}")
            if self.debug_mode:
                self.logger.debug(f"Added core term boost: {boost}")
    
        # (C) Multi-word institution
        if len(words_lower) >= min_words:
            boost = confidence_boosts.get("full_name", 0.3)
            confidence_adjustment += boost
            reasons.append(f"Full name: +{boost}")
            if self.debug_mode:
                self.logger.debug(f"Added multi-word boost: {boost}")
    
        if self.debug_mode:
            self.logger.debug(f"Final adjustment: {confidence_adjustment}")
            self.logger.debug(f"Reasons: {reasons}")
    
        # Passed negative checks, so it's valid
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )


        

###############################################################################
# SECTION 3: CORE VALIDATION METHODS
###############################################################################
    
    def _get_validation_rules(self, entity_type: str) -> Dict:
        """
        Get validation rules for entity type from config.
        
        Args:
            entity_type: Type of entity
            
        Returns:
            Dict containing validation rules
        """
        rules = self.validation_params.get(entity_type, {})
        if not rules and self.debug_mode:
            self.logger.debug(f"No validation rules found for {entity_type}")
        return rules

    def _apply_confidence_adjustment(self, confidence: float,
                                   adjustment: float,
                                   min_val: float = 0.0,
                                   max_val: float = 1.0) -> float:
        """
        Apply confidence adjustment within bounds.
        
        Args:
            confidence: Base confidence value
            adjustment: Adjustment to apply
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            
        Returns:
            Adjusted confidence value
        """
        new_confidence = confidence + adjustment
        return max(min_val, min(max_val, new_confidence))

###############################################################################
# SECTION 4: DATE VALIDATION
###############################################################################

    def _validate_date(self, entity: Entity, text: str,
                      rules: Dict, source: str = None) -> ValidationResult:
        """Main entry point for date/time validation."""
        ensemble_config = self._get_ensemble_config()
        if source == "ensemble":
            presidio_weight = ensemble_config.get("presidio_weight")
            spacy_weight = ensemble_config.get("spacy_weight")
            return self._validate_ensemble_date(entity, text, rules, presidio_weight, spacy_weight)
        elif source == "spacy":
            return self._validate_spacy_date(entity, text, rules)
        return self._validate_presidio_date(entity, text, rules)

    def _get_ensemble_config(self) -> Dict:
        routing_config = self.config_loader.get_config("entity_routing")
        ensemble_config = routing_config.get("routing", {}).get("ensemble_required", {})
        return ensemble_config.get("confidence_thresholds", {}).get("DATE_TIME", {})

    def _validate_spacy_date(self, entity: Entity, text: str, rules: Dict) -> ValidationResult:
        """SpaCy validation focusing on linguistic patterns."""
        confidence_adjustments = rules.get("confidence_adjustments", {})
        year_markers = set(rules.get("year_markers", []))
        text_lower = text.lower()
        total_confidence = 0.0
        reasons = []

        # Check for year markers
        if any(marker in text_lower for marker in year_markers):
            boost = confidence_adjustments.get("has_marker", 0.2)
            total_confidence += boost
            reasons.append(f"Year marker context: +{boost}")

        # Check education year pattern first
        education_pattern = rules.get("patterns", {}).get("education_year", {})
        if education_pattern:
            pattern = education_pattern.get("regex", "")
            if pattern and re.match(pattern, entity.text, re.IGNORECASE):
                conf = education_pattern.get("confidence", 1.0)
                total_confidence += conf
                reasons.append(f"Education year pattern: +{conf}")
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=total_confidence,
                    reasons=reasons
                )

        return ValidationResult(
            is_valid=bool(reasons),
            confidence_adjustment=total_confidence,
            reasons=reasons
        )

    def _validate_presidio_date(self, entity: Entity, text: str, rules: Dict) -> ValidationResult:
        """Presidio validation focusing on regex patterns."""
        patterns = rules.get("patterns", {})
        confidence_adjustments = rules.get("confidence_adjustments", {})
        year_markers = set(rules.get("year_markers", []))
        text_lower = text.lower()
        total_confidence = 0.0
        reasons = []
    
        # Extensive logging for debugging
        if self.debug_mode and self.logger:
            self.logger.debug(f"\n=== Date Validation Debug ===")
            self.logger.debug(f"Entity Text: {entity.text}")
            self.logger.debug(f"Full Text Context: {text}")
            self.logger.debug(f"Year Markers: {year_markers}")
            self.logger.debug(f"Patterns: {list(patterns.keys())}")
    
        # Add year marker boost first (always check context)
        if any(marker in text_lower for marker in year_markers):
            boost = confidence_adjustments.get("has_marker", 0.2)
            total_confidence += boost
            reasons.append(f"Year marker context: +{boost}")
            if self.debug_mode and self.logger:
                self.logger.debug(f"Year marker found. Boost: +{boost}")
    
        # Get all patterns in priority order (This should be moved to a conf file)
        pattern_priority = ["education_year", "year_range", "year_month", "month_year", "single_year"]
        
        # Detailed pattern matching and logging
        for pattern_name in pattern_priority:
            pattern_info = patterns.get(pattern_name, {})
            pattern = pattern_info.get("regex", "")
            
            if self.debug_mode and self.logger:
                self.logger.debug(f"\nChecking Pattern: {pattern_name}")
                self.logger.debug(f"Regex Pattern: {pattern}")
            
            try:
                # Handle both dictionary and direct confidence values
                if isinstance(pattern_info.get("confidence"), dict):
                    conf = pattern_info.get("confidence", {}).get("confidence", 0.7)
                else:
                    conf = pattern_info.get("confidence", 0.7)
                
                if self.debug_mode and self.logger:
                    self.logger.debug(f"Confidence: {conf}")
    
                # Actual regex matching
                if pattern and re.match(pattern, entity.text, re.IGNORECASE):
                    if self.debug_mode and self.logger:
                        self.logger.debug(f"MATCH FOUND for {pattern_name}")
                    
                    # Force education patterns to highest confidence
                    if pattern_name == "education_year":
                        conf = 1.0
                    elif pattern_name == "year_range":
                        conf = 0.85
                    
                    total_confidence += conf
                    reasons.append(f"Matched {pattern_name} pattern: +{conf}")
                    
                    if self.debug_mode and self.logger:
                        self.logger.debug(f"Final Confidence: {total_confidence}")
                        self.logger.debug(f"Reasons: {reasons}")
                    
                    return ValidationResult(
                        is_valid=True,
                        confidence_adjustment=total_confidence,
                        reasons=reasons
                    )
            except Exception as e:
                if self.debug_mode and self.logger:
                    self.logger.debug(f"Error processing {pattern_name}: {str(e)}")
    
        # Log if no patterns matched
        if self.debug_mode and self.logger:
            self.logger.debug("NO PATTERNS MATCHED")
            self.logger.debug(f"Total Confidence: {total_confidence}")
            self.logger.debug(f"Reasons: {reasons}")
    
        # Fallback return
        return ValidationResult(
            is_valid=bool(reasons),
            confidence_adjustment=total_confidence,
            reasons=reasons
        )

    def _validate_ensemble_date(self, entity: Entity, text: str, rules: Dict,
                              presidio_weight: float, spacy_weight: float) -> ValidationResult:
        """Combine Presidio and SpaCy results with weights."""
        presidio_result = self._validate_presidio_date(entity, text, rules)
        spacy_result = self._validate_spacy_date(entity, text, rules)

        # Take highest confidence
        if presidio_result.confidence_adjustment >= spacy_result.confidence_adjustment:
            return presidio_result
        return spacy_result

###############################################################################
# SECTION 5: PERSON AND LOCATION VALIDATION
###############################################################################
    
    def _validate_person(self, entity: Entity, text: str,
                        rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate person entities using config rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        confidence_penalties = rules.get("confidence_penalties", {})
        title_markers = set(rules.get("title_markers", []))
        
        text_lower = entity.text.lower()
        name_parts = entity.text.split()
        reasons = []
        confidence_adjustment = 0.0

        # Length validation
        if len(entity.text) < validation_rules.get("min_chars", 2):
            reasons.append(f"Name too short: {len(entity.text)} chars")
            return ValidationResult(is_valid=False, reasons=reasons)

        if len(name_parts) < validation_rules.get("min_words", 1):
            reasons.append("Insufficient word count")
            return ValidationResult(is_valid=False, reasons=reasons)

        # Title check
        has_title = any(marker in text_lower for marker in title_markers)
        if has_title:
            boost = confidence_boosts.get("has_title", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Title marker found: +{boost}")

        # Handle ALL CAPS names
        if entity.text.isupper():
            if validation_rules.get("allow_all_caps", True):
                if len(name_parts) >= 2 or has_title:
                    boost = confidence_boosts.get("multiple_words", 0.1)
                    confidence_adjustment += boost
                    reasons.append(f"Valid ALL CAPS name: +{boost}")
                else:
                    reasons.append("Single word ALL CAPS without title")
                    return ValidationResult(is_valid=False, reasons=reasons)
        else:
            # Regular capitalization check
            if not all(part[0].isupper() for part in name_parts):
                reasons.append("Name parts must be properly capitalized")
                return ValidationResult(is_valid=False, reasons=reasons)
            boost = confidence_boosts.get("proper_case", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Proper capitalization: +{boost}")

        # Multiple word boost
        if len(name_parts) >= 2:
            boost = confidence_boosts.get("multiple_words", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Multiple words: +{boost}")

        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

    def _check_location_formats(self, entity: Entity, format_patterns: Dict, confidence_boosts: Dict) -> Tuple[List[str], float, List[str]]:
        """
        Check location formats using regex patterns.
        
        Returns:
            A tuple (matched_formats, total_boost, reasons)
        """
        matched_formats = []
        total_boost = 0.0
        reasons = []
        text = entity.text
    
        # Check US city-state format first
        us_pattern = format_patterns.get("us_city_state", {}).get("regex", "")
        if us_pattern and re.match(us_pattern, text):
            matched_formats.append("us_city_state")
            boost = confidence_boosts.get("known_format", 0.1) + confidence_boosts.get("has_state", 0.2)
            total_boost += boost
            reasons.append(f"Matches US city-state format: +{boost}")
            if re.search(r'\d{5}(?:-\d{4})?$', text):
                zip_boost = confidence_boosts.get("has_zip", 0.1)
                total_boost += zip_boost
                reasons.append(f"Includes ZIP code: +{zip_boost}")
                matched_formats.append("zip_code")
        else:
            # Check international format (allows diacritics)
            international_pattern = format_patterns.get("international", {}).get("regex", "")
            if international_pattern and re.match(international_pattern, text):
                matched_formats.append("international")
                boost = confidence_boosts.get("international", 0.1)
                total_boost += boost
                reasons.append(f"Matches international format: +{boost}")
        
        # Check metro area format if applicable
        metro_pattern = format_patterns.get("metro_area", {}).get("regex", "")
        if metro_pattern and re.match(metro_pattern, text):
            matched_formats.append("metro_area")
            boost = confidence_boosts.get("metro_area", 0.2)
            total_boost += boost
            reasons.append(f"Matches metro area format: +{boost}")
        
        return matched_formats, total_boost, reasons
    
    
    def _validate_location(self, entity: Entity, text: str,
                           rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate location entities using config rules.
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        confidence_penalties = rules.get("confidence_penalties", {})
        format_patterns = rules.get("format_patterns", {})
        us_states = rules.get("us_states", {})
        
        text_lower = entity.text.lower()
        words = entity.text.split()
        reasons = []
        confidence_adjustment = 0.0
    
        # Basic length check
        if len(entity.text) < validation_rules.get("min_chars", 2):
            reasons.append("Location too short")
            return ValidationResult(is_valid=False, reasons=reasons)
        
        # Capitalization check
        if validation_rules.get("require_capitalization", True):
            if not entity.text[0].isupper():
                reasons.append("Location not properly capitalized")
                return ValidationResult(is_valid=False, reasons=reasons)
        
        # Use helper to check format patterns
        matched_formats, boost_from_formats, format_reasons = self._check_location_formats(entity, format_patterns, confidence_boosts)
        confidence_adjustment += boost_from_formats
        reasons.extend(format_reasons)
        
        # Handle single-word locations
        if len(words) == 1:
            if not validation_rules.get("allow_single_word", True):
                reasons.append("Single word locations not allowed")
                return ValidationResult(is_valid=False, reasons=reasons)
            if entity.text.isupper() and validation_rules.get("allow_all_caps", True):
                boost = confidence_boosts.get("proper_case", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Valid ALL CAPS location: +{boost}")
            elif not matched_formats:
                penalty = confidence_penalties.get("single_word", -0.2)
                confidence_adjustment += penalty
                reasons.append(f"Single word penalty: {penalty}")
        else:
            # For multi-word, if no format was matched, add a multi-word boost
            if not matched_formats:
                boost = confidence_boosts.get("multi_word", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Multi-word location: +{boost}")
        
        # Additional check for US state if not already matched by us_city_state
        if "us_city_state" not in matched_formats:
            state_list = [s.lower() for s in us_states.get("full", []) + us_states.get("abbrev", [])]
            if any(state in text_lower for state in state_list):
                boost = confidence_boosts.get("has_state", 0.2)
                confidence_adjustment += boost
                reasons.append(f"Contains US state: +{boost}")
        
        # Fallback reason if no specific format matched
        if not matched_formats and not reasons:
            reasons.append("Basic validation passed but no specific format matched")
        
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

    def _validate_address(self, entity: Entity, text: str,
                           rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate address entities using config rules.
        """
        if not entity.text:
            return ValidationResult(is_valid=False, reasons=["Empty address"])
            
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        confidence_penalties = rules.get("confidence_penalties", {})
        
        # Load type lists from config
        street_types = set(
            t.lower() for t in 
            rules.get("street_types", {}).get("full", []) +
            rules.get("street_types", {}).get("abbrev", [])
        )
        unit_types = set(t.lower() for t in rules.get("unit_types", []))
        directionals = set(d.lower() for d in rules.get("directionals", []))
        
        text_lower = entity.text.lower()
        reasons = []
        confidence_adjustment = 0.0
        
        # Basic validation
        if len(entity.text) < validation_rules.get("min_chars", 5):
            return ValidationResult(
                is_valid=False,
                reasons=["Address too short"]
            )
    
        # Check for required number
        if validation_rules.get("require_number", True):
            if not any(c.isdigit() for c in entity.text):
                penalty = confidence_penalties.get("no_number", -0.5)
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=penalty,
                    reasons=[f"Missing street number: {penalty}"]
                )
    
        # Check for street type using word boundaries
        has_street_type = False
        for st in street_types:
            if re.search(rf'\b{st}\b(?:[,\s]|$)', text_lower):
                has_street_type = True
                break
    
        if validation_rules.get("require_street_type", True) and not has_street_type:
            penalty = confidence_penalties.get("no_street_type", -0.3)
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=penalty,
                reasons=[f"Missing street type: {penalty}"]
            )
        else:
            boost = confidence_boosts.get("street_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Valid street type: +{boost}")
    
        # Check for unit/suite numbers
        has_unit = False
        for unit in unit_types:
            if re.search(rf'\b{unit}\b', text_lower):
                has_unit = True
                boost = confidence_boosts.get("has_unit", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Has unit/suite: +{boost}")
                break
    
        # Check for directionals
        has_directional = False
        for dir in directionals:
            if re.search(rf'\b{dir}\b', text_lower):
                has_directional = True
                boost = confidence_boosts.get("directional", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Has directional: +{boost}")
                break
    
        # Additional boost for complete address structure
        if len(reasons) >= 2:  # Has at least two valid components
            boost = confidence_boosts.get("full_address", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Complete address structure: +{boost}")
    
        if self.debug_mode:
            self.logger.debug(f"Address validation results for '{entity.text}': {reasons}")
    
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )


###############################################################################
# SECTION 6: SPECIAL ENTITY VALIDATION
###############################################################################

    def _validate_email(self, entity: Entity, text: str,
                           rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate email address using config rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        patterns = rules.get("patterns", {})
        known_domains = set(rules.get("known_domains", []))
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Length validation
        if len(entity.text) < validation_rules.get("min_chars", 5):
            reasons.append("Email too short")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        if len(entity.text) > validation_rules.get("max_length", 254):
            reasons.append("Email too long")
            return ValidationResult(is_valid=False, reasons=reasons)
        
        # Split into local and domain parts
        if '@' not in entity.text:
            reasons.append("Missing @ symbol")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        local_part, domain = entity.text.rsplit('@', 1)
        
        # Local part validation
        if not local_part or local_part[-1] == '.':
            reasons.append("Invalid local part format")
            return ValidationResult(is_valid=False, reasons=reasons)
        
        if '..' in local_part:
            reasons.append("Consecutive dots in local part")
            return ValidationResult(is_valid=False, reasons=reasons)
        
        # Basic format check using pattern
        basic_pattern = patterns.get("basic", "")
        if not re.match(basic_pattern, entity.text):
            reasons.append("Invalid email format")
            return ValidationResult(is_valid=False, reasons=reasons)
        
        # Known domain boost
        if any(domain.lower().endswith(d) for d in known_domains):
            boost = confidence_boosts.get("known_domain", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Known domain: +{boost}")
            
        # Context boost
        context_words = {"email", "contact", "mail", "address", "@"}
        if any(word in text.lower() for word in context_words):
            boost = confidence_boosts.get("has_context", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Email context: +{boost}")
        
        # Format boost
        if re.match(patterns.get("strict", ""), entity.text):
            boost = confidence_boosts.get("proper_format", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Strict format match: +{boost}")
                
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )
        

    def _validate_phone(self, entity: Entity, text: str,
                        rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate phone number using config rules.
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        context_words = set(rules.get("context_words", []))
        patterns = rules.get("patterns", {})
        
        if self.debug_mode:
            self.logger.debug(f"\nValidating phone: {entity.text}")
            self.logger.debug(f"External context text: {text}")
            self.logger.debug(f"Context words: {context_words}")
        
        reasons = []
        confidence_adjustment = 0.0
        format_matched = False
        
        # Extract digits from original text
        digits = ''.join(c for c in entity.text if c.isdigit())
        
        # Global digit count constraints
        if len(digits) < validation_rules.get("min_digits", 10):
            return ValidationResult(is_valid=False, confidence_adjustment=0.0, reasons=["Too few digits"])
        if len(digits) > validation_rules.get("max_digits", 15):
            return ValidationResult(is_valid=False, confidence_adjustment=0.0, reasons=["Too many digits"])
        
        phone_part = entity.text
        
        # Check for extension and strip it
        if patterns.get("extension") and re.search(patterns["extension"], phone_part, re.IGNORECASE):
            boost = confidence_boosts.get("has_extension", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Extension present: +{boost}")
            phone_part = re.split(patterns["extension"], phone_part, flags=re.IGNORECASE)[0].strip()
        
        # Check external context (if provided)
        has_context = False
        if text:
            has_context = any(word in text.lower() for word in context_words)
            # Fallback: if external context does not contain any of the words, check if it contains "phone"
            if not has_context and "phone" in text.lower():
                has_context = True
        # If no external context provided, check entity text itself
        if not text:
            if any(word in entity.text.lower() for word in context_words) or "phone" in entity.text.lower():
                has_context = True
        
        if self.debug_mode:
            self.logger.debug(f"Context match found: {has_context}")
        
        if has_context:
            boost = confidence_boosts.get("has_context", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Phone context: +{boost}")
            if ':' in phone_part:
                phone_part = phone_part.split(":")[-1].strip()
        
        # Recalculate digits from the possibly modified phone_part
        phone_digits = ''.join(c for c in phone_part if c.isdigit())
        
        # Check US and international formats first.
        # Full international: expect exactly 11 digits starting with "1"
        if patterns.get("full_intl") and re.search(patterns["full_intl"], phone_part):
            if len(phone_digits) != 11 or not phone_digits.startswith('1'):
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    reasons=["Invalid international format digit count"]
                )
            boost = confidence_boosts.get("intl_format", 0.2)
            confidence_adjustment += boost
            reasons.append(f"International format: +{boost}")
            format_matched = True
        # Area code format: also expect exactly 11 digits starting with "1"
        elif patterns.get("area_code") and re.search(patterns["area_code"], phone_part):
            if len(phone_digits) != 11 or not phone_digits.startswith('1'):
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    reasons=["Invalid area code format digit count"]
                )
            boost = confidence_boosts.get("format_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Standard format: +{boost}")
            format_matched = True
        # Standard US formats (parentheses or dash/dot): expect exactly 10 digits
        elif (patterns.get("parentheses") and re.search(patterns["parentheses"], phone_part)) or \
             (patterns.get("standard") and re.search(patterns["standard"], phone_part)):
            if len(phone_digits) != 10:
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    reasons=["Invalid US format digit count"]
                )
            boost = confidence_boosts.get("format_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Standard format: +{boost}")
            format_matched = True
        # New branch: UK format detection based on +44 country code
        elif phone_part.strip().startswith("+44"):
            uk_digits = ''.join(c for c in phone_part if c.isdigit())
            # For UK numbers, you might expect 11 or 12 digits (adjust as needed)
            if len(uk_digits) not in [11, 12]:
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    reasons=["Invalid UK format digit count"]
                )
            boost = confidence_boosts.get("uk_format", 0.3)
            confidence_adjustment += boost
            reasons.append(f"UK format: +{boost}")
            format_matched = True
        
        # Add proper grouping boost if a format was matched
        if format_matched:
            boost = confidence_boosts.get("proper_grouping", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Proper grouping: +{boost}")
        
        if self.debug_mode:
            self.logger.debug(f"Final confidence adjustment: {confidence_adjustment}")
            self.logger.debug(f"Format matched: {format_matched}")
            self.logger.debug(f"Has context: {has_context}")
            self.logger.debug(f"Reasons: {reasons}")
        
        # Final validity determination: valid if a recognized format or context exists.
        is_valid = bool(format_matched or has_context)
        
        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

        

    def _validate_gpa(self, entity: Entity, text: str,
                      rules: Dict, source: str = None) -> ValidationResult:
       """
       Validate GPA entities using numeric range and format checks.
       
       Args:
           entity: Entity to validate
           text: Original text context
           rules: Validation rules from config
           source: Optional detector source
       """
       validation_rules = rules.get("validation_rules", {})
       confidence_boosts = rules.get("confidence_boosts", {})
       confidence_penalties = rules.get("confidence_penalties", {})
       indicators = set(rules.get("indicators", []))
       scale_indicators = set(rules.get("scale_indicators", []))
       text_lower = text.lower()
       entity_lower = entity.text.lower()
       reasons = []
       confidence_adjustment = 0.0
    
       # Extract numeric value using more flexible regex
       numeric_match = re.search(r'(-?\d+\.?\d*)', entity.text)
       if not numeric_match:
           reasons.append("No valid GPA value found")
           return ValidationResult(is_valid=False, reasons=reasons)
    
       try:
           gpa_value = float(numeric_match.group(1))
       except ValueError:
           reasons.append("Could not parse GPA value")
           return ValidationResult(is_valid=False, reasons=reasons)
    
       # Early range validation
       min_value = validation_rules.get("min_value", 0.0)
       max_value = validation_rules.get("max_value", 4.0)
    
       # Strict value range check
       if gpa_value < min_value or gpa_value > max_value:
           reasons.append(f"GPA value {gpa_value} outside valid range [{min_value}, {max_value}]")
           penalty = confidence_penalties.get("invalid_value", -0.5)
           return ValidationResult(is_valid=False, 
                                   confidence_adjustment=penalty, 
                                   reasons=reasons)
    
       # Check for scale format (e.g., "3.75/4.0")
       has_scale = '/' in entity.text
       if has_scale:
           scale_part = entity.text.split('/')[1].strip()
           try:
               scale_value = float(scale_part)
               if scale_value != max_value:
                   reasons.append(f"Invalid scale value: {scale_value} (expected {max_value})")
                   penalty = confidence_penalties.get("invalid_value", -0.5)
                   return ValidationResult(is_valid=False, 
                                           confidence_adjustment=penalty, 
                                           reasons=reasons)
               
               boost = confidence_boosts.get("has_scale", 0.1)
               confidence_adjustment += boost
               reasons.append(f"Valid scale indicator: +{boost}")
           except ValueError:
               reasons.append("Invalid scale format")
               return ValidationResult(is_valid=False, reasons=reasons)
    
       # Check for GPA indicators in text context
       has_indicator = any(indicator.lower() in text_lower or indicator.lower() in entity_lower 
                           for indicator in indicators)
       
       # Only require indicator if not in scale format
       if not has_indicator and not has_scale:
           if validation_rules.get("require_indicator", True):
               reasons.append("Missing GPA indicator")
               penalty = confidence_penalties.get("no_indicator", -0.3)
               return ValidationResult(is_valid=False, 
                                       confidence_adjustment=penalty, 
                                       reasons=reasons)
       
       # Add confidence boosts
       if has_indicator:
           boost = confidence_boosts.get("has_indicator", 0.3)  # Increased from 0.2
           confidence_adjustment += boost
           reasons.append(f"Found GPA indicator: +{boost}")
    
       # Check decimal places
       if '.' in str(gpa_value):
           decimal_places = len(str(gpa_value).split('.')[-1])
           if decimal_places > validation_rules.get("decimal_places", 2):
               reasons.append(f"Too many decimal places: {decimal_places}")
               penalty = confidence_penalties.get("suspicious_format", -0.2)
               return ValidationResult(is_valid=False, 
                                       confidence_adjustment=penalty, 
                                       reasons=reasons)
    
       # Add range-based confidence boost
       if 2.0 <= gpa_value <= max_value:
           boost = confidence_boosts.get("proper_value_range", 0.2)
           confidence_adjustment += boost
           reasons.append(f"Valid GPA range: +{boost}")
    
       if self.debug_mode:
           self.logger.debug(f"GPA validation results for '{entity.text}': {reasons}")
    
       return ValidationResult(
           is_valid=True,
           confidence_adjustment=confidence_adjustment,
           reasons=reasons
       )


    def _validate_internet_reference(self, entity: Entity, text: str,
                                       rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate internet reference (URL, social media) using config rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        validation_types = rules.get("validation_types", {})
        known_platforms = rules.get("known_platforms", {})
        
        social_platforms = set(known_platforms.get("social", []))
        common_domains = set(known_platforms.get("common_domains", []))
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Basic validation
        if len(entity.text) < validation_rules.get("min_chars", 4):
            reasons.append("Reference too short")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        if len(entity.text) > validation_rules.get("max_length", 2048):
            reasons.append("Reference too long")
            return ValidationResult(is_valid=False, reasons=reasons)
    
        text_lower = entity.text.lower()
        
        # URL validation
        url_config = validation_types.get("url", {})
        url_patterns = url_config.get("patterns", {})
        
        is_url = any(re.match(pattern, text_lower) 
                     for pattern in url_patterns.values())
        
        # Social handle validation
        handle_config = validation_types.get("social_handle", {})
        handle_patterns = handle_config.get("patterns", {})
        max_handle_length = handle_config.get("max_length", 30)
        
        is_handle = (any(re.match(pattern, entity.text) 
                        for pattern in handle_patterns.values()) 
                    and len(entity.text) <= max_handle_length)
        
        # Validate URL format
        if is_url:
            if any(platform in text_lower for platform in social_platforms):
                boost = confidence_boosts.get("known_platform", 0.2)
                confidence_adjustment += boost
                reasons.append(f"Known social platform URL: +{boost}")
            elif any(domain in text_lower for domain in common_domains):
                boost = confidence_boosts.get("known_platform", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Known domain: +{boost}")
            
            if any(scheme + "://" in text_lower for scheme in url_config.get("allowed_schemes", [])):
                boost = confidence_boosts.get("has_protocol", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Valid URL protocol: +{boost}")
            
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
        
        # Validate social media handle
        elif is_handle:
            # Look for social context in surrounding text
            platform_names = [p.split('.')[0] for p in social_platforms]
            has_social_context = any(platform in text.lower() for platform in platform_names)
            
            if has_social_context:
                boost = confidence_boosts.get("known_platform", 0.2)
                confidence_adjustment += boost
                reasons.append(f"Social media context: +{boost}")
                
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
        
        # If it starts with @ but doesn't match handle pattern, it's invalid
        if entity.text.startswith('@'):
            return ValidationResult(
                is_valid=False,
                reasons=["Invalid social media handle format"]
            )
            
        return ValidationResult(
            is_valid=False,
            reasons=["Not a valid URL or social media handle"]
        )

###############################################################################
# SECTION 7: PROTECTED CLASS VALIDATION
###############################################################################

    def _get_identity_contexts(self, categories: Dict) -> List[str]:
        """
        Returns a merged list of context phrases from identity and organization keys.
        Leadership roles from religious patterns are appended last.
        """
        contexts = []
        if "gender_patterns" in categories:
            contexts.extend(categories["gender_patterns"].get("identity_context", []))
            contexts.extend(categories["gender_patterns"].get("org_context", []))
        if "religious_patterns" in categories:
            contexts.extend(categories["religious_patterns"].get("identity_context", []))
            contexts.extend(categories["religious_patterns"].get("org_context", []))
            contexts.extend(categories["religious_patterns"].get("leadership_roles", []))
        return contexts
    
    def _check_identity_context(self, text: str, identity_term: str, contexts: List[str],
                                boosts: Dict, reasons: List[str], confidence_adjustment: float) -> Tuple[bool, float]:
        """
        Check if the provided text contains any valid identity or organizational context.
        
        First, it directly checks if any organizational keyword (e.g. "organization", "church", etc.)
        appears in the context text. If so, it immediately adds the boost from "org_context_match".
        
        Otherwise, it loops over the merged contexts. For each context phrase, it checks if the phrase 
        appears as an exact substring in the context text. If not, it tokenizes the context and checks 
        each token using a common-prefix rule (with a minimum of 4 characters). Organizational phrases 
        (or those equal to "leader", "leaders", or "leadership") use the boost from "org_context_match";
        otherwise, the boost from "gender_context_match" is used.
        
        Returns:
            A tuple (found: bool, updated_confidence_adjustment: float)
        """
        text_lower = text.lower()
        org_keywords = ["organization", "church", "network", "association", "society"]
        
        # Direct check for any organizational keyword in the text.
        for org_kw in org_keywords:
            if org_kw in text_lower:
                boost = boosts.get("org_context_match", 0.3)
                reasons.append(f"Organizational keyword '{org_kw}' found in context: +{boost}")
                confidence_adjustment += boost
                return True, confidence_adjustment
    
        # Otherwise, use the merged context phrases.
        words = text_lower.split()
        for ctx in contexts:
            ctx_lower = ctx.lower()
            if ctx_lower in text_lower:
                if any(org_kw in ctx_lower for org_kw in org_keywords) or ctx_lower in ["leader", "leaders", "leadership"]:
                    boost = boosts.get("org_context_match", 0.3)
                    reasons.append(f"Organizational context match: +{boost}")
                else:
                    boost = boosts.get("gender_context_match", 0.3)
                    reasons.append(f"Identity context match: +{boost}")
                confidence_adjustment += boost
                return True, confidence_adjustment
            for w in words:
                if len(ctx_lower) >= 4 and len(w) >= 4 and w[:4] == ctx_lower[:4]:
                    if any(org_kw in ctx_lower for org_kw in org_keywords) or ctx_lower in ["leader", "leaders", "leadership"]:
                        boost = boosts.get("org_context_match", 0.3)
                        reasons.append(f"Organizational context match (via prefix): +{boost}")
                    else:
                        boost = boosts.get("gender_context_match", 0.3)
                        reasons.append(f"Identity context match (via prefix): +{boost}")
                    confidence_adjustment += boost
                    return True, confidence_adjustment
        return False, confidence_adjustment
    
    def _check_pronoun_patterns(self, text: str, patterns: Dict,
                                  term_match_score: float,
                                  reasons: List[str]) -> ValidationResult:
        """Check pronoun patterns in text."""
        for pattern_name, pattern in patterns.items():
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    reasons.append(f"Pronoun pattern match ({pattern_name}): +{term_match_score}")
                    return ValidationResult(
                        is_valid=True,
                        confidence_adjustment=term_match_score,
                        reasons=reasons
                    )
            except re.error:
                if self.debug_mode:
                    self.logger.warning(f"Invalid regex pattern: {pattern}")
        return ValidationResult(is_valid=False)
    
    def _find_identity_term(self, text: str, categories: Dict,
                              reasons: List[str]) -> Optional[str]:
        """Find a matching identity term in text."""
        all_terms = (
            categories.get("gender_patterns", {}).get("gender_terms", []) +
            categories.get("religious_patterns", {}).get("religious_terms", []) +
            categories.get("orientation", []) +
            categories.get("race_ethnicity", [])
        )
        text_lower = text.lower()
        for term in all_terms:
            if term.lower() in text_lower:
                reasons.append(f"Found identity term: {term}")
                return term
        return None
    
    def _validate_protected_class(self, entity: Entity, text: str,
                                  rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate protected class entities using linguistic analysis.
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        categories = rules.get("categories", {})
    
        term_match_score = validation_rules.get("term_match_score", 0.3)
        context_match_score = validation_rules.get("context_match_score", 0.3)
        
        reasons = []
        confidence_adjustment = 0.0
    
        # First, check pronoun patterns.
        pronoun_result = self._check_pronoun_patterns(
            entity.text,
            categories.get("pronoun_patterns", {}),
            term_match_score,
            reasons
        )
        if pronoun_result.is_valid:
            return pronoun_result
    
        # Next, find an identity term.
        identity_term = self._find_identity_term(
            entity.text,
            categories,
            reasons
        )
        if identity_term:
            identity_contexts = self._get_identity_contexts(categories)
            found, confidence_adjustment = self._check_identity_context(
                text,
                identity_term,
                identity_contexts,
                confidence_boosts,
                reasons,
                confidence_adjustment
            )
            if found:
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=confidence_adjustment,
                    reasons=reasons
                )
    
        penalty = rules.get("confidence_penalties", {}).get("no_context", -0.4)
        confidence_adjustment += penalty
        reasons.append(f"No protected class context: {penalty}")
        return ValidationResult(
            is_valid=False,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )


###############################################################################
# SECTION 8: PHI VALIDATION
###############################################################################

    def _validate_phi(self, entity: Entity, text: str,
                      rules: Dict, source: str = None) -> ValidationResult:
        """Route PHI validation to appropriate detector method.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
            
        Returns:
            ValidationResult containing validation outcome and confidence adjustments
        """
        if source == "spacy":
            return self._validate_spacy_phi(entity, text, rules)
        elif source == "presidio": 
            return self._validate_presidio_phi(entity, text, rules)
            
        # Default ensemble validation
        ensemble_config = self._get_ensemble_config()
        presidio_weight = ensemble_config.get("presidio_weight", 0.5)
        spacy_weight = ensemble_config.get("spacy_weight", 0.5)
        
        presidio_result = self._validate_presidio_phi(entity, text, rules)
        spacy_result = self._validate_spacy_phi(entity, text, rules)
        
        # Take highest confidence result
        if (presidio_result.confidence_adjustment * presidio_weight >= 
            spacy_result.confidence_adjustment * spacy_weight):
            return presidio_result
        return spacy_result

#####################
# SPACY VALIDATION
#####################

    def _validate_spacy_phi(self, entity: Entity, text: str, rules: Dict) -> ValidationResult:
        """SpaCy-focused PHI validation using medical terminology matching."""
        if not entity or not entity.text or not text:
            return ValidationResult(is_valid=False, reasons=["Missing required input"])
            
        confidence_boosts = rules.get("confidence_boosts", {})
        categories = rules.get("categories", {})
        reasons = []
        confidence_adjustment = 0.0
        
        text_lower = entity.text.lower()
        context_lower = text.lower()
        
        # Check medical terminology against required context words
        context_requirements = rules.get("generic_term_context_requirements", {})
        context_words = context_requirements.get("required_context_words", [])
        min_required = context_requirements.get("minimum_context_words", 2)
        
        # Split on common delimiters to catch terms in different formats
        context_tokens = re.split(r'[:\s,]+', context_lower)
        matching_contexts = [word for word in context_words if word in context_tokens]
        
        if len(matching_contexts) >= min_required:
            boost = confidence_boosts.get("has_medical_context", 0.4)
            confidence_adjustment += boost
            reasons.append(f"Medical context terms found: {', '.join(matching_contexts[:2])}: +{boost}")
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
        
        # Check for specific conditions
        medical_conditions = categories.get("medical_conditions", {})
        pattern_result = self._check_medical_patterns(entity.text, text, medical_conditions, 
                                                    confidence_boosts, reasons, confidence_adjustment)
        if pattern_result.is_valid:
            return pattern_result
            
        # Check ADA accommodations
        ada_patterns = categories.get("disabilities", {}).get("ada_related", [])
        ada_result = self._check_ada_accommodations(text, ada_patterns, confidence_boosts, 
                                                  reasons, confidence_adjustment, rules)
        if ada_result.is_valid:
            return ada_result
            
        # Check veteran disability status
        vet_result = self._check_veteran_disability(text, categories.get("disabilities", {}),
                                                  confidence_boosts, reasons, confidence_adjustment)
        if vet_result.is_valid:
            return vet_result
        
        # Get treatments if no conditions matched
        treatments = []
        for treatment_type in categories.get("treatments", {}).values():
            treatments.extend(treatment_type)
            
        # Check if text contains treatment and context has condition terms
        is_treatment = any(t.lower() in text_lower for t in treatments)
        has_condition = any(word in context_lower for word in context_words 
                          if word in ['condition', 'diagnosis', 'diagnosed'])
        
        if is_treatment and has_condition:
            boost = confidence_boosts.get("treatment_context_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Treatment with condition context: +{boost}")
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)

        penalty = rules.get("confidence_penalties", {}).get("no_context", -0.7)
        confidence_adjustment += penalty
        reasons.append(f"No medical context: {penalty}")
        return ValidationResult(is_valid=False, confidence_adjustment=confidence_adjustment, reasons=reasons)
        

#####################
# PRESIDIO VALIDATION
#####################

    def _validate_presidio_phi(self, entity: Entity, text: str, rules: Dict) -> ValidationResult:
        """Presidio-focused PHI validation using pattern matching."""
        if not entity or not entity.text or not text:
            return ValidationResult(is_valid=False, reasons=["Missing required input"])
            
        confidence_boosts = rules.get("confidence_boosts", {})
        categories = rules.get("categories", {})
        reasons = []
        confidence_adjustment = 0.0
        
        text_lower = entity.text.lower()
        context_lower = text.lower()
        
        # Check specific condition patterns from config
        condition_patterns = rules.get("patterns", {}).get("specific_conditions", {}).get("regex", [])
        for pattern in condition_patterns:
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                boost = confidence_boosts.get("condition_specific_match", 0.5)
                confidence_adjustment += boost 
                reasons.append(f"Matched medical condition pattern: +{boost}")
                return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)

        # Check for disability percentage patterns
        if re.search(r'\d+%\s*(?:disabled|disability)', text_lower + ' ' + context_lower):
            boost = confidence_boosts.get("va_disability_match", 0.4)
            confidence_adjustment += boost
            reasons.append(f"Disability percentage match: +{boost}")
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)

        # Check ADA accommodation patterns
        ada_patterns = categories.get("disabilities", {}).get("ada_related", [])
        ada_result = self._check_ada_accommodations(text, ada_patterns, confidence_boosts, 
                                                  reasons, confidence_adjustment, rules)
        if ada_result.is_valid:
            return ada_result

        # Check veteran disability patterns
        vet_result = self._check_veteran_disability(text, categories.get("disabilities", {}), 
                                                  confidence_boosts, reasons, confidence_adjustment)
        if vet_result.is_valid:
            return vet_result
                
        # Check treatment patterns from config
        treatment_patterns = rules.get("patterns", {}).get("treatment_keywords", {}).get("regex", [])
        for pattern in treatment_patterns:
            if re.search(pattern, text_lower, flags=re.IGNORECASE):
                boost = confidence_boosts.get("treatment_context_match", 0.4)
                confidence_adjustment += boost
                reasons.append(f"Treatment pattern match: +{boost}")
                return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)

        # Check required medical context terms
        context_requirements = rules.get("generic_term_context_requirements", {})
        context_words = context_requirements.get("required_context_words", [])
        min_required = context_requirements.get("minimum_context_words", 2)
        
        matching_contexts = [word for word in context_words if word.lower() in context_lower]
        if len(matching_contexts) >= min_required:
            boost = confidence_boosts.get("has_medical_context", 0.4)
            confidence_adjustment += boost
            context_type = "disability" if "disability" in text_lower else "medical"
            reasons.append(f"{context_type.capitalize()} context terms found ({', '.join(matching_contexts[:2])}): +{boost}")
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)

        # If no valid context found, apply penalty
        penalty = rules.get("confidence_penalties", {}).get("no_context", -0.7)
        confidence_adjustment += penalty
        reasons.append(f"No medical context: {penalty}")
        return ValidationResult(is_valid=False, confidence_adjustment=confidence_adjustment, reasons=reasons)
        

#####################
# HELPER FUNCTIONS
#####################
    
    def _validate_medical_context(self, text: str, term: str,
                                  context_rules: Dict, reasons: List[str],
                                  confidence_adjustment: float) -> ValidationResult:
        """Validate medical term context requirements."""
        required_words = set(context_rules.get("required_context_words", []))
        min_required = context_rules.get("minimum_context_words", 2)
        
        text_lower = text.lower()
        context_matches = sum(1 for word in required_words if word in text_lower)
        
        if context_matches < min_required:
            penalty = rules.get("confidence_penalties", {}).get("no_context", -0.7)
            reasons.append(f"Insufficient medical context: {penalty}")
            return ValidationResult(is_valid=False, confidence_adjustment=penalty, reasons=reasons)
                
        return ValidationResult(is_valid=True)
    
    def _check_medical_patterns(self, entity_text: str, full_text: str,
                                  conditions: Dict, boosts: Dict,
                                  reasons: List[str], confidence_adjustment: float) -> ValidationResult:
        """Check medical condition patterns."""
        all_conditions = {
            condition.lower() 
            for category in conditions.values() 
            for condition in category
        }
        
        if self.debug_mode:
            self.logger.debug(f"Checking {len(all_conditions)} medical conditions")
                
        for condition in all_conditions:
            if condition in entity_text.lower():
                boost = boosts.get("condition_specific_match", 0.4)
                confidence_adjustment += boost
                reasons.append(f"Medical condition match ({condition}): +{boost}")
                return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
                    
        return ValidationResult(is_valid=False)


    def _check_ada_accommodations(self, text: str, ada_patterns: List[str],
                                  boosts: Dict, reasons: List[str],
                                  confidence_adjustment: float, rules: Dict) -> ValidationResult:
        """Check for ADA accommodation patterns."""
        text_lower = text.lower()
        # First check configured patterns
        for pattern in ada_patterns:
            if pattern in text_lower:
                boost = boosts.get("has_medical_context", 0.4)
                reasons.append(f"ADA accommodation pattern: +{boost}")
                confidence_adjustment += boost
                return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
    
        # Check generic ADA terms using categories from config
        disability_config = rules.get("categories", {}).get("disabilities", {})
        accommodation_terms = set(disability_config.get("accommodation_terms", []))
        context_terms = set(disability_config.get("context_terms", []))
        
        words = set(text_lower.split())
        if any(a in words for a in accommodation_terms) and any(c in words for c in context_terms):
            boost = boosts.get("has_medical_context", 0.4)
            reasons.append(f"Accommodation context pattern: +{boost}")
            confidence_adjustment += boost
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
                
        return ValidationResult(is_valid=False)


    def _check_veteran_disability(self, text: str, disabilities: Dict,
                                  boosts: Dict, reasons: List[str],
                                  confidence_adjustment: float) -> ValidationResult:
        """
        Check for veteran disability patterns in the context text.
        
        First checks for disability and veteran term combinations, then looks
        for percentage-based patterns and VA status markers from config.
        """
        text_lower = text.lower()
        tokens = text_lower.split()
        
        # Get veteran terms from config
        vet_terms = disabilities.get("veteran_terms", ["vet", "veteran"])
        disability_terms = disabilities.get("disability_terms", ["disabled", "disability"])
        
        if (any(term in tokens for term in disability_terms) and 
            any(term in tokens for term in vet_terms)):
            boost = boosts.get("va_disability_match", 0.4)
            reasons.append(f"Veteran disability context detected (token-based): +{boost}")
            confidence_adjustment += boost
            return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
        
        percent_match = re.search(r'(\d+)\s*%', text_lower)
        if percent_match:
            veteran_patterns = disabilities.get("veteran_status", [])
            for pattern in veteran_patterns:
                if pattern in text_lower:
                    boost = boosts.get("va_disability_match", 0.4)
                    reasons.append(f"VA disability pattern: +{boost}")
                    confidence_adjustment += boost
                    return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
            va_status_config = disabilities.get("va_status_markers", {})
            va_status_terms = va_status_config.get("rating_terms", [])
            status_boost = va_status_config.get("context_boost", 0.3)
            for status in va_status_terms:
                if status in text_lower:
                    reasons.append(f"VA status context: +{status_boost}")
                    confidence_adjustment += status_boost
                    return ValidationResult(is_valid=True, confidence_adjustment=confidence_adjustment, reasons=reasons)
        
        return ValidationResult(is_valid=False, confidence_adjustment=confidence_adjustment, reasons=reasons)
###############################################################################
# SECTION 9: VALIDATION UTILITIES AND HELPERS
###############################################################################


    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        if not text:
            return ""
            
        # Basic normalization
        normalized = text.lower().strip()
        
        # Remove punctuation except preserved characters
        normalized = ''.join(
            c for c in normalized 
            if c.isalnum() or c in '.-_ '
        )
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized

    def _get_context_window(self, text: str, start: int, end: int,
                           window_size: int = 50) -> str:
        """Get text context window around position."""
        start = max(0, start - window_size)
        end = min(len(text), end + window_size)
        return text[start:end]

    def _analyze_linguistic_features(self, text: str,
                                  patterns: List[str],
                                  terms: List[str]) -> Tuple[bool, List[str]]:
        """
        Analyze linguistic features in text.
        
        Args:
            text: Text to analyze
            patterns: Regex patterns to check
            terms: Terms to look for
            
        Returns:
            Tuple of (match_found, reasons)
        """
        reasons = []
        text_lower = text.lower()
        
        # Check patterns
        for pattern in patterns:
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    reasons.append(f"Matched pattern: {pattern}")
                    return True, reasons
            except re.error:
                if self.debug_mode:
                    self.logger.warning(f"Invalid regex pattern: {pattern}")
                    
        # Check terms
        for term in terms:
            if term.lower() in text_lower:
                reasons.append(f"Found term: {term}")
                return True, reasons
                
        return False, reasons

###############################################################################
# End of ValidationCoordinator
###############################################################################