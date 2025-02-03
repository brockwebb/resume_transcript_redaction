# redactor/validation/validation_coordinator.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set, Tuple
import re
import logging
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
        # Get all entity types from routing
        entity_types = set()
        for section in ["presidio_primary", "spacy_primary", "ensemble_required"]:
            if section in self.routing_config:
                entity_types.update(self.routing_config[section].get("entities", []))
        
        # Map entity types to validation methods if they have config
        for entity_type in entity_types:
            if entity_type in self.validation_params:
                method_name = f"_validate_{entity_type.lower()}"
                if hasattr(self, method_name):
                    self.validation_methods[entity_type] = getattr(self, method_name)
                    if self.debug_mode:
                        self.logger.debug(f"Mapped {entity_type} to {method_name}")

    def validate_entity(self, 
                       entity: Entity,
                       text: str,
                       source: str = None) -> ValidationResult:
        """
        Validate entity using configuration-driven rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            source: Optional detector source (for specific rules)
            
        Returns:
            ValidationResult with validation outcome
        """
        if not entity or not entity.text:
            return ValidationResult(is_valid=False, reasons=["Empty entity"])
            
        # Check if we have validation method for this type
        if entity.entity_type not in self.validation_methods:
            if self.debug_mode:
                self.logger.debug(f"No validation method for type: {entity.entity_type}")
            return ValidationResult(is_valid=True)  # Pass through if no validation
            
        # Get validation method and rules
        validator = self.validation_methods[entity.type]
        validation_rules = self.validation_params.get(entity.entity_type, {})
        
        # Validate with config-driven rules
        result = validator(entity, text, validation_rules, source)
        
        if self.debug_mode:
            self.logger.debug(
                f"Validated {entity.entity_type}: '{entity.text}' - "
                f"Valid: {result.is_valid}, Adjustment: {result.confidence_adjustment}"
            )
            for reason in result.reasons:
                self.logger.debug(f"  Reason: {reason}")
                
        return result

###############################################################################
# SECTION 3: CORE VALIDATION METHODS
###############################################################################

    def _validate_educational(self, entity: Entity, text: str, 
                            rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate educational institution using config-driven rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
            
        Returns:
            ValidationResult with validation outcome
        """
        confidence_boosts = rules.get("confidence_boosts", {})
        confidence_penalties = rules.get("confidence_penalties", {})
        core_terms = set(rules.get("core_terms", []))
        known_acronyms = set(rules.get("known_acronyms", []))
        min_words = rules.get("min_words", 2)
        generic_terms = set(rules.get("generic_terms", []))
        
        text_lower = entity.text.lower()
        words = text_lower.split()
        reasons = []
        confidence_adjustment = 0.0
        
        # Single-word validation
        if len(words) < min_words:
            # Check for known acronym
            clean_text = entity.text.strip().upper()
            if self.debug_mode:
                self.logger.debug(
                    f"Checking acronym: '{clean_text}' against known_acronyms={known_acronyms}"
                )
            
            if clean_text in known_acronyms:
                boost = confidence_boosts.get("known_acronym", 0.4)
                confidence_adjustment += boost
                reasons.append(f"Known acronym match: +{boost}")
            else:
                penalty = confidence_penalties.get("single_word", -0.3)
                confidence_adjustment += penalty
                reasons.append(f"Single word penalty: {penalty}")
                return ValidationResult(
                    is_valid=False,
                    confidence_adjustment=confidence_adjustment,
                    reasons=reasons
                )
        
        # Check for generic terms
        if any(term in text_lower for term in generic_terms):
            penalty = confidence_penalties.get("generic_term", -1.0)
            confidence_adjustment += penalty
            reasons.append(f"Generic term penalty: {penalty}")
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
        
        # Check for core terms
        has_core_term = any(term in text_lower for term in core_terms)
        if not has_core_term:
            penalty = confidence_penalties.get("no_core_terms", -1.0)
            confidence_adjustment += penalty
            reasons.append(f"Missing core terms: {penalty}")
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
        
        # Apply caps penalty if not a known acronym
        if entity.text.isupper() and entity.text not in known_acronyms:
            penalty = confidence_penalties.get("improper_caps", -0.2)
            confidence_adjustment += penalty
            reasons.append(f"Improper capitalization: {penalty}")
        
        # Apply length boost
        if has_core_term and len(words) >= min_words:
            boost = confidence_boosts.get("full_name", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Full institution name: +{boost}")
        
        if self.debug_mode:
            self.logger.debug(
                f"Educational validation result for '{entity.text}':"
                f"\n  Valid: True"
                f"\n  Confidence Adjustment: {confidence_adjustment}"
                f"\n  Reasons: {reasons}"
            )
        
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

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
# SECTION 4: PERSON AND LOCATION VALIDATION
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

    def _validate_location(self, entity: Entity, text: str,
                          rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate location entities using config rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
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
        formats_matched = []

        # Basic validation
        if len(entity.text) < validation_rules.get("min_chars", 2):
            reasons.append("Location too short")
            return ValidationResult(is_valid=False, reasons=reasons)

        # Capitalization check
        if validation_rules.get("require_capitalization", True):
            first_char = entity.text[0]
            if not (first_char.isupper() or first_char != first_char.lower()):
                reasons.append("Location not properly capitalized")
                return ValidationResult(is_valid=False, reasons=reasons)

        # Check format patterns
        self._check_location_formats(
            entity, formats_matched, format_patterns, 
            confidence_boosts, reasons, confidence_adjustment
        )

        # Handle single-word locations
        if len(words) == 1:
            if not validation_rules.get("allow_single_word", True):
                reasons.append("Single word locations not allowed")
                return ValidationResult(is_valid=False, reasons=reasons)

            if entity.text.isupper() and validation_rules.get("allow_all_caps", True):
                boost = confidence_boosts.get("proper_case", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Valid ALL CAPS location: +{boost}")
            elif not formats_matched:
                penalty = confidence_penalties.get("single_word", -0.2)
                confidence_adjustment += penalty
                reasons.append(f"Single word penalty: {penalty}")

        # Multi-word boost
        elif not formats_matched:
            boost = confidence_boosts.get("multi_word", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Multi-word location: +{boost}")

        # Check for US state if not already matched
        if "us_city_state" not in formats_matched:
            state_lists = [
                s.lower() for s in us_states.get("full", []) + us_states.get("abbrev", [])
            ]
            if any(state.lower() in text_lower for state in state_lists):
                boost = confidence_boosts.get("has_state", 0.2)
                confidence_adjustment += boost
                reasons.append(f"Contains US state: +{boost}")

        if not formats_matched and not reasons:
            reasons.append("Basic validation passed but no specific format matched")

        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

    def _check_location_formats(self, entity: Entity, formats_matched: List[str],
                              format_patterns: Dict, confidence_boosts: Dict,
                              reasons: List[str], confidence_adjustment: float) -> None:
        """
        Check location formats using config patterns.
        
        Args:
            entity: Entity to validate
            formats_matched: List to track matched formats
            format_patterns: Format patterns from config
            confidence_boosts: Confidence boosts from config
            reasons: List to store validation reasons
            confidence_adjustment: Running confidence adjustment
        """
        # US city-state format
        us_pattern = format_patterns.get("us_city_state", {}).get("regex", "")
        if us_pattern and re.match(us_pattern, entity.text):
            formats_matched.append("us_city_state")
            base_boost = confidence_boosts.get("known_format", 0.1)
            state_boost = confidence_boosts.get("has_state", 0.2)
            total_boost = base_boost + state_boost
            confidence_adjustment += total_boost
            reasons.append(f"Matches US city-state format: +{total_boost}")

            # Check for ZIP code
            if re.search(r'\d{5}(?:-\d{4})?$', entity.text):
                zip_boost = confidence_boosts.get("has_zip", 0.1)
                confidence_adjustment += zip_boost
                reasons.append(f"Includes ZIP code: +{zip_boost}")
                formats_matched.append("zip_code")

        # International format
        elif not formats_matched:
            intl_pattern = format_patterns.get("international", {}).get("regex", "")
            if intl_pattern and re.match(r'^[A-ZÀ-ÿ][A-Za-zÀ-ÿ\s-]*$', entity.text):
                formats_matched.append("international")
                boost = confidence_boosts.get("known_format", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Matches international format: +{boost}")

        # Metropolitan area format
        elif not formats_matched:
            metro_pattern = format_patterns.get("metro_area", {}).get("regex", "")
            if metro_pattern and re.match(metro_pattern, entity.text):
                formats_matched.append("metro_area")
                boost = confidence_boosts.get("known_format", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Matches metropolitan area format: +{boost}")

###############################################################################
# SECTION 5: SPECIAL ENTITY VALIDATION
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
        known_domains = set(rules.get("known_domains", []))
        patterns = rules.get("patterns", {})
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Length validation
        if len(entity.text) < validation_rules.get("min_chars", 5):
            reasons.append("Email too short")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        if len(entity.text) > validation_rules.get("max_length", 254):
            reasons.append("Email too long")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        # Basic format check
        basic_pattern = patterns.get("basic", "")
        if not basic_pattern or not re.match(basic_pattern, entity.text):
            reasons.append("Invalid email format")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        # Known domain boost
        domain = entity.text.split('@')[-1].lower()
        if any(domain.endswith(d) for d in known_domains):
            boost = confidence_boosts.get("known_domain", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Known domain: +{boost}")
            
        # Context boost
        context_words = {"email", "contact", "mail", "address", "@"}
        if any(word in text.lower() for word in context_words):
            boost = confidence_boosts.get("has_context", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Email context: +{boost}")
            
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

    def _validate_phone(self, entity: Entity, text: str,
                       rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate phone number using config rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        context_words = set(rules.get("context_words", []))
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Extract digits
        digits = ''.join(c for c in entity.text if c.isdigit())
        
        # Length validation
        min_digits = validation_rules.get("min_digits", 10)
        max_digits = validation_rules.get("max_digits", 15)
        
        if len(digits) < min_digits:
            reasons.append("Too few digits")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        if len(digits) > max_digits:
            reasons.append("Too many digits")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        # Format boost
        formats = [
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',  # (123) 456-7890
            r'\d{3}[-.]?\d{3}[-.]?\d{4}',    # 123-456-7890
        ]
        
        if any(re.match(pattern, entity.text) for pattern in formats):
            boost = confidence_boosts.get("format_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Standard format: +{boost}")
            
        # Context boost
        if any(word in text.lower() for word in context_words):
            boost = confidence_boosts.get("has_context", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Phone context: +{boost}")
            
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
        known_platforms = rules.get("known_platforms", {})
        
        social_platforms = set(known_platforms.get("social", []))
        common_domains = set(known_platforms.get("common_domains", []))
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Length validation
        if len(entity.text) < validation_rules.get("min_chars", 4):
            reasons.append("Reference too short")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        if len(entity.text) > validation_rules.get("max_length", 2048):
            reasons.append("Reference too long")
            return ValidationResult(is_valid=False, reasons=reasons)
            
        # Known platform boost
        text_lower = entity.text.lower()
        if any(platform in text_lower for platform in social_platforms):
            boost = confidence_boosts.get("known_platform", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Known social platform: +{boost}")
            
        elif any(domain in text_lower for domain in common_domains):
            boost = confidence_boosts.get("known_platform", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Known domain: +{boost}")
            
        # Context boost
        if any(word in text.lower() for word in ["http", "https", "www", "@"]):
            boost = confidence_boosts.get("has_protocol", 0.1)
            confidence_adjustment += boost
            reasons.append(f"URL/handle context: +{boost}")
            
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

###############################################################################
# SECTION 6: PROTECTED CLASS AND PHI VALIDATION
###############################################################################

    def _validate_protected_class(self, entity: Entity, text: str,
                                rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate protected class entities using linguistic analysis.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        categories = rules.get("categories", {})
        
        term_match_score = validation_rules.get("term_match_score", 0.3)
        context_match_score = validation_rules.get("context_match_score", 0.3)
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Check pronoun patterns first
        pronoun_result = self._check_pronoun_patterns(
            entity.text,
            categories.get("pronoun_patterns", {}),
            term_match_score,
            reasons
        )
        if pronoun_result.is_valid:
            return pronoun_result
            
        # Check identity terms
        identity_term = self._find_identity_term(
            entity.text,
            categories,
            reasons
        )
        
        if identity_term:
            # Get valid contexts
            identity_contexts = self._get_identity_contexts(categories)
            
            # Check for context matches
            if self._check_identity_context(
                text,
                identity_term,
                identity_contexts,
                confidence_boosts,
                reasons,
                confidence_adjustment
            ):
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=confidence_adjustment,
                    reasons=reasons
                )
                
        # No valid matches found
        penalty = rules.get("confidence_penalties", {}).get("no_context", -0.4)
        confidence_adjustment += penalty
        reasons.append(f"No protected class context: {penalty}")
        
        return ValidationResult(
            is_valid=False,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

    def _validate_phi(self, entity: Entity, text: str,
                     rules: Dict, source: str = None) -> ValidationResult:
        """
        Validate PHI entities using medical context analysis.
        
        Args:
            entity: Entity to validate
            text: Original text context
            rules: Validation rules from config
            source: Optional detector source
        """
        validation_rules = rules.get("validation_rules", {})
        confidence_boosts = rules.get("confidence_boosts", {})
        categories = rules.get("categories", {})
        
        reasons = []
        confidence_adjustment = 0.0
        
        # Check always valid conditions first
        always_valid = validation_rules.get("always_valid_conditions", [])
        if any(condition.lower() in entity.text.lower() for condition in always_valid):
            boost = confidence_boosts.get("condition_specific_match", 0.5)
            confidence_adjustment += boost
            reasons.append(f"Explicit condition match: +{boost}")
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
            
        # Check for generic terms that need strict context
        generic_terms = rules.get("generic_reject_terms", [])
        if any(term in entity.text.lower() for term in generic_terms):
            context_result = self._validate_medical_context(
                text,
                entity.text,
                rules.get("generic_term_context_requirements", {}),
                reasons,
                confidence_adjustment
            )
            if not context_result.is_valid:
                return context_result
        
        # Check medical patterns
        pattern_result = self._check_medical_patterns(
            entity.text,
            text,
            categories.get("medical_conditions", {}),
            confidence_boosts,
            reasons,
            confidence_adjustment
        )
        if pattern_result.is_valid:
            return pattern_result
            
        # Check ADA accommodations
        ada_result = self._check_ada_accommodations(
            text,
            categories.get("disabilities", {}).get("ada_related", []),
            confidence_boosts,
            reasons,
            confidence_adjustment
        )
        if ada_result.is_valid:
            return ada_result
            
        # Check veteran disability
        veteran_result = self._check_veteran_disability(
            text,
            categories.get("disabilities", {}),
            confidence_boosts,
            reasons,
            confidence_adjustment
        )
        if veteran_result.is_valid:
            return veteran_result
            
        # No valid context found
        penalty = rules.get("confidence_penalties", {}).get("no_context", -0.7)
        confidence_adjustment += penalty
        reasons.append(f"No valid medical context: {penalty}")
        
        return ValidationResult(
            is_valid=False,
            confidence_adjustment=confidence_adjustment,
            reasons=reasons
        )

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
        """Find matching identity term in text."""
        # Collect all identity terms
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

    def _get_identity_contexts(self, categories: Dict) -> List[str]:
        """Get valid identity context terms from categories."""
        gender_patterns = categories.get("gender_patterns", {})
        religious_patterns = categories.get("religious_patterns", {})
        
        return (
            gender_patterns.get("identity_context", []) +
            religious_patterns.get("identity_context", []) +
            gender_patterns.get("org_context", []) +
            religious_patterns.get("org_context", []) +
            religious_patterns.get("leadership_roles", [])
        )

    def _check_identity_context(self, text: str, identity_term: str,
                              contexts: List[str], boosts: Dict,
                              reasons: List[str], confidence_adjustment: float) -> bool:
        """Check for identity term in valid contexts."""
        text_lower = text.lower()
        for context in contexts:
            if context in text_lower:
                if text_lower.count(identity_term.lower()) > 1:
                    boost = boosts.get("collective_group", 0.2)
                    confidence_adjustment += boost
                    reasons.append(f"Collective group reference: +{boost}")
                else:
                    boost = boosts.get("gender_context_match", 0.3)
                    confidence_adjustment += boost
                    reasons.append(f"Identity context match: +{boost}")
                return True
        return False

    def _validate_medical_context(self, text: str, term: str,
                                context_rules: Dict, reasons: List[str],
                                confidence_adjustment: float) -> ValidationResult:
        """Validate medical term context requirements."""
        required_words = set(context_rules.get("required_context_words", []))
        min_required = context_rules.get("minimum_context_words", 2)
        
        # Count matching context words
        text_lower = text.lower()
        context_matches = sum(1 for word in required_words if word in text_lower)
        
        if context_matches < min_required:
            penalty = -0.7  # Generic term without context
            reasons.append(f"Insufficient medical context: {penalty}")
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=penalty,
                reasons=reasons
            )
            
        return ValidationResult(is_valid=True)

###############################################################################
# SECTION 7: VALIDATION UTILITIES AND HELPERS
###############################################################################

    def _check_medical_patterns(self, entity_text: str, full_text: str,
                              conditions: Dict, boosts: Dict,
                              reasons: List[str], confidence_adjustment: float) -> ValidationResult:
        """Check medical condition patterns."""
        # Flatten condition lists
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
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=confidence_adjustment,
                    reasons=reasons
                )
                
        return ValidationResult(is_valid=False)

    def _check_ada_accommodations(self, text: str, ada_patterns: List[str],
                                boosts: Dict, reasons: List[str],
                                confidence_adjustment: float) -> ValidationResult:
        """Check for ADA accommodation patterns."""
        # Check exact patterns
        text_lower = text.lower()
        for pattern in ada_patterns:
            if pattern in text_lower:
                boost = boosts.get("has_medical_context", 0.4)
                reasons.append(f"ADA accommodation pattern: +{boost}")
                confidence_adjustment += boost
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=confidence_adjustment,
                    reasons=reasons
                )
                
        # Check term combinations
        accommodation_terms = {'accommodation', 'accommodations', 'ada', 'disability'}
        context_terms = {'need', 'needed', 'require', 'required', 'requesting', 'disability'}
        
        words = set(text_lower.split())
        if any(a in words for a in accommodation_terms) and any(c in words for c in context_terms):
            boost = boosts.get("has_medical_context", 0.4)
            reasons.append(f"Accommodation context pattern: +{boost}")
            confidence_adjustment += boost
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=confidence_adjustment,
                reasons=reasons
            )
            
        return ValidationResult(is_valid=False)

    def _check_veteran_disability(self, text: str, disabilities: Dict,
                                boosts: Dict, reasons: List[str],
                                confidence_adjustment: float) -> ValidationResult:
        """Check for veteran disability patterns."""
        veteran_status = disabilities.get("veteran_status", [])
        va_status_config = disabilities.get("va_status_markers", {})
        va_status_terms = va_status_config.get("rating_terms", [])
        status_boost = va_status_config.get("context_boost", 0.3)
        
        text_lower = text.lower()
        
        # Check for percentage patterns
        percent_match = re.search(r'(\d+)\s*%', text)
        if percent_match:
            # Look for disability terms in context
            for status in va_status_terms:
                if status in text_lower:
                    boost = status_boost
                    reasons.append(f"VA status context: +{boost}")
                    confidence_adjustment += boost
                    return ValidationResult(
                        is_valid=True,
                        confidence_adjustment=confidence_adjustment,
                        reasons=reasons
                    )
                    
            # Check veteran status patterns
            for pattern in veteran_status:
                if pattern in text_lower:
                    boost = boosts.get("va_disability_match", 0.4)
                    reasons.append(f"VA disability pattern: +{boost}")
                    confidence_adjustment += boost
                    return ValidationResult(
                        is_valid=True,
                        confidence_adjustment=confidence_adjustment,
                        reasons=reasons
                    )
                    
        return ValidationResult(is_valid=False)

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