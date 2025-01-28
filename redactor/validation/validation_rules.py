# redactor/validation/validation_rules.py

import logging
import json
from pathlib import Path
from typing import Optional
import unicodedata
import re

from redactor.detectors.base_detector import Entity


class ValidationResult:
    """
    Encapsulates the result of validating a single entity:
      - is_valid: Whether the entity is considered valid
      - confidence_adjustment: How much to adjust the entity's confidence
      - reason: String summarizing why the validation passed or failed
    """
    def __init__(self, is_valid: bool, confidence_adjustment: float = 0.0, reason: Optional[str] = None):
        self.is_valid = is_valid
        self.confidence_adjustment = confidence_adjustment
        self.reason = reason


class EntityValidationRules:
    """
    A class holding validation logic for various entity types, especially EDUCATIONAL_INSTITUTION.
    Loads config from redactor/config/validation_params.json.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        # Use the provided logger or fallback to standard library logger
        self.logger = logger if logger else logging.getLogger(__name__)

        # Load validation parameters (JSON) from redactor/config/validation_params.json
        config_path = Path("redactor/config/validation_params.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
                self.logger.debug(f"Loaded validation params from {config_path}")
        except Exception as e:
            # If config fails to load, raise an error (fallback logic will happen upstream if needed)
            self.logger.error("Failed to load validation params: %s", e)
            raise

        self.logger.debug("Validation rules initialized successfully")

    @staticmethod
    def remove_accents(s: str) -> str:
        """
        Normalize and remove diacritical marks (accents), e.g.:
        'École Polytechnique' -> 'Ecole Polytechnique'.
        Helps match 'ecole' or 'polytechnique' as core terms.
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    def validate_educational_institution(self, entity: Entity, context: str) -> ValidationResult:
        """
        Validate an entity of type 'EDUCATIONAL_INSTITUTION' using logic
        from self.params["educational"] in validation_params.json.
    
        Highlights:
          1) If it's a known acronym (e.g. 'MIT'), skip single-word penalty and boost confidence.
          2) If it's single-word but not a known acronym, penalize.
          3) If it contains generic terms, penalize.
          4) If no core terms (like 'university', 'school') are present, penalize.
          5) Otherwise, apply a +0.2 boost for multi-word recognized institutions.
    
        :param entity: The Entity object (text, confidence, etc.).
        :param context: Additional context text (unused here, but available if needed).
        :return: ValidationResult indicating final validity, confidence adjustment, and reasons.
        """
        # ----------------------------
        #  Load config
        # ----------------------------
        edu_config = self.params.get("educational", {})
        core_terms = set(edu_config.get("core_terms", []))  # e.g. ["university", "college", "institute", "school"]
        known_acronyms = set(edu_config.get("known_acronyms", []))  # e.g. ["MIT", "UCLA", ...]
        confidence_boosts = edu_config.get("confidence_boosts", {})  # e.g. {"full_name": 0.3, "known_acronym": 0.4}
        confidence_penalties = edu_config.get("confidence_penalties", {})  # e.g. {"no_core_terms": -1.0, ...}
        generic_terms = set(edu_config.get("generic_terms", []))  # e.g. ["academy of learning", ...]
        min_words = edu_config.get("min_words", 2)
    
        text_str = entity.text.strip()             # remove leading/trailing spaces
        text_lower = text_str.lower()
        text_noaccents = self.remove_accents(text_lower)
        reasons = []
        words = text_str.split()
    
        # ----------------------------
        # 1) Known Acronym Check (Skip single-word penalty)
        # ----------------------------
        # Compare uppercase to handle 'mit', 'Mit', etc.
        if text_str.upper() in known_acronyms:
            boost = confidence_boosts.get("known_acronym", 0.4)
            entity.confidence += boost
            reasons.append(f"Boost for known acronym '{text_str.upper()}': {boost}")
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=boost,
                reason="; ".join(reasons)
            )
    
        # ----------------------------
        # 2) Single-word penalty => invalid if not known acronym
        # ----------------------------
        if len(words) < min_words:
            penalty = confidence_penalties.get("single_word", -0.3)
            entity.confidence += penalty
            reasons.append(f"Penalty for single word: {penalty}")
            return ValidationResult(False, penalty, "; ".join(reasons))
    
        # ----------------------------
        # 3) Generic terms => invalid
        # ----------------------------
        for gterm in generic_terms:
            # Compare ignoring accents and case
            gterm_noaccents = self.remove_accents(gterm.lower())
            if gterm_noaccents in text_noaccents:
                penalty = confidence_penalties.get("generic_term", -1.0)
                entity.confidence += penalty
                reasons.append(f"Penalty for generic term '{gterm}': {penalty}")
                return ValidationResult(False, penalty, "; ".join(reasons))
    
        # ----------------------------
        # 4) Missing core terms => invalid
        # ----------------------------
        # Also allow "ecole", "polytechnique" for accented schools
        extended_core_terms = set(core_terms) | {"ecole", "polytechnique"}
        if not any(term in text_noaccents for term in extended_core_terms):
            penalty = confidence_penalties.get("no_core_terms", -1.0)
            entity.confidence += penalty
            reasons.append(f"Penalty for missing core terms: {penalty}")
            return ValidationResult(False, penalty, "; ".join(reasons))
    
        # ----------------------------
        # 5) If we see at least one core term => +0.2
        # ----------------------------
        boost = 0.2
        entity.confidence += boost
        reasons.append(f"Core term found => Boost: {boost}")
    
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=boost,
            reason="; ".join(reasons)
        )

    def validate_date(self, entity: Entity, context: str) -> ValidationResult:
        """Validate a DATE_TIME entity using context-aware rules."""
        text_str = entity.text.strip()
        text_lower = text_str.lower()
        confidence_adj = 0.0
        reasons = []
        
        # Load date-specific params
        date_config = self.params.get("date", {})
        patterns = date_config.get("patterns", {})
        false_positive_contexts = set(date_config.get("false_positive_contexts", []))
        year_markers = set(date_config.get("year_markers", []))
        current_terms = set(date_config.get("current_terms", []))
        adjustments = date_config.get("confidence_adjustments", {})
    
        # Check against patterns first
        matched_pattern = None
        for pattern_name, pattern_info in patterns.items():
            if re.match(pattern_info["regex"], text_lower, re.IGNORECASE):
                matched_pattern = pattern_name
                base_confidence = pattern_info.get("confidence", 0.7)
                confidence_adj += base_confidence
                reasons.append(f"Matched {pattern_name} pattern: +{base_confidence}")
                entity.confidence += base_confidence  # Apply immediately
                break
    
        if not matched_pattern:
            reasons.append("Did not match any date pattern")
            return ValidationResult(False, confidence_adj, "; ".join(reasons))
    
        # If we got this far, we have a valid date - now check for bonuses/penalties
        
        # False positive check
        if any(fp in context.lower() for fp in false_positive_contexts):
            penalty = adjustments.get("false_positive_context", -0.5)
            confidence_adj += penalty
            entity.confidence += penalty
            reasons.append(f"False positive context penalty: {penalty}")
            return ValidationResult(False, confidence_adj, "; ".join(reasons))
    
        # Year marker bonus
        if any(marker in text_lower for marker in year_markers):
            boost = adjustments.get("has_marker", 0.2)
            confidence_adj += boost
            entity.confidence += boost
            reasons.append(f"Year marker boost: +{boost}")
    
        # Current term bonus for ranges
        if "year_range" in matched_pattern and any(term in text_lower for term in current_terms):
            boost = adjustments.get("has_context", 0.1)
            confidence_adj += boost
            entity.confidence += boost
            reasons.append(f"Current term boost: +{boost}")
    
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=confidence_adj,
            reason="; ".join(reasons)
        )
    

    def validate_entity(self, entity: Entity, text: str) -> ValidationResult:
        """
        Dispatcher for validation by entity type.
        Currently supports:
        - EDUCATIONAL_INSTITUTION
        - DATE_TIME
        """
        if entity.entity_type == "EDUCATIONAL_INSTITUTION":
            return self.validate_educational_institution(entity, text)
        elif entity.entity_type == "DATE_TIME":
            return self.validate_date(entity, text)
    
        # Default: unknown entity => invalid
        return ValidationResult(False, 0.0, f"Unknown entity type: {entity.entity_type}")
    
        
    def validate_from_context(self, entity: Entity, context: str) -> ValidationResult:
        """
        Optional context-based validation. For example, add +0.1 if the context suggests an educational section.
        """
        if not entity.text or not context:
            return ValidationResult(False, -1.0, "Invalid input")

        # Primary validation
        validation_result = self.validate_entity(entity, entity.text)

        # If context suggests an education section, optionally add an extra boost
        if "education" in context.lower() and validation_result.is_valid:
            validation_result.confidence_adjustment += 0.1
            if validation_result.reason:
                validation_result.reason += "; Context boost applied"
            else:
                validation_result.reason = "Context boost applied"

        return validation_result
