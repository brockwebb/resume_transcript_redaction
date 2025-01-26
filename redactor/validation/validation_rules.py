# redactor/validation/validation_rules.py
from typing import Dict, Any, Optional
from pathlib import Path
import json
import logging
from redactor.detectors.base_detector import Entity

class ValidationResult:
    """
    Holds the result of a validation check, indicating:
      - is_valid: Whether the entity passes validation.
      - confidence_adjustment: How much to adjust the entity's confidence.
      - reason: Explanation for the validation result.
    """
    def __init__(self, is_valid: bool, confidence_adjustment: float = 0.0, reason: Optional[str] = None):
        self.is_valid = is_valid
        self.confidence_adjustment = confidence_adjustment
        self.reason = reason


class EntityValidationRules:
    """
    A class encapsulating custom validation logic for different entity types.
    Currently supports validating EDUCATIONAL_INSTITUTION entities.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initializes the validation rules, loading any required configuration
        from a JSON file (e.g. validation_params.json).
        """
        # Use caller-provided logger if available; otherwise, create a module-level logger
        self.logger = logger if logger else logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Attempt to load validation parameters from JSON
        config_path = Path("redactor/config/validation_params.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self.params = json.load(f)
                self.logger.debug("Loaded validation params from %s", config_path)
        except Exception as e:
            self.logger.error("Failed to load validation params: %s", e)
            # Raise so that any higher-level code knows configuration loading failed
            raise

    def validate_educational_institution(self, entity_text: str, context: str) -> ValidationResult:
        """
        Validates a potential 'EDUCATIONAL_INSTITUTION' entity based on the text (entity_text)
        and any additional context (context). Returns a ValidationResult indicating whether it
        is valid and how much to adjust the confidence score.

        :param entity_text: The raw text of the potential educational institution (e.g., "MIT").
        :param context: Additional surrounding text or context (not always used in this method).
        :return: ValidationResult with is_valid, confidence_adjustment, and reason.
        """

        # Define some sample educational keywords
        core_edu_terms = {
            "university", "college", "institute", "polytechnic",
            "academy", "school", "universidad", "université"
        }

        text_lower = entity_text.lower()
        reasons = []

        # 1. Check for acronyms (e.g., "MIT", "UCLA")
        #    Criteria: uppercase and length up to 5 (but more than 1 to exclude single char).
        if entity_text.isupper() and 1 < len(entity_text) <= 5:
            reasons.append("Valid acronym-based institution")
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=0.2,
                reason="; ".join(reasons)
            )

        # 2. Check for "university of" or "college of"
        if "university of" in text_lower or "college of" in text_lower:
            reasons.append("Contains 'university of' or 'college of'")
            return ValidationResult(
                is_valid=True,
                confidence_adjustment=0.2,
                reason="; ".join(reasons)
            )

        # 3. Look at the last word for known educational terms
        institution_parts = text_lower.split()
        if len(institution_parts) >= 2:
            last_word = institution_parts[-1].strip()

            # Ends with one of our core educational terms
            if last_word in core_edu_terms:
                reasons.append("Ends with a core educational term")
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=0.2,
                    reason="; ".join(reasons)
                )

            # Specific multi-word scenario, e.g., "Florida International University"
            if last_word == "university" and len(institution_parts) >= 3:
                reasons.append("Multi-word ending with 'university'")
                return ValidationResult(
                    is_valid=True,
                    confidence_adjustment=0.2,
                    reason="; ".join(reasons)
                )

        # If none of the above patterns match, we mark it as invalid and (optionally) penalize
        reasons.append("No education terms")
        return ValidationResult(
            is_valid=False,
            confidence_adjustment=-0.9,  # Adjust penalty as desired
            reason="; ".join(reasons)
        )
