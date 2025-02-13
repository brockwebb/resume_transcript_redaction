# redactor/validation/validation_coordinator.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set, Tuple
import re
import logging
import json
from dataclasses import dataclass, replace
from pathlib import Path

from ..detectors.base_detector import Entity
from app.utils.config_loader import ConfigLoader
from .entity_validation import (
    validate_educational_institution,
    validate_person,
    validate_date,
    validate_location,
    validate_address,
    validate_email,
    validate_phone,
    validate_gpa,
    validate_internet_reference,
    validate_protected_class,
    validate_phi
)

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
                 config_loader: ConfigLoader,
                 logger: Optional[logging.Logger] = None,
                 debug_mode: bool = False):
        """
        Initialize validation coordinator with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            debug_mode: Enable detailed debug logging
        """
        # Set up logging
        self.logger = logger or logging.getLogger(__name__)
        self.debug_mode = debug_mode
        
        # Initialize configuration
        self.config_loader = config_loader
        
        # Load configurations
        try:
            self.validation_params = self._load_validation_params()
            self.routing_config = self._load_routing_config()
        except Exception as e:
            self.logger.error(f"Error loading validation configurations: {e}")
            raise
        
        # Initialize validation methods dictionary
        self.validation_methods = {}
        self._init_validation_methods()
        
        if self.debug_mode:
            self._log_initialization()

###############################################################################
# SECTION 3: VALIDATION METHOD MANAGEMENT
###############################################################################

    def _init_validation_methods(self) -> None:
        """Initialize validation methods mapping."""
        self.validation_methods = {
            "EDUCATIONAL_INSTITUTION": validate_educational_institution,
            "PERSON": validate_person,
            "DATE_TIME": validate_date,
            "LOCATION": validate_location,
            "ADDRESS": validate_address,
            "EMAIL_ADDRESS": validate_email,
            "PHONE_NUMBER": validate_phone,
            "GPA": validate_gpa,
            "INTERNET_REFERENCE": validate_internet_reference,
            "URL": validate_internet_reference,  # Uses same validator
            "IP_ADDRESS": validate_internet_reference,  # Uses same validator
            "PROTECTED_CLASS": validate_protected_class,
            "PHI": validate_phi
        }
        
        if self.debug_mode:
            self.logger.debug(f"Initialized validation methods: {list(self.validation_methods.keys())}")

    def get_validation_method(self, entity_type: str):
        """
        Get the appropriate validation method for an entity type.
        
        Args:
            entity_type: Type of entity to validate
            
        Returns:
            Validation function for the entity type, or None if not found
        """
        method = self.validation_methods.get(entity_type)
        if not method and self.debug_mode:
            self.logger.debug(f"No validation method found for {entity_type}")
        return method

###############################################################################
# SECTION 4: CONFIGURATION MANAGEMENT
###############################################################################

    def _load_validation_params(self) -> Dict:
        """
        Load validation parameters from configuration.
        
        Returns:
            Dict containing validation parameters
        """
        try:
            validation_config = self.config_loader.get_config("validation_params")
            if not validation_config:
                self.logger.warning("No validation parameters found")
                return {}
            
            if self.debug_mode:
                self.logger.debug("Loaded validation parameters successfully")
            
            return validation_config
        
        except Exception as e:
            self.logger.error(f"Error loading validation parameters: {e}")
            raise

    def _load_routing_config(self) -> Dict:
        """
        Load entity routing configuration.
        
        Returns:
            Dict containing routing configuration
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                self.logger.warning("Incomplete routing configuration")
                return {}
            
            if self.debug_mode:
                self.logger.debug("Loaded routing configuration successfully")
            
            return routing_config.get("routing", {})
        
        except Exception as e:
            self.logger.error(f"Error loading routing configuration: {e}")
            raise

###############################################################################
# SECTION 5: VALIDATION LOGIC
###############################################################################

    def validate_entity(self, entity: Entity, text: str, source: str = None) -> ValidationResult:
        """
        Validate entity using configuration-driven rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            source: Optional detector source
        
        Returns:
            ValidationResult with validation outcome
        """
        if not entity or not entity.text:
            return ValidationResult(is_valid=False, reasons=["Empty entity"])

        if self.debug_mode:
            self.logger.debug(f"\n=== Starting Validation ===")
            self.logger.debug(f"Entity: {entity.text} ({entity.entity_type})")
            self.logger.debug(f"Initial confidence: {entity.confidence}")

        # Get validation method
        validator = self.get_validation_method(entity.entity_type)
        
        if not validator:
            if self.debug_mode:
                self.logger.debug(f"No validator found for {entity.entity_type}")
            return ValidationResult(is_valid=True)

        # Get validation rules for this entity type
        validation_rules = self.validation_params.get(entity.entity_type, {})

        if self.debug_mode:
            self.logger.debug(f"Using validator for {entity.entity_type}")
            self.logger.debug(f"Validation rules: {validation_rules}")

        # Perform validation
        validation_result = validator(entity, text, validation_rules, self.logger)

        # HERE - We need to clamp the confidence adjustment before creating ValidationResult
        if validation_result.get('confidence_adjustment'):
            # Calculate what adjustment would put us at 1.0
            max_adjustment = max(0.0, 1.0 - entity.confidence)
            # Clamp the adjustment
            clamped_adjustment = min(max_adjustment, validation_result['confidence_adjustment'])
            validation_result['confidence_adjustment'] = clamped_adjustment
        
        # Convert dict result to ValidationResult
        result = ValidationResult(
            is_valid=validation_result['is_valid'],
            confidence_adjustment=validation_result.get('confidence_adjustment', 0.0),
            reasons=validation_result.get('reasons', [])
        )

        
        if self.debug_mode:
            self.logger.debug(f"\n=== Validation Complete ===")
            self.logger.debug(f"Valid: {result.is_valid}")
            self.logger.debug(f"Adjustment: {result.confidence_adjustment}")
            self.logger.debug(f"Reasons: {result.reasons}")

        # Create new entity with adjusted confidence if needed
        if result.is_valid and result.confidence_adjustment != 0:
            entity = replace(entity, confidence=entity.confidence + result.confidence_adjustment)
            result.metadata = {"updated_entity": entity}

        return result

###############################################################################
# END OF FILE
###############################################################################