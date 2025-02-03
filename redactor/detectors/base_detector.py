# redactor/detectors/base_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Entity:
    """
    Represents a detected entity with its metadata.
    Core data structure for all entity detection results.
    """
    text: str
    entity_type: str
    confidence: float
    start: int
    end: int
    source: str
    metadata: Optional[Dict[str, Any]] = None

###############################################################################
# SECTION 2: BASE DETECTOR CLASS
###############################################################################

class BaseDetector(ABC):
    """
    Abstract base class for entity detectors.
    Provides common functionality and interface for all detectors.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize detector with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            confidence_threshold: Minimum confidence threshold
        """
        self.logger = logger
        self.config_loader = config_loader
        self.confidence_threshold = confidence_threshold
        self._cached_configs = {}
        
        # Set debug mode based on logger configuration
        self.debug_mode = False
        if logger:
            if hasattr(logger, 'level'):
                self.debug_mode = logger.level == 10
            elif hasattr(logger, 'getEffectiveLevel'):
                self.debug_mode = logger.getEffectiveLevel() == 10
            elif hasattr(logger, 'log_level'):
                self.debug_mode = logger.log_level == "DEBUG"
        
        if self.debug_mode and self.logger:
            self.logger.debug("Initialized BaseDetector in debug mode")
            
        # Load and cache core configurations
        if not self._load_core_configs():
            if self.logger:
                self.logger.error("Failed to load core configurations")
            raise ValueError("Failed to load core configurations")
            
        # Validate detector-specific configuration
        if not self._validate_config():
            if self.logger:
                self.logger.error("Configuration validation failed")
            raise ValueError("Configuration validation failed")

    def _load_core_configs(self) -> bool:
        """Load and cache core configurations."""
        try:
            # Load validation parameters
            validation_config = self.config_loader.get_config("validation_params")
            if validation_config:
                self._cached_configs["validation_params"] = validation_config
                if self.debug_mode:
                    self.logger.debug("Loaded validation_params configuration")
            else:
                if self.logger:
                    self.logger.error("Missing validation_params configuration")
                return False

            # Load entity routing
            routing_config = self.config_loader.get_config("entity_routing")
            if routing_config:
                self._cached_configs["entity_routing"] = routing_config
                if self.debug_mode:
                    self.logger.debug("Loaded entity_routing configuration")
            else:
                if self.logger:
                    self.logger.error("Missing entity_routing configuration")
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading core configs: {str(e)}")
            return False

    def _validate_config(self) -> bool:
        """
        Base configuration validation.
        Validates presence of required configs and basic structure.
        
        Returns:
            bool: Whether configuration is valid
        """
        try:
            # Get cached configs
            validation_config = self._cached_configs.get("validation_params")
            routing_config = self._cached_configs.get("entity_routing")
            
            if not validation_config or not routing_config:
                if self.logger:
                    self.logger.error("Missing required cached configurations")
                return False

            # Verify routing structure
            routing = routing_config.get("routing", {})
            if not routing:
                if self.logger:
                    self.logger.error("Missing routing configuration")
                return False

            # Verify at least one valid routing section exists
            valid_sections = ["presidio_primary", "spacy_primary", "ensemble_required"]
            if not any(section in routing for section in valid_sections):
                if self.logger:
                    self.logger.error("No valid routing sections found")
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating config: {str(e)}")
            return False

###############################################################################
# SECTION 3: ABSTRACT INTERFACE METHODS
###############################################################################

    @abstractmethod
    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities in text.
        Must be implemented by each detector.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List[Entity]: List of detected entities
        """
        pass

###############################################################################
# SECTION 4: UTILITY METHODS
###############################################################################

    def get_confidence(self, entity: Entity) -> float:
        """
        Get confidence score for entity.
        
        Args:
            entity: Entity to score
            
        Returns:
            float: Confidence score between 0 and 1
        """
        if not entity.confidence or entity.confidence < 0:
            return 0.0
        return min(1.0, entity.confidence)

    def check_threshold(self, confidence: float) -> bool:
        """
        Check if confidence meets threshold.
        
        Args:
            confidence: Confidence score to check
            
        Returns:
            bool: Whether confidence meets threshold
        """
        return confidence >= self.confidence_threshold

    def prepare_text(self, text: str) -> str:
        """
        Prepare text for detection.
        Basic text normalization.
        
        Args:
            text: Input text
            
        Returns:
            str: Prepared text
        """
        if not text:
            return ""
        return text.strip()