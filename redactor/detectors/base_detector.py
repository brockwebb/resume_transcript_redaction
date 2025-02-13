# redactor/detectors/base_detector.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from evaluation.models import Entity


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
        
        # Cache needed configurations
        self._cache_configs()
        
        if self.debug_mode and self.logger:
            self.logger.debug("Initialized BaseDetector in debug mode")

    def _cache_configs(self) -> None:
        """Cache commonly used configurations."""
        try:
            config_names = ["validation_params", "entity_routing"]
            for name in config_names:
                config = self.config_loader.get_config(name)
                if config:
                    self._cached_configs[name] = config
                elif self.debug_mode:
                    self.logger.debug(f"No config found for {name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error caching configs: {str(e)}")

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

    @abstractmethod
    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate detected entity.
        Must be implemented by each detector using ValidationCoordinator.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool indicating if entity is valid
        """
        pass

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
            
        # Basic text normalization
        try:
            text = text.strip()
            text = " ".join(text.split())  # Normalize whitespace
            return text
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error preparing text: {str(e)}")
            return ""

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
            
        try:
            # Basic normalization
            text = text.lower().strip()
            
            # Remove punctuation except preserved characters
            text = ''.join(
                c for c in text 
                if c.isalnum() or c in '.-_ '
            )
            
            # Normalize whitespace
            text = ' '.join(text.split())
            
            return text
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error normalizing text: {str(e)}")
            return ""

    def _safe_log(self, msg: str, level: str = "debug") -> None:
        """
        Safely log message if logger exists.
        
        Args:
            msg: Message to log
            level: Log level (debug, info, warning, error)
        """
        if not self.logger:
            return
            
        try:
            if level == "debug" and self.debug_mode:
                self.logger.debug(msg)
            elif level == "info":
                self.logger.info(msg)
            elif level == "warning":
                self.logger.warning(msg)
            elif level == "error":
                self.logger.error(msg)
                
        except Exception:
            pass  # Fail silently if logging fails