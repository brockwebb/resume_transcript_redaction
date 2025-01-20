# redactor/detectors/base_detector.py

from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Entity:
    """Represents a detected entity with its metadata."""
    text: str
    entity_type: str
    confidence: float
    start: int
    end: int
    source: str
    metadata: Optional[Dict[str, Any]] = None

class BaseDetector(ABC):
    """Abstract base class for entity detectors."""
    
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
        self._validate_config()

    @abstractmethod
    def _validate_config(self) -> bool:
        """
        Validate detector-specific configuration.
        
        Returns:
            bool: Whether configuration is valid
        """
        pass

    @abstractmethod
    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        pass

    @abstractmethod
    def validate_detection(self, entity: Entity) -> bool:
        """
        Validate a detected entity.
        
        Args:
            entity: Entity to validate
            
        Returns:
            bool: Whether entity is valid
        """
        pass

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

    def log_detection(self, entity: Entity, context: str = "") -> None:
        """
        Log entity detection details.
        
        Args:
            entity: Detected entity
            context: Optional context information
        """
        if self.logger:
            self.logger.debug(
                f"Detected {entity.entity_type}: '{entity.text}' "
                f"(confidence: {entity.confidence:.2f}) {context}"
            )

    def prepare_text(self, text: str) -> str:
        """
        Prepare text for detection.
        
        Args:
            text: Input text
            
        Returns:
            str: Prepared text
        """
        if not text:
            return ""
        return text.strip()