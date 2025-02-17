# evaluation/models.py
from enum import Enum, auto
from dataclasses import dataclass, replace, field
from typing import Optional, List, Union, Dict, Any
import logging


class SensitivityLevel(Enum):
    """
    Enumeration of data sensitivity levels.
    
    Represents the potential privacy impact of an identified entity.
    """
    HIGH = "high"
    MEDIUM = "medium" 
    LOW = "low"

@dataclass(frozen=True)
class Entity:
    """
    Represents a detected entity with comprehensive metadata.
    
    Immutable dataclass to ensure consistent entity representation
    across the redaction system's detection and validation workflow.
    
    Attributes:
        text: The actual text of the detected entity
        entity_type: Classified type of the entity (e.g., PERSON, ADDRESS)
        start_char: Starting character position in the source text
        end_char: Ending character position in the source text
        confidence: Detection confidence score (0.0 to 1.0)
        page: Optional page number of the document
        sensitivity: Assessed sensitivity level of the entity
        detector_source: Origin of entity detection (e.g., Presidio, SpaCy)
        validation_rules: List of validation rules applied
        original_confidence: Initial confidence before adjustments
    """
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    page: Optional[int] = None
    sensitivity: Optional[SensitivityLevel] = None
    detector_source: Optional[str] = None
    validation_rules: Optional[List[str]] = field(default_factory=list)
    original_confidence: Optional[float] = None

    def __post_init__(self):
        """
        Validate entity attributes during initialization.
        """
        # Validate character positions
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError(f"Character positions must be non-negative, got start={self.start_char}, end={self.end_char}")
        
        # Validate text
        if not self.text or not isinstance(self.text, str):
            raise ValueError("Entity text must be a non-empty string")

    def adjust_confidence(
        self, 
        adjustment: float, 
        logger: Optional[logging.Logger] = None
    ) -> 'Entity':
        """
        Create a new Entity with adjusted confidence.
        
        Args:
            adjustment: Confidence score adjustment (positive or negative)
            logger: Optional logger for tracking confidence changes
        
        Returns:
            New Entity instance with adjusted confidence
        
        Raises:
            ValueError: If adjustment would push confidence outside [0, 1]
        """
        new_confidence = max(0.0, min(1.0, self.confidence + adjustment))
        
        if logger:
            logger.debug(
                f"Confidence Adjustment: "
                f"{self.confidence:.2f} -> {new_confidence:.2f} "
                f"(+{adjustment:.2f}) for '{self.text}'"
            )
        
        return replace(
            self, 
            confidence=new_confidence,
            original_confidence=self.original_confidence or self.confidence,
            validation_rules=self.validation_rules + [f"confidence_adj_{adjustment}"]
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Entity to a dictionary representation.
        
        Handles Enum serialization and provides a clean dictionary output.
        
        Returns:
            Dictionary representation of the Entity
        """
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": self.confidence,
            "page": self.page,
            "sensitivity": self.sensitivity.value if self.sensitivity else None,
            "detector_source": self.detector_source,
            "validation_rules": self.validation_rules,
            "original_confidence": self.original_confidence
        }

    def to_json_dict(self) -> Dict[str, Any]:
        """
        Prepare Entity for JSON serialization.
        
        Converts Enum values to strings and handles potential complex types.
        
        Returns:
            JSON-compatible dictionary representation
        """
        dict_data = self.to_dict()
        # Convert any enum values to strings
        dict_data["sensitivity"] = str(dict_data["sensitivity"]) if dict_data["sensitivity"] else None
        return dict_data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Entity':
        """
        Create an Entity from a dictionary representation.
        
        Supports reconstruction of Entity from serialized data.
        
        Args:
            data: Dictionary containing Entity attributes
        
        Returns:
            Reconstructed Entity instance
        """
        # Handle sensitivity level conversion
        if isinstance(data.get('sensitivity'), str):
            data['sensitivity'] = SensitivityLevel(data['sensitivity'])
        
        return cls(**data)

    def overlaps(self, other: 'Entity') -> bool:
        """
        Check if this Entity overlaps with another Entity.
        
        Args:
            other: Another Entity to compare
        
        Returns:
            Boolean indicating if Entities have overlapping character ranges
        """
        return not (
            self.end_char <= other.start_char or 
            self.start_char >= other.end_char
        )


def adjust_entity_confidence(
    entity: Entity, 
    adjustment: float, 
    logger: Optional[logging.Logger] = None
) -> Entity:
    """
    Standalone function for adjusting entity confidence.
    
    Adjusts the entity's confidence value by the given adjustment.
    Maintains strict immutability by using replace().
    
    Args:
        entity: Original Entity instance to adjust
        adjustment: Confidence score adjustment (positive or negative)
        logger: Optional logger for tracking confidence changes
    
    Returns:
        New Entity instance with adjusted confidence
    """
    # Apply adjustment without clamping
    new_confidence = entity.confidence + adjustment
    
    if logger:
        logger.debug(
            f"Confidence Adjustment: {entity.confidence:.2f} -> {new_confidence:.2f} "
            f"({adjustment:+.2f}) for '{entity.text}'"
        )
    
    return replace(
        entity, 
        confidence=new_confidence,
        original_confidence=entity.original_confidence or entity.confidence,
        validation_rules=list(entity.validation_rules or []) + [f"confidence_adj_{adjustment:+.2f}"]
    )