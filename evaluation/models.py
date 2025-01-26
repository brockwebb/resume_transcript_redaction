# evaluation/models.py

# evaluation/models.py
from enum import Enum
from dataclasses import dataclass
from typing import Optional

class SensitivityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass(frozen=True)
class Entity:
    text: str
    entity_type: str
    start_char: int
    end_char: int
    confidence: float
    page: Optional[int] = None
    sensitivity: Optional[SensitivityLevel] = None

def to_dict(self):
    return {
        "text": self.text,
        "entity_type": self.entity_type,
        "start_char": self.start_char,
        "end_char": self.end_char,
        "confidence": self.confidence,
        "page": self.page,
        "sensitivity": self.sensitivity.value if self.sensitivity else None
    }

def to_json_dict(self):
    dict_data = self.to_dict()
    # Convert any enum values to strings
    dict_data["sensitivity"] = str(dict_data["sensitivity"]) if dict_data["sensitivity"] else None
    return dict_data