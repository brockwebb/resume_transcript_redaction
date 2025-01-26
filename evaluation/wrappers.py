from dataclasses import dataclass
from pathlib import Path
import fitz
from typing import List
from evaluation.models import Entity, SensitivityLevel

class RedactionWrapper:
    def __init__(self, processor):
        self.processor = processor
        
    def detect_entities(self, path: Path) -> List[Entity]:
       doc = fitz.open(str(path))
       text = ""
       for page in doc:
           text += page.get_text()
       
       # Use ensemble to get entity types
       detected = self.processor.ensemble.detect_entities(text)
       entities = []
       
       for entity in detected:
           entities.append(Entity(
               text=entity.text,
               entity_type=entity.entity_type,
               start_char=entity.start,
               end_char=entity.end,
               confidence=entity.confidence,
               page=1
           ))
       
       return entities