from dataclasses import dataclass
from pathlib import Path
import fitz
from typing import List
from .metrics.entity_metrics import Entity, SensitivityLevel

class RedactionWrapper:
    def __init__(self, processor):
        self.processor = processor
        
    def detect_entities(self, path: Path) -> List[Entity]:
        # Read PDF
        doc = fitz.open(str(path))
        
        # Get words with their positions
        words = []
        for page in doc:
            words.extend(page.get_text("words"))
        
        # Get redaction rectangles
        redaction_rects = self.processor._detect_redactions(words)
        
        # Convert to entities
        entities = []
        for rect in redaction_rects:
            # Find text within rectangle bounds
            text = ""
            start_char = 0
            end_char = 0
            
            # Match words to redaction rectangles
            for word_tuple in words:
                word_rect = fitz.Rect(word_tuple[0:4])
                if rect.intersects(word_rect):
                    text = word_tuple[4]  # The actual word text
                    start_char = int(word_tuple[0])  # x0 as char position
                    end_char = int(word_tuple[2])   # x1 as char position
                    
            entity = Entity(
                text=text,
                entity_type="UNKNOWN",  # Need to get this from processor somehow
                start_char=start_char,
                end_char=end_char,
                confidence=1.0,
                page=1
            )
            entities.append(entity)
        
        return entities