from dataclasses import dataclass, replace
from pathlib import Path
import fitz
from typing import List, Optional, Dict 
from evaluation.models import Entity, SensitivityLevel

class RedactionWrapper:
    def __init__(self, processor):
        self.processor = processor
        
    def detect_entities(self, path: Path, target_entity_type: Optional[str] = None) -> List[Entity]:
        """
        Detect entities from a document with optional entity type filtering.
        
        Args:
            path: Path to document
            target_entity_type: Optional entity type to filter for
            
        Returns:
            List of detected entities
        """
        # Extract text from document
        doc = fitz.open(str(path))
        text = ""
        for page in doc:
            text += page.get_text()
       
        # Get raw detections from ensemble
        detected = self.processor.ensemble.detect_entities(text)
        entities = []
       
        for entity in detected:
            # Skip if we're filtering for a specific type and this doesn't match
            if target_entity_type and entity.entity_type != target_entity_type:
                continue
                
            # Create base entity first
            base_entity = Entity(
                text=entity.text,
                entity_type=entity.entity_type,
                start_char=entity.start_char if hasattr(entity, 'start_char') else entity.start,
                end_char=entity.end_char if hasattr(entity, 'end_char') else entity.end,
                confidence=entity.confidence,
                page=1
            )
            
            # Use replace to add additional attributes if they exist
            updated_entity = replace(
                base_entity,
                detector_source=getattr(entity, 'detector_source', None),
                validation_rules=getattr(entity, 'validation_rules', []),
                original_confidence=getattr(entity, 'original_confidence', entity.confidence)
            )
            
            entities.append(updated_entity)
       
        return entities

class MetricsWrapper:
    """Wrapper to handle different data structures for metrics calculations."""
    
    def __init__(self, metrics_calculator):
        self.calculator = metrics_calculator
        
    def calculate_type_metrics(self, entity_type: str, matches: Dict) -> Dict:
        """Calculate metrics handling different match data structures."""
        # First make all entity types hashable
        safe_matches = {}
        for key, entities in matches.items():
            if isinstance(entities, (list, set)):
                # Safe conversion to list with no list-typed attributes
                safe_entities = []
                for entity in entities:
                    if hasattr(entity, 'entity_type') and isinstance(entity.entity_type, list):
                        # Create copy with tuple instead of list
                        safe_entity = replace(entity, entity_type=tuple(entity.entity_type))
                        safe_entities.append(safe_entity)
                    else:
                        safe_entities.append(entity)
                safe_matches[key] = safe_entities
            else:
                safe_matches[key] = entities
        
        # Now calculate metrics using the safe matches
        return self.calculator._calculate_type_metrics(entity_type, safe_matches)