#/evaluation/metrics/entity_metrics.py

from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from enum import Enum

class SensitivityLevel(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass(frozen=True)
class Entity:
    """Represents an entity detected in text"""
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

    def __eq__(self, other):
        if not isinstance(other, Entity):
            return False
        return (self.text == other.text and 
                self.entity_type == other.entity_type and
                self.start_char == other.start_char and 
                self.end_char == other.end_char)

    def __hash__(self):
        return hash((self.text, self.entity_type, self.start_char, self.end_char))

@dataclass
class EntityMatchResult:
    """Results from comparing detected entities to ground truth"""
    true_positives: Set[Entity]
    false_positives: Set[Entity]
    false_negatives: Set[Entity]
    partial_matches: Set[tuple[Entity, Entity]]  # (ground_truth, detected)

class EntityMetrics:
    """Calculates and tracks entity detection metrics"""
    
    def __init__(self):
        self.results_by_type: Dict[str, EntityMatchResult] = {}
        self.results_by_sensitivity: Dict[SensitivityLevel, EntityMatchResult] = {}
    
    def calculate_metrics(self, 
                        ground_truth: List[Entity], 
                        detected: List[Entity],
                        overlap_threshold: float = 0.7) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for entity detection
        
        Args:
            ground_truth: List of annotated ground truth entities
            detected: List of detected entities
            overlap_threshold: Minimum character overlap ratio to consider a match
            
        Returns:
            Dictionary containing precision, recall, and F1 scores
        """
        match_result = self._match_entities(ground_truth, detected, overlap_threshold)
        
        # Calculate basic metrics
        precision = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_positives)
        ) if (len(match_result.true_positives) + len(match_result.false_positives)) > 0 else 0
        
        recall = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_negatives)
        ) if (len(match_result.true_positives) + len(match_result.false_negatives)) > 0 else 0
        
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "partial_matches": len(match_result.partial_matches),
            "false_positives": len(match_result.false_positives),
            "false_negatives": len(match_result.false_negatives)
        }
    
    def _match_entities(self, 
                       ground_truth: List[Entity], 
                       detected: List[Entity],
                       overlap_threshold: float) -> EntityMatchResult:
        """
        Match detected entities against ground truth entities
        
        Args:
            ground_truth: List of ground truth entities
            detected: List of detected entities
            overlap_threshold: Minimum overlap ratio to consider a match
            
        Returns:
            EntityMatchResult containing matched and unmatched entities
        """
        true_positives = set()
        false_positives = set(detected)  # Start with all detected, remove as matched
        false_negatives = set(ground_truth)  # Start with all ground truth, remove as matched
        partial_matches = set()
        
        for gt_entity in ground_truth:
            for det_entity in detected:
                if self._is_matching_entity(gt_entity, det_entity, overlap_threshold):
                    true_positives.add(det_entity)
                    false_positives.remove(det_entity)
                    false_negatives.remove(gt_entity)
                elif self._is_partial_match(gt_entity, det_entity, overlap_threshold):
                    partial_matches.add((gt_entity, det_entity))
        
        return EntityMatchResult(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            partial_matches=partial_matches
        )

    def get_type_based_metrics(self, results):
        """Get metrics organized by detector and entity type"""
        return {
            "presidio_detections": self._calculate_metrics_for_types(results, [
                "ADDRESS", "EMAIL_ADDRESS", "PHONE_NUMBER", "DATE_TIME",
                "URL", "SOCIAL_MEDIA", "LOCATION", "GPA", 
                "EDUCATIONAL_INSTITUTION", "DEMOGRAPHIC"
            ]),
            "spacy_detections": self._calculate_metrics_for_types(results, [
                "PERSON", "GPE"
            ]),
            "ensemble_detections": self._calculate_metrics_for_types(results, [
                "LOCATION_FULL", "EDUCATIONAL_INSTITUTION", "DEMOGRAPHIC"
            ])
        }

    def _calculate_metrics_for_types(self, results: Dict, entity_types: List[str]) -> Dict[str, Dict[str, float]]:
       """Calculate metrics broken down by entity type"""
       metrics_by_type = {}
       
       for entity_type in entity_types:
           # Filter results for this entity type
           type_matches = {
               "true_positives": self._filter_by_type(results.true_positives, entity_type),
               "false_positives": self._filter_by_type(results.false_positives, entity_type),
               "false_negatives": self._filter_by_type(results.false_negatives, entity_type)
           }
           
           # Calculate core metrics
           tp = len(type_matches["true_positives"])
           fp = len(type_matches["false_positives"]) 
           fn = len(type_matches["false_negatives"])
           
           precision = tp / (tp + fp) if (tp + fp) > 0 else 0
           recall = tp / (tp + fn) if (tp + fn) > 0 else 0
           f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
           metrics_by_type[entity_type] = {
               "precision": precision,
               "recall": recall,
               "f1_score": f1,
               "false_positives": fp,
               "false_negatives": fn,
               "detection_count": tp + fp  # Total detections of this type
           }
           
       return metrics_by_type

    def _filter_by_type(self, entities: Set[Entity], entity_type: str) -> Set[Entity]:
       """Helper to filter entity sets by type"""
       return {e for e in entities if e.entity_type == entity_type}
        
    
    @staticmethod
    def _is_matching_entity(gt_entity: Entity, 
                           detected_entity: Entity, 
                           threshold: float) -> bool:
        """
        Determine if two entities match based on overlap and type
        
        Args:
            gt_entity: Ground truth entity
            detected_entity: Detected entity
            threshold: Minimum overlap threshold
            
        Returns:
            Boolean indicating whether entities match
        """
        if gt_entity.entity_type != detected_entity.entity_type:
            return False
            
        # Calculate character overlap
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        union = len(gt_chars | det_chars)
        
        overlap_ratio = overlap / union if union > 0 else 0
        return overlap_ratio >= threshold
    
    @staticmethod
    def _is_partial_match(gt_entity: Entity,
                         detected_entity: Entity,
                         threshold: float) -> bool:
        """
        Check if entities partially match but don't meet full match criteria
        
        Args:
            gt_entity: Ground truth entity
            detected_entity: Detected entity
            threshold: Overlap threshold for partial match
            
        Returns:
            Boolean indicating partial match
        """
        if gt_entity.entity_type != detected_entity.entity_type:
            return False
            
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        min_length = min(len(gt_chars), len(det_chars))
        
        overlap_ratio = overlap / min_length if min_length > 0 else 0
        return 0 < overlap_ratio < threshold