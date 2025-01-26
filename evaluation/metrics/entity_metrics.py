# evaluation/metrics/entity_metrics.py

from evaluation.models import Entity, SensitivityLevel
from evaluation.comparison.entity_matcher import MatchType
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

@dataclass
class EntityMatchResult:
   true_positives: Set[Entity]
   false_positives: Set[Entity]
   false_negatives: Set[Entity]
   partial_matches: Set[tuple[Entity, Entity]]

class EntityMetrics:
   def __init__(self):
       self.results_by_type: Dict[str, EntityMatchResult] = {}
       self.results_by_sensitivity: Dict[SensitivityLevel, EntityMatchResult] = {}

   def calculate_metrics(self, ground_truth: List[Entity], detected: List[Entity],
                       overlap_threshold: float = 0.7) -> Dict[str, float]:
       match_result = self._match_entities(ground_truth, detected, overlap_threshold)
       
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

   def _match_entities(self, ground_truth: List[Entity], detected: List[Entity], 
                      overlap_threshold: float) -> EntityMatchResult:
       true_positives = set()
       false_positives = set(detected)
       false_negatives = set(ground_truth)
       partial_matches = set()

       for gt_entity in ground_truth:
           for det_entity in detected:
               if self._is_matching_entity(gt_entity, det_entity, overlap_threshold):
                   true_positives.add(det_entity)
                   if det_entity in false_positives:
                       false_positives.remove(det_entity)
                   if gt_entity in false_negatives:
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

   def _calculate_metrics_for_types(self, matches_dict: Dict, entity_types: List[str]) -> Dict[str, Dict[str, float]]:
       metrics_by_type = {}
       
       for entity_type in entity_types:
           exact_matches = {e for e in matches_dict.get(MatchType.EXACT.value, set()) 
                         if e.entity_type == entity_type}
           partial = {e for e in matches_dict.get(MatchType.PARTIAL.value, set())
                   if e.entity_type == entity_type}
           false_positives = {e for e in matches_dict.get("unmatched_detected", set())
                           if e.entity_type == entity_type}
           false_negatives = {e for e in matches_dict.get("unmatched_ground_truth", set())
                          if e.entity_type == entity_type}

           tp = len(exact_matches)
           fp = len(false_positives)
           fn = len(false_negatives)
           
           precision = tp / (tp + fp) if (tp + fp) > 0 else 0
           recall = tp / (tp + fn) if (tp + fn) > 0 else 0
           f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

           metrics_by_type[entity_type] = {
               "precision": precision,
               "recall": recall,
               "f1_score": f1,
               "exact_matches": tp,
               "false_positives": fp,
               "false_negatives": fn,
               "partial_matches": len(partial),
               "total_detections": tp + fp
           }
       
       return metrics_by_type

   def _aggregate_detector_metrics(self, detector_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
       aggregated = {}
       
       for test_metrics in detector_metrics:
           for entity_type, metrics in test_metrics.items():
               if entity_type not in aggregated:
                   aggregated[entity_type] = {
                       "precision": [],
                       "recall": [],
                       "false_positives": 0,
                       "false_negatives": 0,
                       "total_detections": 0
                   }
               
               agg = aggregated[entity_type]
               agg["precision"].append(metrics["precision"])
               agg["recall"].append(metrics["recall"]) 
               agg["false_positives"] += metrics["false_positives"]
               agg["false_negatives"] += metrics["false_negatives"]
               agg["total_detections"] += metrics["total_detections"]

       for entity_type in aggregated:
           metrics = aggregated[entity_type]
           metrics["precision"] = sum(metrics["precision"]) / len(metrics["precision"])
           metrics["recall"] = sum(metrics["recall"]) / len(metrics["recall"])
           
       return aggregated

   def _is_matching_entity(self, gt_entity: Entity, detected_entity: Entity, threshold: float) -> bool:
       if gt_entity.entity_type != detected_entity.entity_type:
           return False
           
       gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
       det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
       
       overlap = len(gt_chars & det_chars)
       union = len(gt_chars | det_chars)
       
       overlap_ratio = overlap / union if union > 0 else 0
       return overlap_ratio >= threshold

   def _is_partial_match(self, gt_entity: Entity, det_entity: Entity, threshold: float) -> bool:
       if gt_entity.entity_type != det_entity.entity_type:
           return False
              
       start = max(gt_entity.start_char, det_entity.start_char)
       end = min(gt_entity.end_char, det_entity.end_char)
       if end <= start:
           return 0.0
          
       overlap = end - start
       min_length = min(gt_entity.end_char - gt_entity.start_char, 
                       det_entity.end_char - det_entity.start_char)
      
       overlap_ratio = overlap / min_length if min_length > 0 else 0
       return 0 < overlap_ratio < threshold