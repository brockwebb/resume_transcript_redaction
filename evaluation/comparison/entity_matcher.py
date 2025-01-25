# evaluation/comparison/entity_matcher.py

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from enum import Enum
from evaluation.metrics.entity_metrics import Entity, SensitivityLevel



@dataclass
class EntityOverlap:
    ground_truth: 'Entity' 
    detected: 'Entity'
    char_overlap: int
    overlap_ratio: float
    type_match: bool
    sensitivity_match: Optional[bool] = None  # Make optional

class MatchType(Enum):
    EXACT = "exact"
    PARTIAL = "partial"
    SENSITIVITY_MISMATCH = "sensitivity_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    NO_MATCH = "no_match"

class EntityMatcher:
    """Handles complex entity matching scenarios for evaluation"""
    
    def __init__(self, 
                exact_match_threshold: float = 0.9,
                partial_match_threshold: float = 0.5):
        self.exact_match_threshold = exact_match_threshold
        self.partial_match_threshold = partial_match_threshold
        self.match_cache: Dict[Tuple[str, str], EntityOverlap] = {}
        
    def find_matches(self, ground_truth: List['Entity'], detected: List['Entity']) -> Dict[str, Set['Entity']]:
        matches = {
            MatchType.EXACT.value: set(),
            MatchType.PARTIAL.value: set(),
            MatchType.SENSITIVITY_MISMATCH.value: set(),
            MatchType.TYPE_MISMATCH.value: set(),
            MatchType.NO_MATCH.value: set(),
            "unmatched_ground_truth": set(ground_truth),
            "unmatched_detected": set(detected)
        }
    
        gt_spatial_index = self._create_spatial_index(ground_truth)
        
        for det_entity in detected:
            candidates = self._get_candidates(det_entity, gt_spatial_index)
            best_match = self._find_best_match(det_entity, candidates)
            
            if best_match:
                match_type = self._determine_match_type(best_match)
                matches[match_type.value].add(det_entity)
                
                # Only try to remove if entity exists in sets
                if det_entity in matches["unmatched_detected"]:
                    matches["unmatched_detected"].remove(det_entity)
                if best_match.ground_truth in matches["unmatched_ground_truth"]:
                    matches["unmatched_ground_truth"].remove(best_match.ground_truth)
    
        return matches
    
    def _create_spatial_index(self, entities: List['Entity']) -> Dict[int, Set['Entity']]:
        """
        Create simple spatial index for character positions
        
        Args:
            entities: List of entities to index
            
        Returns:
            Dictionary mapping character positions to entities
        """
        index = defaultdict(set)
        for entity in entities:
            for pos in range(entity.start_char, entity.end_char):
                index[pos].add(entity)
        return index
    
    def _get_candidates(self, 
                       entity: 'Entity',
                       spatial_index: Dict[int, Set['Entity']]) -> Set['Entity']:
        """
        Get candidate entities that could potentially match based on position
        
        Args:
            entity: Entity to find candidates for
            spatial_index: Spatial index of ground truth entities
            
        Returns:
            Set of candidate entities that might match
        """
        candidates = set()
        for pos in range(entity.start_char, entity.end_char):
            candidates.update(spatial_index.get(pos, set()))
        return candidates
    
    def _find_best_match(self,
                        detected: 'Entity',
                        candidates: Set['Entity']) -> Optional[EntityOverlap]:
        """
        Find the best matching ground truth entity among candidates
        
        Args:
            detected: Detected entity to match
            candidates: Set of candidate ground truth entities
            
        Returns:
            Best matching EntityOverlap or None if no good match found
        """
        best_match = None
        best_overlap = 0
        
        for gt_entity in candidates:
            cache_key = (self._entity_key(gt_entity), self._entity_key(detected))
            
            # Check cache first
            if cache_key in self.match_cache:
                overlap = self.match_cache[cache_key]
            else:
                overlap = self._calculate_overlap(gt_entity, detected)
                self.match_cache[cache_key] = overlap
            
            if overlap.overlap_ratio > best_overlap:
                best_overlap = overlap.overlap_ratio
                best_match = overlap
                
        return best_match
    
    def _calculate_overlap(self,
                         gt_entity: 'Entity',
                         detected: 'Entity') -> EntityOverlap:
        """
        Calculate detailed overlap information between two entities
        
        Args:
            gt_entity: Ground truth entity
            detected: Detected entity
            
        Returns:
            EntityOverlap object with overlap details
        """
        # Calculate character overlap
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected.start_char, detected.end_char))
        
        char_overlap = len(gt_chars & det_chars)
        char_union = len(gt_chars | det_chars)
        
        overlap_ratio = char_overlap / char_union if char_union > 0 else 0
        
        return EntityOverlap(
            ground_truth=gt_entity,
            detected=detected,
            char_overlap=char_overlap,
            overlap_ratio=overlap_ratio,
            type_match=gt_entity.entity_type == detected.entity_type,
            sensitivity_match=gt_entity.sensitivity == detected.sensitivity
        )
    
    def _determine_match_type(self, overlap: EntityOverlap) -> MatchType:
        """
        Determine the type of match based on overlap characteristics
        
        Args:
            overlap: EntityOverlap object to classify
            
        Returns:
            MatchType enum indicating the type of match
        """
        if overlap.overlap_ratio >= self.exact_match_threshold:
            if not overlap.type_match:
                return MatchType.TYPE_MISMATCH
            elif not overlap.sensitivity_match:
                return MatchType.SENSITIVITY_MISMATCH
            return MatchType.EXACT
        elif overlap.overlap_ratio >= self.partial_match_threshold:
            return MatchType.PARTIAL
        return MatchType.NO_MATCH
    
    @staticmethod
    def _entity_key(entity: 'Entity') -> str:
        """Create unique key for entity caching"""
        return f"{entity.start_char}:{entity.end_char}:{entity.entity_type}"

    def get_matching_statistics(self, matches: Dict[str, Set['Entity']]) -> Dict[str, int]:
        """
        Calculate statistics about entity matches
        
        Args:
            matches: Dictionary of match results from find_matches()
            
        Returns:
            Dictionary of match statistics
        """
        return {
            "exact_matches": len(matches[MatchType.EXACT.value]),
            "partial_matches": len(matches[MatchType.PARTIAL.value]),
            "sensitivity_mismatches": len(matches[MatchType.SENSITIVITY_MISMATCH.value]),
            "type_mismatches": len(matches[MatchType.TYPE_MISMATCH.value]),
            "unmatched_ground_truth": len(matches["unmatched_ground_truth"]),
            "unmatched_detected": len(matches["unmatched_detected"])
        }

    def to_json(self, matches: Dict[str, Set['Entity']]) -> str:
        """
        Convert match results to JSON format for storage/analysis
        
        Args:
            matches: Dictionary of match results
            
        Returns:
            JSON string representation of matches
        """
        # Convert sets to lists for JSON serialization
        json_dict = {
            k: [vars(e) for e in v] if isinstance(v, set) else v 
            for k, v in matches.items()
        }
        return json.dumps(json_dict, indent=2)