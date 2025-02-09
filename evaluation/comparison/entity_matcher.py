# evaluation/comparison/entity_matcher.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import json
from enum import Enum
import logging

from evaluation.models import Entity, SensitivityLevel

###############################################################################
# SECTION 2: MATCH TYPE ENUMERATION
###############################################################################
class MatchType(Enum):
    """
    Represents different types of entity matches.
    
    Defines the possible outcomes when comparing ground truth 
    and detected entities.
    """
    EXACT = "exact"
    PARTIAL = "partial"
    SENSITIVITY_MISMATCH = "sensitivity_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    NO_MATCH = "no_match"

###############################################################################
# SECTION 3: ENTITY OVERLAP DATA STRUCTURE
###############################################################################
@dataclass
class EntityOverlap:
    """
    Represents detailed overlap information between two entities.
    
    Attributes:
        ground_truth: Ground truth Entity
        detected: Detected Entity
        char_overlap: Number of overlapping characters
        overlap_ratio: Percentage of character overlap
        type_match: Whether entity types match
        sensitivity_match: Whether sensitivity levels match
    """
    ground_truth: Entity 
    detected: Entity
    char_overlap: int
    overlap_ratio: float
    type_match: bool
    sensitivity_match: Optional[bool] = None

###############################################################################
# SECTION 4: ENTITY MATCHER CLASS
###############################################################################
class EntityMatcher:
    """
    Advanced entity matching system for complex evaluation scenarios.
    
    Handles matching of ground truth and detected entities with 
    configurable thresholds and detailed matching strategies.
    """
    
    def __init__(self, 
                 exact_match_threshold: float = 0.9,
                 partial_match_threshold: float = 0.5,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize EntityMatcher with configurable matching thresholds.
        
        Args:
            exact_match_threshold: Minimum overlap ratio for exact match
            partial_match_threshold: Minimum overlap ratio for partial match
            logger: Optional logger for debugging
        """
        self.exact_match_threshold = exact_match_threshold
        self.partial_match_threshold = partial_match_threshold
        self.match_cache: Dict[Tuple[str, str], EntityOverlap] = {}
        self.logger = logger or logging.getLogger(__name__)

    def find_matches(self, ground_truth: List[Entity], detected: List[Entity]) -> Dict[str, Set[Entity]]:
        """
        Find matches between ground truth and detected entities.
        
        Args:
            ground_truth: List of ground truth entities
            detected: List of detected entities
        
        Returns:
            Dictionary of matched and unmatched entities
        """
        matches = {
            MatchType.EXACT.value: set(),
            MatchType.PARTIAL.value: set(),
            MatchType.SENSITIVITY_MISMATCH.value: set(),
            MatchType.TYPE_MISMATCH.value: set(),
            MatchType.NO_MATCH.value: set(),
            "unmatched_ground_truth": set(ground_truth),
            "unmatched_detected": set(detected)
        }
       
        # Create spatial index for efficient matching
        gt_spatial_index = self._create_spatial_index(ground_truth)
       
        # Match detected entities
        for det_entity in detected:
            candidates = self._get_candidates(det_entity, gt_spatial_index)
           
            # Find best match
            match_found = False
            for gt_entity in candidates:
                if self._is_match(gt_entity, det_entity):
                    matches[MatchType.EXACT.value].add(det_entity)
                    matches["unmatched_detected"].remove(det_entity)
                    matches["unmatched_ground_truth"].remove(gt_entity)
                    match_found = True
                    break
                elif self._is_partial_match(gt_entity, det_entity):
                    matches[MatchType.PARTIAL.value].add(det_entity)
                    match_found = True
                    break
            
            # Track unmatched entities
            if not match_found:
                matches[MatchType.NO_MATCH.value].add(det_entity)
       
        return matches

    
# In entity_matcher.py, modify find_matches method
def find_matches(self, ground_truth: List[Entity], detected: List[Entity]) -> Dict[str, Set[Entity]]:
    """
    Enhanced match finding with comprehensive debugging
    """
    print("\n=== DETAILED MATCH DEBUGGING ===")
    print(f"Ground Truth Entities: {len(ground_truth)}")
    print(f"Detected Entities: {len(detected)}")
    
    matches = {
        MatchType.EXACT.value: set(),
        MatchType.PARTIAL.value: set(),
        "unmatched_ground_truth": set(ground_truth),
        "unmatched_detected": set(detected)
    }
    
    # Debug each ground truth entity
    for gt_entity in ground_truth:
        print(f"\nAnalyzing Ground Truth Entity: {gt_entity}")
        print(f"  Type: {gt_entity.entity_type}")
        print(f"  Text: {gt_entity.text}")
        print(f"  Position: {gt_entity.start_char}-{gt_entity.end_char}")
        
        # Check against each detected entity
        match_found = False
        for det_entity in detected:
            print(f"\n  Comparing with Detected Entity: {det_entity}")
            print(f"    Type: {det_entity.entity_type}")
            print(f"    Text: {det_entity.text}")
            print(f"    Position: {det_entity.start_char}-{det_entity.end_char}")
            
            # Perform match check
            if self._is_match(gt_entity, det_entity):
                print("    ✅ EXACT MATCH FOUND")
                matches[MatchType.EXACT.value].add(det_entity)
                matches["unmatched_ground_truth"].remove(gt_entity)
                matches["unmatched_detected"].remove(det_entity)
                match_found = True
                break
            elif self._is_partial_match(gt_entity, det_entity):
                print("    🟨 PARTIAL MATCH FOUND")
                matches[MatchType.PARTIAL.value].add(det_entity)
                match_found = True
                break
        
        if not match_found:
            print("    ❌ NO MATCH FOUND")
    
    # Detailed summary
    print("\nMATCH SUMMARY:")
    print(f"Exact Matches: {len(matches[MatchType.EXACT.value])}")
    print(f"Partial Matches: {len(matches[MatchType.PARTIAL.value])}")
    print(f"Unmatched Ground Truth: {len(matches['unmatched_ground_truth'])}")
    print(f"Unmatched Detected: {len(matches['unmatched_detected'])}")
    
    return matches

# In _is_match method, add more verbose debugging
def _is_match(self, gt: Entity, detected: Entity) -> bool:
    print("\n--- DETAILED MATCH ANALYSIS ---")
    print(f"Ground Truth:  {gt.text} (Type: {gt.entity_type}, Pos: {gt.start_char}-{gt.end_char})")
    print(f"Detected:     {detected.text} (Type: {detected.entity_type}, Pos: {detected.start_char}-{detected.end_char})")
    
    # Normalize type and text
    def normalize(text: str) -> str:
        return ''.join(c.lower() for c in text if c.isalnum())
    
    # Type matching
    norm_gt_type = normalize(gt.entity_type)
    norm_det_type = normalize(detected.entity_type)
    type_match = norm_gt_type == norm_det_type
    print(f"Type Match: {type_match} ({norm_gt_type} vs {norm_det_type})")
    if not type_match:
        return False
    
    # Text matching
    norm_gt_text = normalize(gt.text)
    norm_det_text = normalize(detected.text)
    text_match = norm_gt_text == norm_det_text
    print(f"Text Match: {text_match} ({norm_gt_text} vs {norm_det_text})")
    if not text_match:
        return False
    
    # Overlap calculation
    overlap_ratio = self._calculate_overlap_ratio(gt, detected)
    print(f"Overlap Ratio: {overlap_ratio:.2f}")
    print(f"Threshold: {self.exact_match_threshold}")
    
    match_result = overlap_ratio >= self.exact_match_threshold
    print(f"Final Match Result: {match_result}")
    
    return match_result

    
    def _is_partial_match(self, gt: Entity, detected: Entity) -> bool:
        """
        Determine if two entities have a partial match.
        
        Args:
            gt: Ground truth entity
            detected: Detected entity
        
        Returns:
            Boolean indicating partial match
        """
        # Type must match for partial match
        if gt.entity_type != detected.entity_type:
            return False
        
        # Overlap ratio between 0.3 and 0.5
        overlap_ratio = self._calculate_overlap_ratio(gt, detected)
        return 0.3 < overlap_ratio <= 0.5

    def _calculate_overlap_ratio(self, e1: Entity, e2: Entity) -> float:
        """
        Calculate the character overlap ratio between two entities.
        
        Args:
            e1: First entity
            e2: Second entity
        
        Returns:
            Ratio of character overlap (0.0 to 1.0)
        """
        # Ensure valid character ranges
        start1, end1 = e1.start_char, e1.end_char
        start2, end2 = e2.start_char, e2.end_char
    
        # Find overlap region
        start = max(start1, start2)
        end = min(end1, end2)
    
        # No overlap case
        if end <= start:
            return 0.0
    
        # Calculate overlap
        overlap = end - start
        total = max(end1, end2) - min(start1, start2)
    
        return overlap / total

    def _create_spatial_index(self, entities: List[Entity]) -> Dict[int, Set[Entity]]:
        """
        Create a spatial index of entities by character position.
        
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
                       entity: Entity,
                       spatial_index: Dict[int, Set[Entity]]) -> Set[Entity]:
        """
        Find potential matching entities based on spatial index.
        
        Args:
            entity: Entity to find candidates for
            spatial_index: Spatial index of ground truth entities
        
        Returns:
            Set of candidate entities
        """
        candidates = set()
        for pos in range(entity.start_char, entity.end_char):
            candidates.update(spatial_index.get(pos, set()))
        return candidates

    def _log_match_analysis(self, gt: Entity, detected: Entity, overlap_ratio: float) -> None:
        """
        Log detailed match analysis for debugging.
        
        Args:
            gt: Ground truth entity
            detected: Detected entity
            overlap_ratio: Calculated overlap ratio
        """
        if not self.logger.isEnabledFor(logging.DEBUG):
            return
        
        self.logger.debug("\n=== ENTITY MATCH ANALYSIS ===")
        self.logger.debug(f"Ground Truth: {gt}")
        self.logger.debug(f"Detected:    {detected}")
        
        # Detailed match criteria logging
        self.logger.debug(f"\nType Comparison:")
        self.logger.debug(f"  Ground Truth Type: {gt.entity_type}")
        self.logger.debug(f"  Detected Type:     {detected.entity_type}")
        
        self.logger.debug(f"\nOverlap Analysis:")
        self.logger.debug(f"  Ground Truth Start: {gt.start_char}")
        self.logger.debug(f"  Ground Truth End:   {gt.end_char}")
        self.logger.debug(f"  Detected Start:     {detected.start_char}")
        self.logger.debug(f"  Detected End:       {detected.end_char}")
        self.logger.debug(f"  Overlap Ratio:      {overlap_ratio}")

    def get_matching_statistics(self, matches: Dict[str, Set[Entity]]) -> Dict[str, int]:
        """
        Calculate comprehensive statistics about entity matches.
        
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

    def to_json(self, matches: Dict[str, Set[Entity]]) -> str:
        """
        Convert match results to JSON format for storage/analysis.
        
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

###############################################################################
# END OF FILE
###############################################################################