# evaluation/metrics/entity_metrics.py

###############################################################################
# SECTION 1: IMPORTS AND DEPENDENCIES
###############################################################################
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from enum import Enum

from evaluation.models import Entity, SensitivityLevel
from evaluation.comparison.entity_matcher import MatchType
from app.utils.config_loader import ConfigLoader

###############################################################################
# SECTION 2: DATA STRUCTURES
###############################################################################
@dataclass
class EntityMatchResult:
    """
    Comprehensive result of entity matching process.
    
    Attributes:
        true_positives: Entities correctly identified
        false_positives: Detected entities not in ground truth
        false_negatives: Ground truth entities not detected
        partial_matches: Entities with significant but incomplete overlap
    """
    true_positives: Set[Entity]
    false_positives: Set[Entity]
    false_negatives: Set[Entity]
    partial_matches: Set[tuple[Entity, Entity]]

###############################################################################
# SECTION 3: ENTITY MATCHING CONFIGURATION
###############################################################################
class OverlapStrategy(Enum):
    """
    Strategies for determining entity match overlap.
    
    - STRICT: Requires high percentage of character overlap
    - LENIENT: Allows more flexible matching
    - CUSTOM: Allows custom threshold configuration
    """
    STRICT = 0.7
    LENIENT = 0.5
    CUSTOM = None

###############################################################################
# SECTION 4: MAIN METRICS CALCULATION CLASS
###############################################################################
class EntityMetrics:
    """Calculates performance metrics for entity detection systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_loader = ConfigLoader()
        self.entity_config = self._load_entity_config(config_path)
        self.results_by_type: Dict[str, EntityMatchResult] = {}
        self.results_by_sensitivity: Dict[SensitivityLevel, EntityMatchResult] = {}

    def _load_entity_config(self, config_path: Optional[str] = None) -> Dict:
        config = self.config_loader.get_config("entity_routing")
        if config is None:
            import yaml
            config_file = config_path or "redactor/config/entity_routing.yaml"
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)
        return config

    def calculate_metrics(self, ground_truth: List[Entity], detected: List[Entity], 
                         overlap_threshold: float = OverlapStrategy.STRICT.value) -> Dict[str, float]:
        print("\n=== METRICS CALCULATION ===")
        print(f"Ground Truth Entities: {len(ground_truth)}")
        print(f"Detected Entities: {len(detected)}")
        
        # Use debug match analysis first
        self.debug_match_analysis(ground_truth, detected)

        """Calculate comprehensive detection metrics."""
        match_result = self._match_entities(ground_truth, detected, overlap_threshold)
        
        print("\n--- MATCH RESULT DETAILS ---")
        print(f"True Positives: {len(match_result.true_positives)}")
        print(f"False Positives: {len(match_result.false_positives)}")
        print(f"False Negatives: {len(match_result.false_negatives)}")
        print(f"Partial Matches: {len(match_result.partial_matches)}")    
       
        
        # Calculate precision
        precision = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_positives)
        ) if (len(match_result.true_positives) + len(match_result.false_positives)) > 0 else 0
        
        # Calculate recall
        recall = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_negatives)
        ) if (len(match_result.true_positives) + len(match_result.false_negatives)) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "partial_matches": len(match_result.partial_matches),
            "false_positives": len(match_result.false_positives),
            "false_negatives": len(match_result.false_negatives)
        }

    def _is_matching_entity(self, gt_entity: Entity, detected_entity: Entity, threshold: float = 0.7) -> bool:
        # Log ALL details about both entities
        print("\n" + "="*80)
        print("COMPREHENSIVE ENTITY MATCHING ANALYSIS")
        print("="*80)
        
        print("Ground Truth Entity:")
        print(f"  Text: '{gt_entity.text}'")
        print(f"  Type: '{gt_entity.entity_type}'")
        print(f"  Start: {gt_entity.start_char}")
        print(f"  End: {gt_entity.end_char}")
        
        print("\nDetected Entity:")
        print(f"  Text: '{detected_entity.text}'")
        print(f"  Type: '{detected_entity.entity_type}'")
        print(f"  Start: {detected_entity.start_char}")
        print(f"  End: {detected_entity.end_char}")
        
        # Normalization functions with detailed logging
        def normalize_type(type_str: str) -> str:
            normalized = type_str.replace('_', '').upper().strip()
            print(f"  Normalized Type: '{normalized}'")
            return normalized
        
        def normalize_text(text: str) -> str:
            normalized = ''.join(c.lower() for c in text if c.isalnum())
            print(f"  Normalized Text: '{normalized}'")
            return normalized
        
        # Type comparison with detailed logging
        norm_gt_type = normalize_type(gt_entity.entity_type)
        norm_det_type = normalize_type(detected_entity.entity_type)
        
        if norm_gt_type != norm_det_type:
            print("❌ TYPE MISMATCH")
            return False
        
        # Text comparison with detailed logging
        norm_gt_text = normalize_text(gt_entity.text)
        norm_det_text = normalize_text(detected_entity.text)
        
        if norm_gt_text != norm_det_text:
            print("❌ TEXT MISMATCH")
            return False
        
        # Position and overlap analysis
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        union = len(gt_chars | det_chars)
        
        overlap_ratio = overlap / union if union > 0 else 0
        
        print("\nOverlap Analysis:")
        print(f"  Overlap:       {overlap}")
        print(f"  Union:         {union}")
        print(f"  Overlap Ratio: {overlap_ratio:.2f}")
        print(f"  Threshold:     {threshold}")
        
        # Match conditions
        match_conditions = [
            overlap_ratio >= threshold,
            abs(gt_entity.start_char - detected_entity.start_char) <= 2,
            abs(gt_entity.end_char - detected_entity.end_char) <= 2
        ]
        
        match_result = all(match_conditions)
        
        print("\nMatch Conditions:")
        condition_names = [
            "Overlap Ratio >= Threshold",
            "Start Character Close",
            "End Character Close"
        ]
        for name, condition in zip(condition_names, match_conditions):
            print(f"  {name}: {'✅ PASSED' if condition else '❌ FAILED'}")
        
        print(f"\nFinal Match Result: {'✅ MATCH' if match_result else '❌ NO MATCH'}")
        
        return match_result

    def _is_partial_match(self, gt_entity: Entity, detected_entity: Entity, threshold: float = 0.5) -> bool:
        """Check for partial entity matches."""
        if gt_entity.entity_type != detected_entity.entity_type:
            return False
            
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        union = len(gt_chars | det_chars)
        
        overlap_ratio = overlap / union if union > 0 else 0
        return 0.3 <= overlap_ratio < threshold

    def _match_entities(self, ground_truth: List[Entity], detected: List[Entity], 
                       overlap_threshold: float) -> EntityMatchResult:
        """Match ground truth entities with detected entities."""
        true_positives = set()
        false_positives = set(detected)
        false_negatives = set(ground_truth)
        partial_matches = set()
    
        for gt_entity in ground_truth:
            for det_entity in detected:
                if self._is_matching_entity(gt_entity, det_entity, overlap_threshold):
                    true_positives.add(det_entity)
                    false_positives.discard(det_entity)
                    false_negatives.discard(gt_entity)
                elif self._is_partial_match(gt_entity, det_entity, overlap_threshold):
                    partial_matches.add((gt_entity, det_entity))
    
        return EntityMatchResult(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            partial_matches=partial_matches
        )

    
    ###########################################################################
    # SECTION 5: MATCHING AND COMPARISON METHODS
    ###########################################################################

    def _is_matching_entity(self, gt_entity: Entity, detected_entity: Entity, threshold: float = 0.7) -> bool:
        def detailed_entity_repr(entity):
            return (
                f"Entity(text='{entity.text}', "
                f"type='{entity.entity_type}', "
                f"start={entity.start_char}, "
                f"end={entity.end_char}, "
                f"confidence={getattr(entity, 'confidence', 1.0)}, "
                f"sensitivity={getattr(entity, 'sensitivity', 'N/A')})"
            )
    
        print("\n" + "="*80)
        print("COMPREHENSIVE ENTITY MATCHING ANALYSIS")
        print("="*80)
        
        print("Ground Truth Entity:")
        print(detailed_entity_repr(gt_entity))
        
        print("\nDetected Entity:")
        print(detailed_entity_repr(detected_entity))
        
        # Normalization functions
        def normalize_type(type_str: str) -> str:
            return type_str.replace('_', '').upper().strip()
        
        def normalize_text(text: str) -> str:
            return ''.join(c.lower() for c in text if c.isalnum())
        
        # Type comparison
        norm_gt_type = normalize_type(gt_entity.entity_type)
        norm_det_type = normalize_type(detected_entity.entity_type)
        
        print("\nType Comparison:")
        print(f"  Original GT Type:   '{gt_entity.entity_type}'")
        print(f"  Original Det Type:  '{detected_entity.entity_type}'")
        print(f"  Normalized GT Type: '{norm_gt_type}'")
        print(f"  Normalized Det Type:'{norm_det_type}'")
        
        if norm_gt_type != norm_det_type:
            print("❌ TYPE MISMATCH")
            return False
        
        # Text comparison
        norm_gt_text = normalize_text(gt_entity.text)
        norm_det_text = normalize_text(detected_entity.text)
        
        print("\nText Comparison:")
        print(f"  Original GT Text:   '{gt_entity.text}'")
        print(f"  Original Det Text:  '{detected_entity.text}'")
        print(f"  Normalized GT Text: '{norm_gt_text}'")
        print(f"  Normalized Det Text:'{norm_det_text}'")
        
        if norm_gt_text != norm_det_text:
            print("❌ TEXT MISMATCH")
            return False
        
        # Position analysis
        print("\nPosition Analysis:")
        print(f"  GT Start:   {gt_entity.start_char}")
        print(f"  Det Start:  {detected_entity.start_char}")
        print(f"  GT End:     {gt_entity.end_char}")
        print(f"  Det End:    {detected_entity.end_char}")
        
        start_diff = abs(gt_entity.start_char - detected_entity.start_char)
        end_diff = abs(gt_entity.end_char - detected_entity.end_char)
        
        print(f"  Start Difference: {start_diff}")
        print(f"  End Difference:   {end_diff}")
        
        # Character overlap calculation
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        union = len(gt_chars | det_chars)
        
        overlap_ratio = overlap / union if union > 0 else 0
        
        print(f"\nOverlap Analysis:")
        print(f"  Overlap:        {overlap}")
        print(f"  Union:          {union}")
        print(f"  Overlap Ratio:  {overlap_ratio}")
        print(f"  Threshold:      {threshold}")
        
        # Flexible matching criteria
        match_conditions = [
            start_diff <= 2,   # Allow 2-character difference in start
            end_diff <= 2,     # Allow 2-character difference in end
            overlap_ratio >= threshold  # Ensure good character overlap
        ]
        
        match_result = all(match_conditions)
        
        print("\nMatch Conditions:")
        for i, condition in enumerate(match_conditions, 1):
            print(f"  Condition {i}: {'✅ PASSED' if condition else '❌ FAILED'}")
        
        print(f"\nFinal Match Result: {'✅ MATCH' if match_result else '❌ NO MATCH'}")
        
        return match_result

    def debug_match_analysis(self, ground_truth: List[Entity], detected: List[Entity]) -> None:
        """Detailed analysis of matching results with comprehensive output."""
        print("\n=== ENTITY MATCHING DEBUG ANALYSIS ===")
        print(f"Ground Truth Entities: {len(ground_truth)}")
        print(f"Detected Entities: {len(detected)}")
        
        for i, gt_entity in enumerate(ground_truth):
            print(f"\nAnalyzing Ground Truth Entity {i+1}:")
            match_found = False
            
            for j, det_entity in enumerate(detected):
                print(f"\nComparing with Detected Entity {j+1}:")
                match_found = self._is_matching_entity(gt_entity, det_entity)
                
                if match_found:
                    print("✅ MATCH FOUND - Moving to next ground truth entity")
                    break
            
            if not match_found:
                print("❌ NO MATCHES FOUND for this ground truth entity")


    ###########################################################################
    # SECTION 5: MATCHING AND COMPARISON METHODS
    ###########################################################################

    def _is_matching_entity(self, gt_entity: Entity, detected_entity: Entity, threshold: float = 0.7) -> bool:
        def detailed_entity_repr(entity):
            return (
                f"Entity(text='{entity.text}', "
                f"type='{entity.entity_type}', "
                f"start={entity.start_char}, "
                f"end={entity.end_char}, "
                f"confidence={getattr(entity, 'confidence', 1.0)}, "
                f"sensitivity={getattr(entity, 'sensitivity', 'N/A')})"
            )
    
        print("\n" + "="*80)
        print("COMPREHENSIVE ENTITY MATCHING ANALYSIS")
        print("="*80)
        
        print("Ground Truth Entity:")
        print(detailed_entity_repr(gt_entity))
        
        print("\nDetected Entity:")
        print(detailed_entity_repr(detected_entity))
        
        # Normalization functions
        def normalize_type(type_str: str) -> str:
            return type_str.replace('_', '').upper().strip()
        
        def normalize_text(text: str) -> str:
            return ''.join(c.lower() for c in text if c.isalnum())
        
        # Type comparison
        norm_gt_type = normalize_type(gt_entity.entity_type)
        norm_det_type = normalize_type(detected_entity.entity_type)
        
        print("\nType Comparison:")
        print(f"  Original GT Type:   '{gt_entity.entity_type}'")
        print(f"  Original Det Type:  '{detected_entity.entity_type}'")
        print(f"  Normalized GT Type: '{norm_gt_type}'")
        print(f"  Normalized Det Type:'{norm_det_type}'")
        
        if norm_gt_type != norm_det_type:
            print("❌ TYPE MISMATCH")
            return False
        
        # Text comparison
        norm_gt_text = normalize_text(gt_entity.text)
        norm_det_text = normalize_text(detected_entity.text)
        
        print("\nText Comparison:")
        print(f"  Original GT Text:   '{gt_entity.text}'")
        print(f"  Original Det Text:  '{detected_entity.text}'")
        print(f"  Normalized GT Text: '{norm_gt_text}'")
        print(f"  Normalized Det Text:'{norm_det_text}'")
        
        if norm_gt_text != norm_det_text:
            print("❌ TEXT MISMATCH")
            return False
        
        # Position analysis
        print("\nPosition Analysis:")
        print(f"  GT Start:   {gt_entity.start_char}")
        print(f"  Det Start:  {detected_entity.start_char}")
        print(f"  GT End:     {gt_entity.end_char}")
        print(f"  Det End:    {detected_entity.end_char}")
        
        start_diff = abs(gt_entity.start_char - detected_entity.start_char)
        end_diff = abs(gt_entity.end_char - detected_entity.end_char)
        
        print(f"  Start Difference: {start_diff}")
        print(f"  End Difference:   {end_diff}")
        
        # Character overlap calculation
        gt_chars = set(range(gt_entity.start_char, gt_entity.end_char))
        det_chars = set(range(detected_entity.start_char, detected_entity.end_char))
        
        overlap = len(gt_chars & det_chars)
        union = len(gt_chars | det_chars)
        
        overlap_ratio = overlap / union if union > 0 else 0
        
        print(f"\nOverlap Analysis:")
        print(f"  Overlap:        {overlap}")
        print(f"  Union:          {union}")
        print(f"  Overlap Ratio:  {overlap_ratio}")
        print(f"  Threshold:      {threshold}")
        
        # Flexible matching criteria
        match_conditions = [
            start_diff <= 2,   # Allow 2-character difference in start
            end_diff <= 2,     # Allow 2-character difference in end
            overlap_ratio >= threshold  # Ensure good character overlap
        ]
        
        match_result = all(match_conditions)
        
        print("\nMatch Conditions:")
        for i, condition in enumerate(match_conditions, 1):
            print(f"  Condition {i}: {'✅ PASSED' if condition else '❌ FAILED'}")
        
        print(f"\nFinal Match Result: {'✅ MATCH' if match_result else '❌ NO MATCH'}")
        
        return match_result

    def debug_match_analysis(self, ground_truth: List[Entity], detected: List[Entity]) -> None:
        """Detailed analysis of matching results with comprehensive output."""
        print("\n=== ENTITY MATCHING DEBUG ANALYSIS ===")
        print(f"Ground Truth Entities: {len(ground_truth)}")
        print(f"Detected Entities: {len(detected)}")
        
        for i, gt_entity in enumerate(ground_truth):
            print(f"\nAnalyzing Ground Truth Entity {i+1}:")
            match_found = False
            
            for j, det_entity in enumerate(detected):
                print(f"\nComparing with Detected Entity {j+1}:")
                match_found = self._is_matching_entity(gt_entity, det_entity)
                
                if match_found:
                    print("✅ MATCH FOUND - Moving to next ground truth entity")
                    break
            
            if not match_found:
                print("❌ NO MATCHES FOUND for this ground truth entity")

        
    ###########################################################################
    # SECTION 6: METRIC CALCULATION METHODS
    ###########################################################################
    def calculate_metrics(self, 
                           ground_truth: List[Entity], 
                           detected: List[Entity],
                           overlap_threshold: float = OverlapStrategy.STRICT.value) -> Dict[str, float]:
        """
        Calculate comprehensive detection metrics.
        
        Args:
            ground_truth: List of ground truth entities
            detected: List of detected entities
            overlap_threshold: Minimum overlap for match
        
        Returns:
            Dictionary of performance metrics
        """
        match_result = self._match_entities(ground_truth, detected, overlap_threshold)
        
        # Calculate precision
        precision = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_positives)
        ) if (len(match_result.true_positives) + len(match_result.false_positives)) > 0 else 0
        
        # Calculate recall
        recall = len(match_result.true_positives) / (
            len(match_result.true_positives) + len(match_result.false_negatives)
        ) if (len(match_result.true_positives) + len(match_result.false_negatives)) > 0 else 0
        
        # Calculate F1 score
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
        Match ground truth entities with detected entities.
        
        Args:
            ground_truth: List of ground truth entities
            detected: List of detected entities
            overlap_threshold: Minimum overlap for match
        
        Returns:
            Comprehensive match result
        """
        # Initialize match tracking sets
        true_positives = set()
        false_positives = set(detected)
        false_negatives = set(ground_truth)
        partial_matches = set()
    
        # Compare each ground truth entity with detected entities
        for gt_entity in ground_truth:
            for det_entity in detected:
                # Check for exact match
                if self._is_matching_entity(gt_entity, det_entity, overlap_threshold):
                    true_positives.add(det_entity)
                    false_positives.discard(det_entity)
                    false_negatives.discard(gt_entity)
                # Check for partial match
                elif self._is_partial_match(gt_entity, det_entity):
                    partial_matches.add((gt_entity, det_entity))
    
        return EntityMatchResult(
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            partial_matches=partial_matches
        )

    ###########################################################################
    # SECTION 7: DETECTOR AND TYPE-BASED METRIC ANALYSIS
    ###########################################################################
    def get_type_based_metrics(self, results):
        """
        Generate metrics broken down by detector type.
        
        Args:
            results: Matching results to analyze
        
        Returns:
            Metrics categorized by detector type
        """
        presidio_entities = set(self.entity_config["routing"]["presidio_primary"]["entities"])
        spacy_entities = set(self.entity_config["routing"]["spacy_primary"]["entities"])
        ensemble_entities = set(self.entity_config["routing"]["ensemble_required"]["entities"])
    
        return {
            "presidio_detections": self._calculate_metrics_for_types(results, presidio_entities),
            "spacy_detections": self._calculate_metrics_for_types(results, spacy_entities),
            "ensemble_detections": self._calculate_metrics_for_types(results, ensemble_entities),
        }

    def _calculate_metrics_for_types(self, matches_dict: Dict, entity_types: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Calculate detailed metrics for specific entity types.
        
        Args:
            matches_dict: Dictionary of matching results
            entity_types: List of entity types to analyze
        
        Returns:
            Detailed metrics for specified entity types
        """
        metrics_by_type = {}
        
        for entity_type in entity_types:
            # Filter matches by entity type
            exact_matches = {e for e in matches_dict.get(MatchType.EXACT.value, set()) 
                            if e.entity_type == entity_type}
            partial = {e for e in matches_dict.get(MatchType.PARTIAL.value, set())
                       if e.entity_type == entity_type}
            false_positives = {e for e in matches_dict.get("unmatched_detected", set())
                               if e.entity_type == entity_type}
            false_negatives = {e for e in matches_dict.get("unmatched_ground_truth", set())
                               if e.entity_type == entity_type}
    
            # Calculate metrics
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

    ###########################################################################
    # SECTION 8: AGGREGATION AND REPORTING METHODS
    ###########################################################################
    def _aggregate_detector_metrics(self, detector_metrics: List[Dict]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across multiple detection runs.
        
        Args:
            detector_metrics: List of metric dictionaries from different runs
        
        Returns:
            Aggregated metrics
        """
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
    
        # Calculate average metrics
        for entity_type in aggregated:
            metrics = aggregated[entity_type]
            metrics["precision"] = sum(metrics["precision"]) / len(metrics["precision"])
            metrics["recall"] = sum(metrics["recall"]) / len(metrics["recall"])
            
        return aggregated