# evaluation/metrics/entity_metrics.py

###############################################################################
# SECTION 1: IMPORTS AND DEPENDENCIES
###############################################################################
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Set, Optional
from app.utils.config_loader import ConfigLoader
from evaluation.models import Entity, SensitivityLevel
from evaluation.comparison.entity_matcher import EntityMatcher, MatchType

###############################################################################
# SECTION 2: DATA STRUCTURES AND TYPES
###############################################################################
@dataclass
class EntityMatchResult:
    """Comprehensive result of entity matching process."""
    true_positives: Set[Entity]
    false_positives: Set[Entity]
    false_negatives: Set[Entity]
    partial_matches: Set[tuple[Entity, Entity]]

class OverlapStrategy(Enum):
    """Strategies for determining entity match overlap."""
    STRICT = 0.7
    LENIENT = 0.5
    CUSTOM = None

###############################################################################
# SECTION 3: ANALYSIS UTILITIES 
###############################################################################
def _get_context_window(self, entity: Entity, window_size: int = 50) -> str:
    """Get text context window around entity."""
    start = max(0, entity.start_char - window_size)
    end = min(len(self.text), entity.end_char + window_size)
    return self.text[start:end]

def _get_validation_details(self, entity: Entity) -> Dict:
    """Get validation details for an entity based on its type."""
    validation_rules = self.validation_params.get(entity.entity_type, {})
    return {
        "rules_applied": validation_rules,
        "confidence": getattr(entity, "confidence", 1.0),
        "validation_output": self._check_validation_rules(entity, validation_rules)
    }

def _check_validation_rules(self, entity: Entity, rules: Dict) -> Dict:
    """Check entity against type-specific validation rules."""
    results = {}
    if "core_terms" in rules:
        results["core_terms_found"] = [
            term for term in rules["core_terms"] 
            if term.lower() in entity.text.lower()
        ]
    if "min_words" in rules:
        results["word_count"] = len(entity.text.split())
        results["meets_min_words"] = results["word_count"] >= rules["min_words"]
    
    return results


    
###############################################################################
# SECTION 4: MAIN METRICS CALCULATION CLASS
###############################################################################
class EntityMetrics:
    """Calculates performance metrics for entity detection systems."""
    
    def __init__(self, config_loader: Optional[ConfigLoader] = None):
        self.config_loader = config_loader or ConfigLoader()
        self.validation_params = self.config_loader.get_config("validation_params")
        self.entity_routing = self.config_loader.get_config("entity_routing")
        self.entity_matcher = EntityMatcher()
        self.results_by_type: Dict[str, EntityMatchResult] = {}
        self.results_by_sensitivity: Dict[SensitivityLevel, EntityMatchResult] = {}

    def _is_matching_entity(self, gt: Entity, detected: Entity) -> bool:
        """
        Determine if the detected entity matches the ground truth entity.
        
        Uses both exact and partial matching criteria from the EntityMatcher.
        """
        return (self.entity_matcher._is_match(gt, detected) or 
                self.entity_matcher._is_partial_match(gt, detected))
        

    def analyze_entity_type_metrics(self, ground_truth: List[Entity], 
                                  detected: List[Entity], 
                                  target_entities: List[str] = None) -> Dict:
        """Analyze metrics for specific entity types."""
        if target_entities:
            gt_entities = [e for e in ground_truth if e.entity_type in target_entities]
            det_entities = [e for e in detected if e.entity_type in target_entities]
        else:
            gt_entities = ground_truth
            det_entities = detected
        
        matches = self.entity_matcher.find_matches(gt_entities, det_entities)
        match_stats = self.entity_matcher.get_matching_statistics(matches)
        base_metrics = self.calculate_metrics(gt_entities, det_entities)
        
        results = {}
        for entity_type in (target_entities or set(e.entity_type for e in ground_truth)):
            validation_rules = self.validation_params.get(entity_type, {})
            routing_info = self._get_routing_info(entity_type)
            
            results[entity_type] = {
                "metrics": base_metrics,
                "detector_info": {
                    "entity_type": entity_type,
                    "detector_type": routing_info["detector"],
                    "confidence_threshold": routing_info["threshold"],
                    "validation_rules": validation_rules
                },
                "matches": matches,
                "match_statistics": match_stats,
                "ground_truth": self._analyze_entities([e for e in gt_entities 
                                                      if e.entity_type == entity_type]),
                "detected": self._analyze_entities([e for e in det_entities 
                                                  if e.entity_type == entity_type]),
                "errors": {
                    "missed_entities": self._analyze_missed_entities(
                        [e for e in gt_entities if e.entity_type == entity_type],
                        [e for e in det_entities if e.entity_type == entity_type]
                    ),
                    "false_positives": self._analyze_false_positives(
                        [e for e in gt_entities if e.entity_type == entity_type],
                        [e for e in det_entities if e.entity_type == entity_type]
                    )
                }
            }
        
        return results

    def calculate_metrics(self, ground_truth: List[Entity], detected: List[Entity]) -> Dict[str, float]:
        """Calculate precision, recall, F1."""
        matches = self.entity_matcher.find_matches(ground_truth, detected)
        
        # Calculate metrics
        tp = len(matches.get(MatchType.EXACT.value, set()))
        fp = len(matches.get("unmatched_detected", set()))
        fn = len(matches.get("unmatched_ground_truth", set()))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "partial_matches": len(matches.get(MatchType.PARTIAL.value, set()))
        }

    def _get_routing_info(self, entity_type: str) -> Dict:
        """Get detector routing information for entity type."""
        routing = self.entity_routing["routing"]
        
        if entity_type in routing["presidio_primary"]["entities"]:
            return {
                "detector": "presidio",
                "threshold": routing["presidio_primary"]["thresholds"].get(entity_type, 0.7)
            }
        elif entity_type in routing["spacy_primary"]["entities"]:
            return {
                "detector": "spacy",
                "threshold": routing["spacy_primary"]["thresholds"].get(entity_type, 0.6)
            }
        else:
            thresholds = routing["ensemble_required"]["confidence_thresholds"].get(entity_type, {})
            return {
                "detector": "ensemble",
                "threshold": thresholds.get("minimum_combined", 0.7)
            }

    def _analyze_entities(self, entities: List[Entity]) -> Dict:
        """Analyze entities with validation details."""
        return {
            "count": len(entities),
            "entities": [{
                "text": e.text,
                "confidence": getattr(e, "confidence", 1.0),
                "position": {
                    "start": e.start_char,
                    "end": e.end_char
                },
                "validation": self._get_validation_details(e)
            } for e in entities]
        }
        
    def _get_validation_details(self, entity: Entity) -> Dict:
        """Get validation details for an entity with type-specific rules."""
        validation_rules = self.validation_params.get(entity.entity_type, {})
        return {
            "rules_applied": validation_rules,
            "confidence": getattr(entity, "confidence", 1.0),
            "validation_output": {
                "core_terms": validation_rules.get("core_terms", []),
                "word_count": len(entity.text.split()),
                "meets_min_words": len(entity.text.split()) >= validation_rules.get("min_words", 0),
                "entity_type": entity.entity_type
            }
        }

    def _analyze_missed_entities(self, gt_entities: List[Entity], 
                               detected_entities: List[Entity]) -> List[Dict]:
        """Analyze entities that were missed by detection."""
        missed = []
        for gt in gt_entities:
            if not any(self._is_matching_entity(gt, d) for d in detected_entities):
                validation_details = self._get_validation_details(gt)
                missed.append({
                    "text": gt.text,
                    "entity_type": gt.entity_type,
                    "position": {
                        "start": gt.start_char,
                        "end": gt.end_char
                    },
                    "validation": validation_details,
                    "possible_causes": self._analyze_miss_causes(gt)
                })
        return missed

    def _analyze_false_positives(self, gt_entities: List[Entity], 
                               detected_entities: List[Entity]) -> List[Dict]:
        """Analyze false positive detections."""
        false_positives = []
        for det in detected_entities:
            if not any(self._is_matching_entity(gt, det) for gt in gt_entities):
                validation_details = self._get_validation_details(det)
                false_positives.append({
                    "text": det.text,
                    "entity_type": det.entity_type,
                    "confidence": det.confidence,
                    "position": {
                        "start": det.start_char,
                        "end": det.end_char
                    },
                    "validation": validation_details,
                    "likely_causes": self._analyze_false_positive_causes(det)
                })
        return false_positives

    def _validate_entity(self, entity: Entity, target_entities: List[str] = None) -> Dict:
        """Streamlined validation for target entities only."""
        if target_entities and entity.entity_type not in target_entities:
            return {}
            
        validation_rules = self.validation_params.get(entity.entity_type, {})
        routing_info = self._get_routing_info(entity.entity_type)
        
        return {
            "entity_type": entity.entity_type,
            "text": entity.text,
            "confidence": entity.confidence,
            "rules_applied": validation_rules,
            "detector": routing_info["detector"],
            "threshold": routing_info["threshold"]
        }
    
    def validate_targeted_entities(self, entities: List[Entity], target_types: List[str]) -> Dict:
        """Validates only specified entity types with minimal output."""
        results = {}
        for entity in entities:
            if entity.entity_type in target_types:
                validation = {
                    "confidence": entity.confidence,
                    "validation_rules": self.validation_params.get(entity.entity_type, {}),
                    "detector": self._get_routing_info(entity.entity_type)["detector"]
                }
                results[entity.text] = validation
        return results
    
###############################################################################
# SECTION 5: ENTITY ANALYSIS METHODS 
###############################################################################
    def _analyze_entities(self, entities: List[Entity]) -> Dict:
        """Analyze a list of entities with their contexts."""
        return {
            "count": len(entities),
            "entities": [{
                "text": e.text,
                "confidence": getattr(e, "confidence", 1.0),
                "position": {
                    "start": e.start_char,
                    "end": e.end_char
                },
                "validation": self._get_validation_details(e)
            } for e in entities]
        }
    
    def _analyze_missed_entities(self, gt_entities: List[Entity], 
                               detected_entities: List[Entity]) -> List[Dict]:
        """Analyze entities that were missed by detection."""
        missed = []
        for gt in gt_entities:
            if not any(self._is_matching_entity(gt, d) for d in detected_entities):
                validation_details = self._get_validation_details(gt)
                missed.append({
                    "text": gt.text,
                    "position": {
                        "start": gt.start_char,
                        "end": gt.end_char
                    },
                    "validation": validation_details,
                    "possible_causes": self._analyze_miss_causes(gt)
                })
        return missed
    
    def _analyze_miss_causes(self, entity: Entity) -> List[str]:
        """Analyze potential causes for missed detection."""
        causes = []
        rules = self.validation_params.get(entity.entity_type, {})
        
        if "core_terms" in rules and not any(term.lower() in entity.text.lower() 
                                           for term in rules["core_terms"]):
            causes.append("No core terms found")
            
        if "min_words" in rules and len(entity.text.split()) < rules["min_words"]:
            causes.append(f"Below minimum word count ({rules['min_words']})")
            
        if "required_context" in rules and not any(ctx.lower() in entity.text.lower() 
                                                 for ctx in rules["required_context"]):
            causes.append("Missing required context")
            
        routing_info = self._get_routing_info(entity.entity_type)
        if hasattr(entity, 'confidence') and entity.confidence < routing_info["threshold"]:
            causes.append(f"Below confidence threshold ({routing_info['threshold']})")
        
        return causes
    
    def _analyze_false_positives(self, gt_entities: List[Entity], 
                               detected_entities: List[Entity]) -> List[Dict]:
        """Analyze false positive detections."""
        false_positives = []
        for det in detected_entities:
            if not any(self._is_matching_entity(gt, det) for gt in gt_entities):
                validation_details = self._get_validation_details(det)
                false_positives.append({
                    "text": det.text,
                    "confidence": det.confidence,
                    "position": {
                        "start": det.start_char,
                        "end": det.end_char
                    },
                    "validation": validation_details,
                    "likely_causes": self._analyze_false_positive_causes(det)
                })
        return false_positives
    
    def _analyze_false_positive_causes(self, entity: Entity) -> List[str]:
        """Analyze potential causes for false positive detection."""
        causes = []
        rules = self.validation_params.get(entity.entity_type, {})
        
        if "core_terms" in rules:
            matching_terms = [term for term in rules["core_terms"] 
                             if term.lower() in entity.text.lower()]
            if matching_terms:
                causes.append(f"Matched core terms: {', '.join(matching_terms)}")
        
        if hasattr(entity, 'confidence'):
            routing_info = self._get_routing_info(entity.entity_type)
            threshold = routing_info["threshold"]
            if entity.confidence < threshold + 0.1:
                causes.append(f"Borderline confidence: {entity.confidence:.2f} vs threshold {threshold}")
        
        context_words = rules.get("context_words", [])
        if context_words:
            matching_context = [word for word in context_words 
                              if word.lower() in entity.text.lower()]
            if matching_context:
                causes.append(f"Context word matches: {', '.join(matching_context)}")
        
        return causes

###############################################################################
# SECTION 6: TYPE-BASED METRICS ANALYSIS
###############################################################################
    def get_type_based_metrics(self, matches: Dict) -> Dict[str, Dict[str, float]]:
        """Calculate metrics broken down by entity type."""
        
        # Get entity types from routing config
        presidio_entities = set(self.entity_routing["routing"]["presidio_primary"]["entities"])
        spacy_entities = set(self.entity_routing["routing"]["spacy_primary"]["entities"])
        ensemble_entities = set(self.entity_routing["routing"]["ensemble_required"]["entities"])
    
        metrics_by_detector = {
            "presidio": self._calculate_metrics_for_types(matches, presidio_entities),
            "spacy": self._calculate_metrics_for_types(matches, spacy_entities),
            "ensemble": self._calculate_metrics_for_types(matches, ensemble_entities)
        }
    
        # Add summary stats
        metrics_by_detector["summary"] = {
            "total_entities": len(matches.get("true_positives", [])) + 
                             len(matches.get("false_negatives", [])),
            "total_detections": len(matches.get("true_positives", [])) + 
                               len(matches.get("false_positives", [])),
            "detection_rate": self._calculate_detection_rate(matches)
        }
    
        return metrics_by_detector
    
    def _calculate_metrics_for_types(self, matches: Dict, entity_types: Set[str]) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for specific entity types."""
        metrics = {}
        
        for entity_type in entity_types:
            # Filter matches by entity type
            type_matches = {
                "true_positives": {e for e in matches.get("true_positives", set()) 
                                 if e.entity_type == entity_type},
                "false_positives": {e for e in matches.get("false_positives", set())
                                 if e.entity_type == entity_type},
                "false_negatives": {e for e in matches.get("false_negatives", set())
                                 if e.entity_type == entity_type}
            }
            
            # Calculate basic metrics
            tp = len(type_matches["true_positives"])
            fp = len(type_matches["false_positives"])
            fn = len(type_matches["false_negatives"])
            
            # Calculate precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            metrics[entity_type] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "confidence_stats": self._get_confidence_stats(type_matches["true_positives"]),
                "validation_stats": self._get_validation_stats(entity_type)
            }
        
        return metrics
    
    def _get_confidence_stats(self, entities: Set[Entity]) -> Dict[str, float]:
        """Calculate confidence score statistics for a set of entities."""
        if not entities:
            return {"min": 0, "max": 0, "avg": 0}
            
        confidences = [getattr(e, "confidence", 1.0) for e in entities]
        return {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences)
        }
    
    def _get_validation_stats(self, entity_type: str) -> Dict:
        """Get validation statistics for an entity type."""
        validation_rules = self.validation_params.get(entity_type, {})
        routing_info = self._get_routing_info(entity_type)
        
        return {
            "validation_rules": validation_rules,
            "detector": routing_info["detector"],
            "confidence_threshold": routing_info["threshold"]
        }
    
    def _calculate_detection_rate(self, matches: Dict) -> float:
        """Calculate overall detection rate."""
        total = len(matches.get("true_positives", [])) + len(matches.get("false_negatives", []))
        if total == 0:
            return 0
        return len(matches.get("true_positives", [])) / total
    
###############################################################################
# SECTION 7: RESULT AGGREGATION AND REPORTING
###############################################################################
    def aggregate_results(self, metrics_list: List[Dict]) -> Dict:
        """
        Aggregate metrics across multiple test runs.
        """
        aggregated = {
            "overall": {
                "precision": [],
                "recall": [],
                "f1_score": [],
                "total_entities": 0,
                "total_detections": 0
            },
            "by_type": {},
            "by_detector": {
                "presidio": {"count": 0, "success_rate": 0},
                "spacy": {"count": 0, "success_rate": 0},
                "ensemble": {"count": 0, "success_rate": 0}
            }
        }
    
        # Aggregate metrics across runs
        for run_metrics in metrics_list:
            self._update_overall_metrics(aggregated["overall"], run_metrics)
            self._update_type_metrics(aggregated["by_type"], run_metrics)
            self._update_detector_metrics(aggregated["by_detector"], run_metrics)
    
        # Calculate averages
        self._calculate_final_averages(aggregated)
        
        return aggregated
    
    def _update_overall_metrics(self, overall: Dict, run_metrics: Dict) -> None:
        """Update overall metric aggregations with new run."""
        overall["precision"].append(run_metrics["metrics"]["precision"])
        overall["recall"].append(run_metrics["metrics"]["recall"])
        overall["f1_score"].append(run_metrics["metrics"]["f1_score"])
        overall["total_entities"] += len(run_metrics["ground_truth"]["entities"])
        overall["total_detections"] += len(run_metrics["detected"]["entities"])
    
    def _update_type_metrics(self, type_metrics: Dict, run_metrics: Dict) -> None:
        """Update per-type metric aggregations."""
        entity_type = run_metrics.get("detector_info", {}).get("entity_type")
        if not entity_type:
            return
    
        if entity_type not in type_metrics:
            type_metrics[entity_type] = {
                "precision": [],
                "recall": [],
                "f1_score": [],
                "false_positives": 0,
                "false_negatives": 0
            }
    
        metrics = type_metrics[entity_type]
        metrics["precision"].append(run_metrics["metrics"]["precision"])
        metrics["recall"].append(run_metrics["metrics"]["recall"])
        metrics["f1_score"].append(run_metrics["metrics"]["f1_score"])
        metrics["false_positives"] += run_metrics["metrics"]["false_positives"]
        metrics["false_negatives"] += run_metrics["metrics"]["false_negatives"]
    
    def _update_detector_metrics(self, detector_metrics: Dict, run_metrics: Dict) -> None:
        """Update detector-specific metrics."""
        detector = run_metrics.get("detector_info", {}).get("detector_type")
        if not detector:
            return
    
        detector_metrics[detector]["count"] += 1
        success = run_metrics["metrics"]["f1_score"] > 0.7  # Configurable threshold
        detector_metrics[detector]["success_rate"] += 1 if success else 0
    
    def _calculate_final_averages(self, aggregated: Dict) -> None:
        """Calculate final averages for aggregated metrics."""
        # Overall averages
        for metric in ["precision", "recall", "f1_score"]:
            values = aggregated["overall"][metric]
            aggregated["overall"][metric] = sum(values) / len(values) if values else 0
    
        # Type-based averages
        for type_metrics in aggregated["by_type"].values():
            for metric in ["precision", "recall", "f1_score"]:
                values = type_metrics[metric]
                type_metrics[metric] = sum(values) / len(values) if values else 0
    
        # Detector success rates
        for detector_metrics in aggregated["by_detector"].values():
            if detector_metrics["count"] > 0:
                detector_metrics["success_rate"] /= detector_metrics["count"]
    
    def generate_report(self, aggregated_results: Dict) -> str:
        """Generate a formatted report of aggregated results."""
        report = ["=== Entity Detection Analysis Report ===\n"]
        
        # Overall Performance
        report.append("Overall Performance:")
        report.append(f"Precision: {aggregated_results['overall']['precision']:.2%}")
        report.append(f"Recall: {aggregated_results['overall']['recall']:.2%}")
        report.append(f"F1 Score: {aggregated_results['overall']['f1_score']:.2%}")
        report.append(f"Total Entities: {aggregated_results['overall']['total_entities']}")
        report.append(f"Total Detections: {aggregated_results['overall']['total_detections']}\n")
        
        # Performance by Entity Type
        report.append("Performance by Entity Type:")
        for entity_type, metrics in aggregated_results["by_type"].items():
            report.append(f"\n{entity_type}:")
            report.append(f"  Precision: {metrics['precision']:.2%}")
            report.append(f"  Recall: {metrics['recall']:.2%}")
            report.append(f"  F1 Score: {metrics['f1_score']:.2%}")
            report.append(f"  False Positives: {metrics['false_positives']}")
            report.append(f"  False Negatives: {metrics['false_negatives']}")
        
        # Detector Performance
        report.append("\nDetector Performance:")
        for detector, metrics in aggregated_results["by_detector"].items():
            report.append(f"\n{detector.title()}:")
            report.append(f"  Success Rate: {metrics['success_rate']:.2%}")
            report.append(f"  Total Runs: {metrics['count']}")
        
        return "\n".join(report)    
    
    def _get_confidence_stats(self, entities: Set[Entity]) -> Dict[str, float]:
        """Calculate confidence score statistics for entities."""
        if not entities:
            return {"min": 0, "max": 0, "avg": 0}
        
        confidences = [getattr(e, "confidence", 1.0) for e in entities]
        return {
            "min": min(confidences),
            "max": max(confidences),
            "avg": sum(confidences) / len(confidences)
        }

    def _calculate_detection_rate(self, matches: Dict) -> float:
        """Calculate overall detection rate."""
        total = len(matches.get("true_positives", [])) + len(matches.get("false_negatives", []))
        return len(matches.get("true_positives", [])) / total if total > 0 else 0

    ###########################################################################
    # SECTION 8: RESULT AGGREGATION AND REPORTING
    ###########################################################################
    
    def aggregate_results(self, metrics_list: List[Dict]) -> Dict:
        """Aggregate metrics across multiple test runs."""
        aggregated = {
            "overall": {
                "precision": [],
                "recall": [],
                "f1_score": [],
                "total_entities": 0,
                "total_detections": 0
            },
            "by_type": {},
            "by_detector": {
                "presidio": {"count": 0, "success_rate": 0},
                "spacy": {"count": 0, "success_rate": 0},
                "ensemble": {"count": 0, "success_rate": 0}
            }
        }

        for run_metrics in metrics_list:
            self._update_overall_metrics(aggregated["overall"], run_metrics)
            self._update_type_metrics(aggregated["by_type"], run_metrics)
            self._update_detector_metrics(aggregated["by_detector"], run_metrics)

        self._calculate_final_averages(aggregated)
        return aggregated

    def generate_report(self, aggregated_results: Dict) -> str:
        """Generate formatted report of aggregated results."""
        report_lines = ["=== Entity Detection Analysis Report ===\n"]
        
        # Overall Performance
        report_lines.extend([
            "Overall Performance:",
            f"Precision: {aggregated_results['overall']['precision']:.2%}",
            f"Recall: {aggregated_results['overall']['recall']:.2%}",
            f"F1 Score: {aggregated_results['overall']['f1_score']:.2%}",
            f"Total Entities: {aggregated_results['overall']['total_entities']}",
            f"Total Detections: {aggregated_results['overall']['total_detections']}\n"
        ])
        
        # Performance by Entity Type
        report_lines.append("Performance by Entity Type:")
        for entity_type, metrics in aggregated_results["by_type"].items():
            report_lines.extend([
                f"\n{entity_type}:",
                f"  Precision: {metrics['precision']:.2%}",
                f"  Recall: {metrics['recall']:.2%}",
                f"  F1 Score: {metrics['f1_score']:.2%}",
                f"  False Positives: {metrics['false_positives']}",
                f"  False Negatives: {metrics['false_negatives']}"
            ])
        
        # Detector Performance
        report_lines.append("\nDetector Performance:")
        for detector, metrics in aggregated_results["by_detector"].items():
            report_lines.extend([
                f"\n{detector.title()}:",
                f"  Success Rate: {metrics['success_rate']:.2%}",
                f"  Total Runs: {metrics['count']}"
            ])
        
        return "\n".join(report_lines)