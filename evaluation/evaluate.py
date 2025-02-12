# evaluation/evaluate.py

###############################################################################
# IMPORTS AND CONFIG
###############################################################################
from pathlib import Path
import sys
import argparse
import logging  # Add this line
from dataclasses import dataclass
from typing import List, Dict, Set
import yaml
from tabulate import tabulate
from datetime import datetime  # Add this too since we use it in batch processing
from .test_runner import TestRunner
from redactor.detectors.ensemble_coordinator import EnsembleCoordinator
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from evaluation.models import Entity, SensitivityLevel
from evaluation.metrics.entity_metrics import EntityMetrics

# Constants
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent
VALID_ENTITIES = [
    "EDUCATIONAL_INSTITUTION", "PERSON", "ADDRESS", "EMAIL_ADDRESS", 
    "PHONE_NUMBER", "INTERNET_REFERENCE", "LOCATION", "GPA",
    "DATE_TIME", "PROTECTED_CLASS", "PHI"
]

###############################################################################
# DATA MODELS
###############################################################################
@dataclass
class DetectionResult:
    text: str
    entity_type: str
    confidence: float
    detector: str
    mapped_type: str = None

@dataclass
class TestCase:
    test_id: str
    original_path: Path
    annotation_path: Path

###############################################################################
# UTILITY FUNCTIONS
###############################################################################
def print_metrics(metrics: Dict, indent: str = ""):
    for key, value in metrics.items():
        if isinstance(value, dict):
            print(f"{indent}{key}:")
            print_metrics(value, indent + "  ")
        else:
            print(f"{indent}{key}: {value}")

def validate_project_root():
    required_dirs = ['evaluation', 'redactor', 'data']
    missing_dirs = [d for d in required_dirs if not (PROJECT_ROOT / d).is_dir()]
    if missing_dirs:
        raise RuntimeError(
            f"Script must be run from project root. Missing directories: {missing_dirs}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Expected project root: {PROJECT_ROOT}\n"
            f"Run script using: python -m evaluation.evaluate [args]"
        )

###############################################################################
# EVALUATOR CLASSES 
###############################################################################
class StageOneEvaluator:
    """First stage: Entity detection and mapping analysis"""

    ###############################################################################
    # SECTION 1: INITIALIZATION & CONFIG
    ###############################################################################
    def __init__(self):
        self.test_suite_dir = Path("data/test_suite")
        if not self.test_suite_dir.exists():
            raise FileNotFoundError(
                f"Test suite directory not found: {self.test_suite_dir}\n"
                "Make sure you're running from the project root"
            )
        
        self.logger = RedactionLogger(
            name="evaluation",
            log_level="INFO", 
            log_dir="logs",
            console_output=True
        )
        
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()
        
        self.ensemble = EnsembleCoordinator(
            config_loader=self.config_loader,
            logger=self.logger
        )
        
        self.entity_metrics = EntityMetrics(config_loader=self.config_loader) 

    ###############################################################################
    # SECTION 2: ANALYSIS & DETECTION 
    ###############################################################################
        
    def analyze_detection(self, test_id: str, target_entities: List[str] = None) -> Dict:
        """Analyze entity detection for a single test case"""
        try:
            text = self._load_test_document(test_id)
            detected_entities = self.ensemble.detect_entities(text)
            
            # Convert while preserving metadata
            detected = []
            for entity in detected_entities:
                detected.append(Entity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    start_char=getattr(entity, 'start_char', 0),
                    end_char=getattr(entity, 'end_char', getattr(entity, 'start_char', 0) + len(entity.text)),
                    confidence=getattr(entity, 'confidence', 0.0),
                    sensitivity=getattr(entity, 'sensitivity', SensitivityLevel.LOW),
                    detector_source=getattr(entity, 'detector_source', 'unknown'),
                    validation_rules=getattr(entity, 'validation_rules', [])
                ))
            
            # Filter if target entities specified
            if target_entities:
                detected = [e for e in detected if e.entity_type in target_entities]
            
            # Load and convert ground truth
            ground_truth = self._load_ground_truth(test_id)
            ground_truth_entities = [
                Entity(
                    text=gt.text,
                    entity_type=gt.entity_type,
                    start_char=getattr(gt, 'start_char', 0),
                    end_char=getattr(gt, 'end_char', len(gt.text)),
                    confidence=1.0,  # Ground truth always has full confidence
                    sensitivity=SensitivityLevel.LOW,
                    detector_source='ground_truth',
                    validation_rules=[]
                )
                for gt in ground_truth
            ]
            
            if target_entities:
                ground_truth_entities = [gt for gt in ground_truth_entities 
                                       if gt.entity_type in target_entities]
            
            # Generate comprehensive analysis
            analysis_results = {
                "unmapped_entities": self._find_unmapped_entities(detected),
                "mapping_conflicts": self._find_mapping_conflicts(detected),
                "missed_entities": self._find_missed_entities(ground_truth_entities, detected),
                "detection_analysis": {
                    "entities": detected,
                    "statistics": self._analyze_detections(ground_truth_entities, detected),
                    "detector_performance": self._analyze_detector_performance(detected)
                }
            }
            
            # Add validation results if target entities specified
            if target_entities:
                analysis_results["validation"] = self.entity_metrics.validate_targeted_entities(
                    detected, target_entities
                )
                
            return analysis_results
                
        except Exception as e:
            self.logger.error(f"Error in analyze_detection: {str(e)}")
            raise
    
    def _analyze_detector_performance(self, detected: List[Entity]) -> Dict:
        """Analyze performance metrics by detector source"""
        detector_stats = {}
        
        for entity in detected:
            detector = getattr(entity, 'detector_source', 'unknown')
            if detector not in detector_stats:
                detector_stats[detector] = {
                    "total_detections": 0,
                    "confidence_sum": 0,
                    "by_entity_type": {}
                }
                
            stats = detector_stats[detector]
            stats["total_detections"] += 1
            stats["confidence_sum"] += entity.confidence
            
            # Track by entity type
            if entity.entity_type not in stats["by_entity_type"]:
                stats["by_entity_type"][entity.entity_type] = 0
            stats["by_entity_type"][entity.entity_type] += 1
        
        # Calculate averages
        for detector in detector_stats:
            total = detector_stats[detector]["total_detections"]
            if total > 0:
                detector_stats[detector]["average_confidence"] = \
                    detector_stats[detector]["confidence_sum"] / total
            del detector_stats[detector]["confidence_sum"]
        
        return detector_stats

    ###############################################################################
    # SECTION 2B: DOCUMENT LOADING AND PREPROCESSING
    ###############################################################################
    
    def _load_test_document(self, test_id: str) -> str:
        """Load document content from PDF"""
        pdf_path = self.test_suite_dir / "originals" / f"{test_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Test document not found: {pdf_path}")
        
        try:
            import fitz  # PyMuPDF
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            return text
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            raise
    
    def _load_ground_truth(self, test_id: str) -> List[Entity]:
        """Load ground truth annotations"""
        annotation_path = self.test_suite_dir / "annotations" / f"{test_id}.yaml"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
        try:
            with open(annotation_path) as f:
                annotations = yaml.safe_load(f)
                
            ground_truth = []
            for ann in annotations.get("entities", []):
                entity = Entity(
                    text=ann["text"],
                    entity_type=ann["type"],
                    start_char=ann.get("location", {}).get("start_char", 0),
                    end_char=ann.get("location", {}).get("end_char", len(ann["text"])),
                    confidence=1.0,  # Ground truth always has full confidence
                    sensitivity=SensitivityLevel.LOW,
                    detector_source='ground_truth'
                )
                ground_truth.append(entity)
                
            return ground_truth
        except Exception as e:
            self.logger.error(f"Error loading annotations from {annotation_path}: {str(e)}")
            raise

    ###############################################################################
    # SECTION 2C: ENTITY ANALYSIS HELPERS
    ###############################################################################
    
    def _find_unmapped_entities(self, detected: List[Entity]) -> List[str]:
        """Find entities without proper type mappings"""
        routing_config = self.config_loader.get_config("entity_routing")
        presidio_entities = set(routing_config["routing"]["presidio_primary"]["entities"])
        spacy_entities = set(routing_config["routing"]["spacy_primary"]["entities"]) 
        ensemble_entities = set(routing_config["routing"]["ensemble_required"]["entities"])
        all_known_entities = presidio_entities | spacy_entities | ensemble_entities
        
        unmapped = set()
        for entity in detected:
            if entity.entity_type not in all_known_entities:
                unmapped.add(entity.entity_type)
        
        return list(unmapped)
    
    def _find_mapping_conflicts(self, detected: List[Entity]) -> List[Dict]:
        """Find cases where same text maps to different entity types"""
        conflicts = []
        text_mappings = {}
        
        for entity in detected:
            if entity.text in text_mappings:
                if entity.entity_type != text_mappings[entity.text]:
                    conflicts.append({
                        "text": entity.text,
                        "mappings": [
                            text_mappings[entity.text],
                            entity.entity_type
                        ]
                    })
            else:
                text_mappings[entity.text] = entity.entity_type
                
        return conflicts
    
    def _find_missed_entities(self, truth: List[Entity], detected: List[Entity]) -> List[Entity]:
        """Find ground truth entities missed by detection"""
        def normalize_text(text: str) -> str:
            """Normalize text for comparison"""
            return ' '.join(text.lower().split())
        
        detected_texts = {normalize_text(d.text) for d in detected}
        missed = []
        
        for truth_entity in truth:
            truth_text = normalize_text(truth_entity.text)
            if not any(truth_text in d_text or d_text in truth_text 
                      for d_text in detected_texts):
                missed.append(truth_entity)
                
        return missed
    
    def _is_matching_entity(self, t: Entity, d: Entity) -> bool:
        """Check if detected entity matches ground truth entity"""
        def normalize_text(text: str) -> str:
            return ' '.join(text.lower().split())
        
        # Check type match
        if t.entity_type != d.entity_type:
            return False
        
        # Check text overlap
        t_text = normalize_text(t.text)
        d_text = normalize_text(d.text)
        
        return (t_text == d_text or 
                t_text in d_text or 
                d_text in t_text)
    
    ###############################################################################
    # SECTION 3: OUTPUT & DISPLAY
    ###############################################################################  
    def display_analysis(self, analysis: Dict, target_entities: List[str] = None):
        """Display core analysis results with clear entity detection information"""
        if "detection_analysis" not in analysis or not analysis["detection_analysis"].get("entities"):
            print("\nNo entities detected.")
            return
    
        # Overall Statistics
        total_entities = len(analysis["detection_analysis"]["entities"])
        print(f"\nDetected {total_entities} total entities")
        
        # Display detector performance
        if "detector_performance" in analysis["detection_analysis"]:
            print("\nDetector Performance:")
            print("--------------------")
            for detector, stats in analysis["detection_analysis"]["detector_performance"].items():
                print(f"\n{detector.upper()}:")
                print(f"  Total Detections: {stats['total_detections']}")
                print(f"  Average Confidence: {stats.get('average_confidence', 0):.2f}")
                
                if stats.get("by_entity_type"):
                    print("  Entity Types Found:")
                    for entity_type, count in stats["by_entity_type"].items():
                        print(f"    - {entity_type}: {count}")
    
        # Display detected entities with enhanced information
        print("\nDetected Entities:")
        print("-----------------")
        for entity in analysis["detection_analysis"]["entities"]:
            # Filter for target entities if specified
            if target_entities and entity.entity_type not in target_entities:
                continue
                
            print(f"\nEntity: {entity.text}")
            print(f"Type: {entity.entity_type}")
            print(f"Confidence: {entity.confidence:.2f}")
            print(f"Detector: {getattr(entity, 'detector_source', 'unknown')}")
            print(f"Position: chars {entity.start_char}-{entity.end_char}")
            
            # Show validation details if available
            validation = analysis.get("validation", {}).get(entity.text, {})
            if validation:
                print("Validation:")
                if validation.get("validation_rules"):
                    print("  Rules Applied:")
                    for rule in validation["validation_rules"]:
                        print(f"    - {rule}")
                        
        # Display any issues found
        if analysis.get("mapping_conflicts"):
            print("\nMapping Conflicts Found:")
            print("----------------------")
            for conflict in analysis["mapping_conflicts"]:
                print(f"- Text '{conflict['text']}' mapped to multiple types: {', '.join(conflict['mappings'])}")
        
        if analysis.get("missed_entities"):
            print("\nMissed Entities:")
            print("---------------")
            for entity in analysis["missed_entities"]:
                print(f"- {entity.text} ({entity.entity_type})")
    
    def display_detailed_analysis(self, detailed_metrics: Dict, target_entities: List[str] = None):
        """Display comprehensive entity-specific analysis"""
        print("\nDetailed Entity Analysis")
        print("======================")
        
        # Display metrics for targeted entities
        if target_entities:
            for entity_type in target_entities:
                if entity_type in detailed_metrics.get("metrics", {}):
                    print(f"\nMetrics for {entity_type}:")
                    print("------------------")
                    metrics = detailed_metrics["metrics"]
                    print(f"Precision: {metrics.get('precision', 0):.2%}")
                    print(f"Recall: {metrics.get('recall', 0):.2%}")
                    print(f"F1 Score: {metrics.get('f1_score', 0):.2%}")
                    
                    # Show detector-specific performance
                    detector_info = detailed_metrics.get("detector_info", {})
                    if detector_info:
                        print(f"\nDetector Information:")
                        print(f"  Primary Detector: {detector_info.get('detector_type', 'unknown')}")
                        print(f"  Confidence Threshold: {detector_info.get('confidence_threshold', 0):.2f}")
                        
                        if detector_info.get("validation_rules"):
                            print("  Validation Rules Applied:")
                            for rule, config in detector_info["validation_rules"].items():
                                print(f"    - {rule}: {config}")
        else:
            # Display overall metrics
            metrics = detailed_metrics.get("metrics", {})
            print("\nOverall Performance Metrics:")
            print("---------------------------")
            print(f"Precision: {metrics.get('precision', 0):.2%}")
            print(f"Recall: {metrics.get('recall', 0):.2%}")
            print(f"F1 Score: {metrics.get('f1_score', 0):.2%}")
        
        # Display entity counts and detection rates
        print("\nEntity Detection Summary:")
        print("-----------------------")
        print(f"Ground Truth: {len(detailed_metrics.get('ground_truth', {}).get('entities', []))}")
        print(f"Detected: {len(detailed_metrics.get('detected', {}).get('entities', []))}")
        
        # Display errors if present
        errors = detailed_metrics.get("errors", {})
        if errors:
            print("\nDetection Errors:")
            print("----------------")
            
            if errors.get("missed_entities"):
                print("\nMissed Entities:")
                for entity in errors["missed_entities"]:
                    if not target_entities or entity.get("entity_type") in target_entities:
                        print(f"- {entity['text']}")
                        if entity.get('possible_causes'):
                            print(f"  Possible causes: {', '.join(entity['possible_causes'])}")
                        
            if errors.get("false_positives"):
                print("\nFalse Positives:")
                for entity in errors["false_positives"]:
                    if not target_entities or entity.get("entity_type") in target_entities:
                        print(f"- {entity['text']} (Confidence: {entity.get('confidence', 0):.2f})")
                        if entity.get('likely_causes'):
                            print(f"  Likely causes: {', '.join(entity['likely_causes'])}")

    ###############################################################################
    # SECTION 4: VALIDATION AND STATISTICS
    ###############################################################################
    
    def _analyze_detections(self, truth: List[Entity], detected: List[Entity]) -> Dict:
        """Analyze detection performance by entity type and detection source"""
        analysis = {
            "by_type": {},
            "by_detector": {},
            "confidence_distribution": {},
            "validation_effectiveness": {},
            "configuration_coverage": self._analyze_config_coverage()
        }
        
        # Analyze by entity type
        truth_by_type = {}
        for t in truth:
            if t.entity_type not in truth_by_type:
                truth_by_type[t.entity_type] = {"total": 0, "detected": 0}
            truth_by_type[t.entity_type]["total"] += 1
        
        for d in detected:
            if d.entity_type in truth_by_type:
                truth_by_type[d.entity_type]["detected"] += 1
        
        # Calculate detection rates
        for entity_type, counts in truth_by_type.items():
            detection_rate = counts["detected"] / counts["total"] if counts["total"] > 0 else 0
            analysis["by_type"][entity_type] = {
                "total": counts["total"],
                "detected": counts["detected"],
                "detection_rate": detection_rate
            }
        
        # Analyze by detector
        detector_stats = {}
        for d in detected:
            detector = getattr(d, 'detector_source', 'unknown')
            if detector not in detector_stats:
                detector_stats[detector] = {
                    "total": 0,
                    "true_positives": 0,
                    "confidence_sum": 0,
                    "by_entity_type": {}
                }
                
            stats = detector_stats[detector]
            stats["total"] += 1
            stats["confidence_sum"] += d.confidence
            
            # Track by entity type
            if d.entity_type not in stats["by_entity_type"]:
                stats["by_entity_type"][d.entity_type] = 0
            stats["by_entity_type"][d.entity_type] += 1
            
            # Check if it's a true positive
            if any(self._is_matching_entity(t, d) for t in truth):
                stats["true_positives"] += 1
        
        # Calculate detector precision
        for detector, stats in detector_stats.items():
            stats["precision"] = stats["true_positives"] / stats["total"] if stats["total"] > 0 else 0
            stats["average_confidence"] = stats["confidence_sum"] / stats["total"] if stats["total"] > 0 else 0
            del stats["confidence_sum"]  # Clean up intermediate calculation
            
        analysis["by_detector"] = detector_stats
        
        # Analyze confidence distribution
        confidence_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for d in detected:
            for low, high in confidence_ranges:
                if low <= d.confidence < high:
                    range_key = f"{low:.1f}-{high:.1f}"
                    if range_key not in analysis["confidence_distribution"]:
                        analysis["confidence_distribution"][range_key] = 0
                    analysis["confidence_distribution"][range_key] += 1
        
        # Analyze validation rule effectiveness
        analysis["validation_effectiveness"] = self._analyze_validation_effectiveness(detected)
        
        return analysis
    
    def _analyze_config_coverage(self) -> Dict:
        """Analyze how well the configuration covers the detected entities"""
        routing_config = self.config_loader.get_config("entity_routing")
        validation_config = self.config_loader.get_config("validation_params")
        
        coverage = {
            "routing": {
                "total_entity_types": 0,
                "configured_types": 0,
                "routing_rules": {},
                "threshold_coverage": {}
            },
            "validation": {
                "total_rules": 0,
                "rules_by_type": {},
                "missing_validations": []
            }
        }
        
        # Analyze routing coverage
        for section in ["presidio_primary", "spacy_primary", "ensemble_required"]:
            if section in routing_config["routing"]:
                entities = routing_config["routing"][section]["entities"]
                coverage["routing"]["total_entity_types"] += len(entities)
                coverage["routing"]["routing_rules"][section] = len(entities)
                
                # Check threshold coverage
                thresholds = routing_config["routing"][section].get("thresholds", {})
                coverage["routing"]["threshold_coverage"][section] = {
                    "total": len(entities),
                    "with_thresholds": len(thresholds),
                    "missing_thresholds": [e for e in entities if e not in thresholds]
                }
        
        # Analyze validation coverage
        for entity_type in VALID_ENTITIES:
            rules = validation_config.get(entity_type, {})
            coverage["validation"]["rules_by_type"][entity_type] = len(rules)
            coverage["validation"]["total_rules"] += len(rules)
            
            if not rules:
                coverage["validation"]["missing_validations"].append(entity_type)
        
        return coverage
    
    def _analyze_validation_effectiveness(self, detected: List[Entity]) -> Dict:
        """Analyze how effective each validation rule is"""
        effectiveness = {}
        
        for entity in detected:
            entity_type = entity.entity_type
            if entity_type not in effectiveness:
                effectiveness[entity_type] = {
                    "total_validations": 0,
                    "rules_triggered": {},
                    "confidence_impact": {}
                }
                
            stats = effectiveness[entity_type]
            stats["total_validations"] += 1
            
            # Track which rules were applied
            validation_rules = getattr(entity, 'validation_rules', []) or []
            for rule in validation_rules:
                if rule not in stats["rules_triggered"]:
                    stats["rules_triggered"][rule] = 0
                stats["rules_triggered"][rule] += 1
                
            # Track confidence changes from validation, handling potential None values
            original_confidence = getattr(entity, 'original_confidence', None)
            if original_confidence is not None:  # Only calculate if we have both values
                confidence_change = entity.confidence - original_confidence
                
                if confidence_change != 0:
                    range_key = "increase" if confidence_change > 0 else "decrease"
                    if range_key not in stats["confidence_impact"]:
                        stats["confidence_impact"][range_key] = {
                            "count": 0,
                            "total_change": 0
                        }
                    stats["confidence_impact"][range_key]["count"] += 1
                    stats["confidence_impact"][range_key]["total_change"] += abs(confidence_change)
        
        # Calculate averages for confidence impacts, handling empty cases
        for entity_type in effectiveness:
            for impact_type in effectiveness[entity_type]["confidence_impact"]:
                impact = effectiveness[entity_type]["confidence_impact"][impact_type]
                if impact.get("count", 0) > 0:
                    impact["average_change"] = impact["total_change"] / impact["count"]
                    del impact["total_change"]  # Clean up intermediate calculation
        
        return effectiveness


    ###############################################################################
    # SECTION 5: TEST MANAGEMENT AND COMPARISON
    ###############################################################################
    
    def run_batch_evaluation(self, test_cases: List[str], target_entities: List[str] = None) -> Dict:
        """Run evaluation across multiple test cases and aggregate results"""
        batch_results = {
            "timestamp": datetime.now().isoformat(),
            "test_cases": {},
            "aggregate_metrics": {
                "by_entity": {},
                "by_detector": {},
                "validation_stats": {},
                "confidence_stats": {}
            },
            "configuration_analysis": self._analyze_config_coverage()
        }
        
        # Run each test case
        for test_id in test_cases:
            try:
                results = self.analyze_detection(test_id, target_entities)
                batch_results["test_cases"][test_id] = results
                self._update_aggregate_metrics(batch_results["aggregate_metrics"], results)
            except Exception as e:
                self.logger.error(f"Error processing test case {test_id}: {str(e)}")
                batch_results["test_cases"][test_id] = {"error": str(e)}
        
        # Calculate final aggregated metrics
        self._finalize_aggregate_metrics(batch_results["aggregate_metrics"])
        
        return batch_results
    
    def _update_aggregate_metrics(self, aggregate: Dict, results: Dict) -> None:
        """Update running aggregate metrics with results from a single test case"""
        detection_analysis = results.get("detection_analysis", {})
        
        # Update entity-type metrics
        for entity_type, stats in detection_analysis.get("by_type", {}).items():
            if entity_type not in aggregate["by_entity"]:
                aggregate["by_entity"][entity_type] = {
                    "total_detected": 0,
                    "total_truth": 0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0
                }
            
            entity_stats = aggregate["by_entity"][entity_type]
            entity_stats["total_detected"] += stats.get("detected", 0)
            entity_stats["total_truth"] += stats.get("total", 0)
        
        # Update detector metrics
        for detector, stats in detection_analysis.get("by_detector", {}).items():
            if detector not in aggregate["by_detector"]:
                aggregate["by_detector"][detector] = {
                    "total_detections": 0,
                    "true_positives": 0,
                    "confidence_sum": 0,
                    "by_entity_type": {}
                }
            
            detector_stats = aggregate["by_detector"][detector]
            detector_stats["total_detections"] += stats.get("total", 0)
            detector_stats["true_positives"] += stats.get("true_positives", 0)
            detector_stats["confidence_sum"] += stats.get("total", 0) * stats.get("average_confidence", 0)
            
            # Update entity type breakdown
            for entity_type, count in stats.get("by_entity_type", {}).items():
                if entity_type not in detector_stats["by_entity_type"]:
                    detector_stats["by_entity_type"][entity_type] = 0
                detector_stats["by_entity_type"][entity_type] += count
        
        # Update validation statistics
        validation_effectiveness = detection_analysis.get("validation_effectiveness", {})
        for entity_type, stats in validation_effectiveness.items():
            if entity_type not in aggregate["validation_stats"]:
                aggregate["validation_stats"][entity_type] = {
                    "total_validations": 0,
                    "rules_triggered": {},
                    "confidence_impacts": {"increase": 0, "decrease": 0}
                }
            
            val_stats = aggregate["validation_stats"][entity_type]
            val_stats["total_validations"] += stats.get("total_validations", 0)
            
            # Aggregate rule triggers
            for rule, count in stats.get("rules_triggered", {}).items():
                if rule not in val_stats["rules_triggered"]:
                    val_stats["rules_triggered"][rule] = 0
                val_stats["rules_triggered"][rule] += count
            
            # Aggregate confidence impacts
            for impact_type, impact_stats in stats.get("confidence_impact", {}).items():
                val_stats["confidence_impacts"][impact_type] += impact_stats.get("count", 0)
    
    def _finalize_aggregate_metrics(self, aggregate: Dict) -> None:
        """Calculate final metrics from aggregated data"""
        # Finalize entity metrics
        for entity_type, stats in aggregate["by_entity"].items():
            if stats["total_detected"] > 0:
                precision = stats["true_positives"] / stats["total_detected"]
                recall = stats["true_positives"] / stats["total_truth"] if stats["total_truth"] > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                stats.update({
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1
                })
        
        # Finalize detector metrics
        for detector, stats in aggregate["by_detector"].items():
            if stats["total_detections"] > 0:
                stats["precision"] = stats["true_positives"] / stats["total_detections"]
                stats["average_confidence"] = stats["confidence_sum"] / stats["total_detections"]
            del stats["confidence_sum"]  # Clean up intermediate calculation
        
        # Finalize validation stats
        for entity_type, stats in aggregate["validation_stats"].items():
            total = stats["total_validations"]
            if total > 0:
                # Calculate rule effectiveness
                stats["rule_effectiveness"] = {
                    rule: count / total 
                    for rule, count in stats["rules_triggered"].items()
                }
                # Calculate confidence impact rates
                stats["confidence_impact_rates"] = {
                    impact_type: count / total
                    for impact_type, count in stats["confidence_impacts"].items()
                }
    
    def compare_test_runs(self, run1_results: Dict, run2_results: Dict) -> Dict:
        """Compare metrics between two test runs"""
        comparison = {
            "entity_type_changes": {},
            "detector_changes": {},
            "validation_changes": {},
            "overall_changes": self._calculate_overall_changes(run1_results, run2_results)
        }
        
        # Compare entity type metrics
        for entity_type in set(run1_results["aggregate_metrics"]["by_entity"].keys()) | \
                           set(run2_results["aggregate_metrics"]["by_entity"].keys()):
            comparison["entity_type_changes"][entity_type] = self._compare_metrics(
                run1_results["aggregate_metrics"]["by_entity"].get(entity_type, {}),
                run2_results["aggregate_metrics"]["by_entity"].get(entity_type, {})
            )
        
        # Compare detector performance
        for detector in set(run1_results["aggregate_metrics"]["by_detector"].keys()) | \
                        set(run2_results["aggregate_metrics"]["by_detector"].keys()):
            comparison["detector_changes"][detector] = self._compare_metrics(
                run1_results["aggregate_metrics"]["by_detector"].get(detector, {}),
                run2_results["aggregate_metrics"]["by_detector"].get(detector, {})
            )
        
        # Compare validation effectiveness
        for entity_type in set(run1_results["aggregate_metrics"]["validation_stats"].keys()) | \
                           set(run2_results["aggregate_metrics"]["validation_stats"].keys()):
            comparison["validation_changes"][entity_type] = self._compare_validation_stats(
                run1_results["aggregate_metrics"]["validation_stats"].get(entity_type, {}),
                run2_results["aggregate_metrics"]["validation_stats"].get(entity_type, {})
            )
        
        return comparison
    
    def _compare_metrics(self, metrics1: Dict, metrics2: Dict) -> Dict:
        """Calculate changes between two sets of metrics"""
        changes = {}
        all_keys = set(metrics1.keys()) | set(metrics2.keys())
        
        for key in all_keys:
            if isinstance(metrics1.get(key), (int, float)) and isinstance(metrics2.get(key), (int, float)):
                val1 = metrics1.get(key, 0)
                val2 = metrics2.get(key, 0)
                changes[key] = {
                    "before": val1,
                    "after": val2,
                    "absolute_change": val2 - val1,
                    "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else float('inf')
                }
        
        return changes
    
    def _compare_validation_stats(self, stats1: Dict, stats2: Dict) -> Dict:
        """Compare validation statistics between runs"""
        comparison = {
            "total_validations": self._compare_metrics(
                {"total": stats1.get("total_validations", 0)},
                {"total": stats2.get("total_validations", 0)}
            )["total"],
            "rule_changes": {},
            "confidence_impact_changes": {}
        }
        
        # Compare rule effectiveness
        rules1 = stats1.get("rule_effectiveness", {})
        rules2 = stats2.get("rule_effectiveness", {})
        for rule in set(rules1.keys()) | set(rules2.keys()):
            comparison["rule_changes"][rule] = self._compare_metrics(
                {"effectiveness": rules1.get(rule, 0)},
                {"effectiveness": rules2.get(rule, 0)}
            )["effectiveness"]
        
        # Compare confidence impacts
        impacts1 = stats1.get("confidence_impact_rates", {})
        impacts2 = stats2.get("confidence_impact_rates", {})
        for impact_type in set(impacts1.keys()) | set(impacts2.keys()):
            comparison["confidence_impact_changes"][impact_type] = self._compare_metrics(
                {"rate": impacts1.get(impact_type, 0)},
                {"rate": impacts2.get(impact_type, 0)}
            )["rate"]
        
        return comparison

    ###############################################################################
    # SECTION 6: STATISTICAL ANALYSIS
    ###############################################################################
    
    def calculate_statistical_metrics(self, batch_results: Dict) -> Dict:
        """Calculate detailed statistical metrics across test runs"""
        stats = {
            "entity_statistics": self._calculate_entity_statistics(batch_results),
            "detector_statistics": self._calculate_detector_statistics(batch_results),
            "confidence_analysis": self._calculate_confidence_statistics(batch_results),
            "validation_statistics": self._calculate_validation_statistics(batch_results),
            "error_analysis": self._calculate_error_statistics(batch_results)
        }
        
        return stats
    
    def _calculate_entity_statistics(self, batch_results: Dict) -> Dict:
        """Calculate statistical metrics for entity detection"""
        entity_stats = {}
        
        for test_id, results in batch_results["test_cases"].items():
            detection_analysis = results.get("detection_analysis", {})
            
            for entity in detection_analysis.get("entities", []):
                entity_type = entity.entity_type
                if entity_type not in entity_stats:
                    entity_stats[entity_type] = {
                        "detection_counts": [],
                        "confidence_scores": [],
                        "validation_success_rate": 0,
                        "context_success_rate": 0,
                        "detection_rate_by_test": {}
                    }
                
                stats = entity_stats[entity_type]
                stats["confidence_scores"].append(entity.confidence)
                stats["detection_rate_by_test"][test_id] = detection_analysis.get("statistics", {}) \
                    .get("by_type", {}).get(entity_type, {}).get("detection_rate", 0)
        
        # Calculate final statistics
        for entity_type, stats in entity_stats.items():
            confidence_scores = stats["confidence_scores"]
            detection_rates = list(stats["detection_rate_by_test"].values())
            
            if confidence_scores:
                stats.update({
                    "confidence_statistics": {
                        "mean": sum(confidence_scores) / len(confidence_scores),
                        "median": sorted(confidence_scores)[len(confidence_scores) // 2],
                        "min": min(confidence_scores),
                        "max": max(confidence_scores),
                        "std_dev": self._calculate_std_dev(confidence_scores)
                    }
                })
            
            if detection_rates:
                stats.update({
                    "detection_rate_statistics": {
                        "mean": sum(detection_rates) / len(detection_rates),
                        "median": sorted(detection_rates)[len(detection_rates) // 2],
                        "min": min(detection_rates),
                        "max": max(detection_rates),
                        "std_dev": self._calculate_std_dev(detection_rates)
                    }
                })
            
            # Clean up intermediate data
            del stats["confidence_scores"]
            del stats["detection_rate_by_test"]
        
        return entity_stats
    
    def _calculate_detector_statistics(self, batch_results: Dict) -> Dict:
        """Calculate statistical metrics for each detector"""
        detector_stats = {}
        
        for results in batch_results["test_cases"].values():
            detection_analysis = results.get("detection_analysis", {})
            
            for detector, stats in detection_analysis.get("by_detector", {}).items():
                if detector not in detector_stats:
                    detector_stats[detector] = {
                        "precision_scores": [],
                        "confidence_scores": [],
                        "detection_counts": [],
                        "entity_type_coverage": {}
                    }
                
                det_stats = detector_stats[detector]
                det_stats["precision_scores"].append(stats.get("precision", 0))
                det_stats["detection_counts"].append(stats.get("total", 0))
                
                # Track entity type coverage
                for entity_type, count in stats.get("by_entity_type", {}).items():
                    if entity_type not in det_stats["entity_type_coverage"]:
                        det_stats["entity_type_coverage"][entity_type] = []
                    det_stats["entity_type_coverage"][entity_type].append(count)
        
        # Calculate final statistics
        for detector, stats in detector_stats.items():
            for metric in ["precision_scores", "detection_counts"]:
                values = stats[metric]
                if values:
                    stats[f"{metric}_statistics"] = {
                        "mean": sum(values) / len(values),
                        "median": sorted(values)[len(values) // 2],
                        "min": min(values),
                        "max": max(values),
                        "std_dev": self._calculate_std_dev(values)
                    }
            
            # Calculate entity type coverage statistics
            coverage_stats = {}
            for entity_type, counts in stats["entity_type_coverage"].items():
                coverage_stats[entity_type] = {
                    "mean": sum(counts) / len(counts),
                    "median": sorted(counts)[len(counts) // 2],
                    "min": min(counts),
                    "max": max(counts),
                    "std_dev": self._calculate_std_dev(counts)
                }
            stats["entity_type_coverage_statistics"] = coverage_stats
            
            # Clean up intermediate data
            del stats["precision_scores"]
            del stats["confidence_scores"]
            del stats["detection_counts"]
            del stats["entity_type_coverage"]
        
        return detector_stats
    
    def _calculate_confidence_statistics(self, batch_results: Dict) -> Dict:
        """Calculate detailed confidence score statistics"""
        confidence_stats = {
            "overall_distribution": {},
            "by_entity_type": {},
            "by_detector": {},
            "validation_impact": {}
        }
        
        # Analyze confidence scores across all results
        for results in batch_results["test_cases"].values():
            detection_analysis = results.get("detection_analysis", {})
            
            for entity in detection_analysis.get("entities", []):
                # Track overall distribution
                confidence_range = self._get_confidence_range(entity.confidence)
                if confidence_range not in confidence_stats["overall_distribution"]:
                    confidence_stats["overall_distribution"][confidence_range] = 0
                confidence_stats["overall_distribution"][confidence_range] += 1
                
                # Track by entity type
                if entity.entity_type not in confidence_stats["by_entity_type"]:
                    confidence_stats["by_entity_type"][entity.entity_type] = {
                        "scores": [],
                        "distribution": {}
                    }
                type_stats = confidence_stats["by_entity_type"][entity.entity_type]
                type_stats["scores"].append(entity.confidence)
                
                if confidence_range not in type_stats["distribution"]:
                    type_stats["distribution"][confidence_range] = 0
                type_stats["distribution"][confidence_range] += 1
                
                # Track by detector
                detector = getattr(entity, 'detector_source', 'unknown')
                if detector not in confidence_stats["by_detector"]:
                    confidence_stats["by_detector"][detector] = {
                        "scores": [],
                        "distribution": {}
                    }
                detector_stats = confidence_stats["by_detector"][detector]
                detector_stats["scores"].append(entity.confidence)
                
                if confidence_range not in detector_stats["distribution"]:
                    detector_stats["distribution"][confidence_range] = 0
                detector_stats["distribution"][confidence_range] += 1
        
        # Calculate final statistics
        for entity_type, stats in confidence_stats["by_entity_type"].items():
            scores = stats["scores"]
            if scores:
                stats["statistics"] = {
                    "mean": sum(scores) / len(scores),
                    "median": sorted(scores)[len(scores) // 2],
                    "min": min(scores),
                    "max": max(scores),
                    "std_dev": self._calculate_std_dev(scores)
                }
            del stats["scores"]
        
        for detector, stats in confidence_stats["by_detector"].items():
            scores = stats["scores"]
            if scores:
                stats["statistics"] = {
                    "mean": sum(scores) / len(scores),
                    "median": sorted(scores)[len(scores) // 2],
                    "min": min(scores),
                    "max": max(scores),
                    "std_dev": self._calculate_std_dev(scores)
                }
            del stats["scores"]
        
        return confidence_stats
    
    def _calculate_validation_statistics(self, batch_results: Dict) -> Dict:
        """Calculate statistical metrics for validation effectiveness"""
        validation_stats = {
            "rule_effectiveness": {},
            "confidence_impacts": {},
            "validation_coverage": {}
        }
        
        for results in batch_results["test_cases"].values():
            validation_effectiveness = results.get("detection_analysis", {}) \
                .get("validation_effectiveness", {})
            
            for entity_type, stats in validation_effectiveness.items():
                if entity_type not in validation_stats["rule_effectiveness"]:
                    validation_stats["rule_effectiveness"][entity_type] = {}
                
                # Track rule effectiveness
                for rule, count in stats.get("rules_triggered", {}).items():
                    if rule not in validation_stats["rule_effectiveness"][entity_type]:
                        validation_stats["rule_effectiveness"][entity_type][rule] = []
                    validation_stats["rule_effectiveness"][entity_type][rule].append(
                        count / stats["total_validations"] if stats["total_validations"] > 0 else 0
                    )
        
        # Calculate final statistics
        for entity_type, rules in validation_stats["rule_effectiveness"].items():
            for rule, rates in rules.items():
                if rates:
                    rules[rule] = {
                        "mean": sum(rates) / len(rates),
                        "median": sorted(rates)[len(rates) // 2],
                        "min": min(rates),
                        "max": max(rates),
                        "std_dev": self._calculate_std_dev(rates)
                    }
        
        return validation_stats
    
    def _calculate_error_statistics(self, batch_results: Dict) -> Dict:
        """Calculate statistics about detection errors"""
        error_stats = {
            "missed_entities": {},
            "false_positives": {},
            "error_patterns": {}
        }
        
        for results in batch_results["test_cases"].values():
            # Track missed entities
            for entity in results.get("missed_entities", []):
                entity_type = entity.entity_type
                if entity_type not in error_stats["missed_entities"]:
                    error_stats["missed_entities"][entity_type] = {
                        "count": 0,
                        "causes": {}
                    }
                
                error_stats["missed_entities"][entity_type]["count"] += 1
                
                # Track causes
                for cause in getattr(entity, 'possible_causes', []):
                    if cause not in error_stats["missed_entities"][entity_type]["causes"]:
                        error_stats["missed_entities"][entity_type]["causes"][cause] = 0
                    error_stats["missed_entities"][entity_type]["causes"][cause] += 1
            
            # Track false positives
            detection_analysis = results.get("detection_analysis", {})
            for entity in detection_analysis.get("entities", []):
                if not any(self._is_matching_entity(t, entity) 
                          for t in results.get("ground_truth", [])):
                    entity_type = entity.entity_type
                    if entity_type not in error_stats["false_positives"]:
                        error_stats["false_positives"][entity_type] = {
                            "count": 0,
                            "confidence_distribution": {},
                            "likely_causes": {}
                        }
                    
                    stats = error_stats["false_positives"][entity_type]
                    stats["count"] += 1
                    
                    # Track confidence distribution
                    confidence_range = self._get_confidence_range(entity.confidence)
                    if confidence_range not in stats["confidence_distribution"]:
                        stats["confidence_distribution"][confidence_range] = 0
                    stats["confidence_distribution"][confidence_range] += 1
                    
                    # Track causes
                    for cause in getattr(entity, 'likely_causes', []):
                        if cause not in stats["likely_causes"]:
                            stats["likely_causes"][cause] = 0
                        stats["likely_causes"][cause] += 1
        
        return error_stats
    
    def _calculate_std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation of a list of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5
    
    def _get_confidence_range(self, confidence: float) -> str:
        """Get the confidence range bucket for a confidence score"""
        ranges = [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for low, high in ranges:
            if low <= confidence < high:
                return f"{low:.1f}-{high:.1f}"
        return "1.0"


class StageTwoEvaluator:
   """Second stage: Parameter tuning and optimization"""
   def __init__(self):
       pass

###############################################################################
# COMMAND LINE INTERFACE
###############################################################################

def main():
    validate_project_root()
    
    parser = argparse.ArgumentParser(
        description="Redaction system evaluation tool",
        usage="python -m evaluation.evaluate [args]"
    )
    parser.add_argument("stage", choices=["detect", "tune", "batch", "compare"])
    parser.add_argument("--test", nargs='+', required=True,
                       help="Test ID(s) to evaluate. Multiple IDs for batch mode.")
    parser.add_argument("--entity", nargs='+', choices=VALID_ENTITIES,
                       help="Target specific entity types")
    parser.add_argument("--format", choices=["text", "json", "csv"], default="text",
                       help="Output format for metrics")
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--compare-with", help="Previous test run ID to compare against")
    parser.add_argument("--stats", action="store_true", 
                       help="Include detailed statistical analysis")
    parser.add_argument("--quiet", action="store_true",
                       help="Reduce logging output")
    
    args = parser.parse_args()

    # Configure logging - more restrictive by default
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Reduce noise from other loggers
    logging.getLogger('config_loader').setLevel(logging.WARNING)
    logging.getLogger('presidio-analyzer').setLevel(logging.WARNING)
    logging.getLogger('evaluation').setLevel(log_level)
    
    try:
        evaluator = StageOneEvaluator()
        
        if args.stage == "detect":
            # Single test case evaluation
            test_case = TestCase(
                test_id=args.test[0],
                original_path=Path(f"data/test_suite/originals/{args.test[0]}.pdf"),
                annotation_path=Path(f"data/test_suite/annotations/{args.test[0]}.yaml")
            )
            
            results = evaluator.analyze_detection(test_case.test_id, args.entity)
            
            if args.stats:
                stats = evaluator.calculate_statistical_metrics({"test_cases": {test_case.test_id: results}})
                results["statistics"] = stats
            
            if args.format == "text":
                evaluator.display_analysis(results, args.entity)
            else:
                output_data = results
                if args.output:
                    if args.format == "json":
                        with open(args.output, 'w') as f:
                            json.dump(output_data, f, indent=2)
                    elif args.format == "csv":
                        # Convert to CSV format
                        pass
                else:
                    print(json.dumps(output_data, indent=2))
                    
        elif args.stage == "batch":
            # Batch evaluation
            batch_results = evaluator.run_batch_evaluation(args.test, args.entity)
            
            if args.stats:
                stats = evaluator.calculate_statistical_metrics(batch_results)
                batch_results["statistics"] = stats
            
            if args.compare_with:
                # Load previous results
                prev_results_path = Path(f"data/test_suite/results/run_{args.compare_with}/summary.json")
                if prev_results_path.exists():
                    with open(prev_results_path) as f:
                        prev_results = json.load(f)
                    comparison = evaluator.compare_test_runs(prev_results, batch_results)
                    batch_results["comparison"] = comparison
            
            if args.output:
                with open(args.output, 'w') as f:
                    json.dump(batch_results, f, indent=2)
            else:
                if args.format == "text":
                    print("\nBatch Evaluation Results:")
                    print("========================")
                    for test_id, results in batch_results["test_cases"].items():
                        print(f"\nTest Case: {test_id}")
                        evaluator.display_analysis(results, args.entity)
                else:
                    print(json.dumps(batch_results, indent=2))
                    
        elif args.stage == "compare":
            if not args.compare_with:
                print("Error: --compare-with is required for compare mode")
                sys.exit(1)
                
            # Run new test
            results = evaluator.analyze_detection(args.test[0], args.entity)
            
            # Load previous results
            prev_results_path = Path(f"data/test_suite/results/run_{args.compare_with}/summary.json")
            if prev_results_path.exists():
                with open(prev_results_path) as f:
                    prev_results = json.load(f)
                comparison = evaluator.compare_test_runs(prev_results, {"test_cases": {args.test[0]: results}})
                
                if args.format == "text":
                    print("\nComparison Results:")
                    print("==================")
                    # Add structured comparison display
                    pass
                else:
                    if args.output:
                        with open(args.output, 'w') as f:
                            json.dump(comparison, f, indent=2)
                    else:
                        print(json.dumps(comparison, indent=2))
            else:
                print(f"Error: Previous results not found at {prev_results_path}")
                sys.exit(1)
                
        elif args.stage == "tune":
            print("Parameter tuning not yet implemented")
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()