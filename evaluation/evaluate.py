# evaluation/evaluate.py

###############################################################################
# IMPORTS AND CONFIG
###############################################################################
from pathlib import Path
import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Set
import yaml
from tabulate import tabulate
from .test_runner import TestRunner
from redactor.detectors.ensemble_coordinator import EnsembleCoordinator
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
#from .metrics.entity_metrics import EntityMetrics
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
            log_level="DEBUG", 
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
    # SECTION 2: ANALYSIS & DETECTION 
    ###############################################################################
    
    def analyze_detection(self, test_id: str, target_entities: List[str] = None) -> Dict:
        """Analyze entity detection for a single test case"""
        try:
            text = self._load_test_document(test_id)
            all_detected = self.ensemble.detect_entities(text)
            
            # Convert to Entity objects
            detected = [
                Entity(
                    text=entity.text,
                    entity_type=entity.entity_type,
                    start_char=getattr(entity, 'start_char', 0),
                    end_char=getattr(entity, 'start_char', 0) + len(entity.text),
                    confidence=entity.confidence,
                    sensitivity=SensitivityLevel.LOW
                )
                for entity in all_detected
            ]
            
            # Filter if target entities specified
            if target_entities:
                detected = [e for e in detected if e.entity_type in target_entities]
            
            ground_truth = self._load_ground_truth(test_id)
            if target_entities:
                ground_truth = [gt for gt in ground_truth if gt.entity_type in target_entities]
                
            ground_truth_entities = [
                Entity(
                    text=gt.text,
                    entity_type=gt.entity_type,
                    start_char=0,
                    end_char=len(gt.text),
                    confidence=1.0,
                    sensitivity=SensitivityLevel.LOW
                )
                for gt in ground_truth
            ]
            
            return {
                "unmapped_entities": self._find_unmapped_entities(detected),
                "mapping_conflicts": self._find_mapping_conflicts(detected),
                "missed_entities": self._find_missed_entities(ground_truth_entities, detected),
                "detection_analysis": {
                    "entities": detected,
                    "statistics": self._analyze_detections(ground_truth_entities, detected)
                },
                "validation": self.entity_metrics.validate_targeted_entities(detected, target_entities) if target_entities else {}
            }
            
        except Exception as e:
            self.logger.error(f"Error in analyze_detection: {str(e)}")
            raise
    
        
    def _load_test_document(self, test_id: str) -> str:
        """Load document content from PDF"""
        pdf_path = self.test_suite_dir / "originals" / f"{test_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Test document not found: {pdf_path}")
        
        import fitz
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text() + "\n"
        return text

    def _load_ground_truth(self, test_id: str) -> List[DetectionResult]:
        """Load ground truth annotations"""
        annotation_path = self.test_suite_dir / "annotations" / f"{test_id}.yaml"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
        with open(annotation_path) as f:
            annotations = yaml.safe_load(f)
            
        return [
            DetectionResult(
                text=ann["text"],
                entity_type=ann["type"],
                confidence=1.0,
                detector="ground_truth"
            )
            for ann in annotations.get("entities", [])
        ]

    def _find_unmapped_entities(self, detected: List[DetectionResult]) -> List[str]:
        """Find entities without proper type mappings"""
        routing_config = self.config_loader.get_config("entity_routing")
        presidio_entities = set(routing_config["routing"]["presidio_primary"]["entities"])
        spacy_entities = set(routing_config["routing"]["spacy_primary"]["entities"]) 
        ensemble_entities = set(routing_config["routing"]["ensemble_required"]["entities"])
        all_known_entities = presidio_entities | spacy_entities | ensemble_entities
        
        unmapped = set()
        for result in detected:
            if result.entity_type not in all_known_entities:
                unmapped.add(result.entity_type)
        
        return list(unmapped)

    def _find_mapping_conflicts(self, detected: List[DetectionResult]) -> List[Dict]:
        """Find cases where same text maps to different entity types"""
        conflicts = []
        text_mappings = {}
        
        for result in detected:
            if result.text in text_mappings:
                if result.entity_type != text_mappings[result.text]:
                    conflicts.append({
                        "text": result.text,
                        "mappings": [
                            text_mappings[result.text],
                            result.entity_type
                        ]
                    })
            else:
                text_mappings[result.text] = result.entity_type
                
        return conflicts

    def _find_missed_entities(self, truth: List[DetectionResult], 
                            detected: List[DetectionResult]) -> List[DetectionResult]:
        """Find ground truth entities missed by detection"""
        def normalize_text(text: str) -> str:
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            edu_keywords = ['university', 'college', 'institute', 'school']
            matching_lines = [line for line in lines 
                             if any(keyword.lower() in line.lower() 
                             for keyword in edu_keywords)]
            return max(matching_lines, key=len) if matching_lines else text.strip()
        
        detected_texts = {normalize_text(r.text) for r in detected}
        missed = []
        
        for t in truth:
            truth_text = normalize_text(t.text)
            if not any(truth_text in d_text or d_text in truth_text 
                      for d_text in detected_texts):
                missed.append(t)
                
        return missed

    def _analyze_detections(self, truth: List[Entity], 
                          detected: List[Entity]) -> Dict:
        """Analyze detection performance by entity type"""
        analysis = {
            "by_type": {},
            "by_detector": {},
            "confidence_distribution": {}
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
        
        # Analyze by confidence
        for d in detected:
            # Use hardcoded detectors or add a way to track detector source
            detector = "ensemble"  # Default detector
            if detector not in analysis["by_detector"]:
                analysis["by_detector"][detector] = 0
            analysis["by_detector"][detector] += 1
        
        # Analyze confidence distribution
        confidence_ranges = [(0, 0.3), (0.3, 0.6), (0.6, 0.8), (0.8, 1.0)]
        for d in detected:
            for low, high in confidence_ranges:
                if low <= d.confidence < high:
                    range_key = f"{low:.1f}-{high:.1f}"
                    if range_key not in analysis["confidence_distribution"]:
                        analysis["confidence_distribution"][range_key] = 0
                    analysis["confidence_distribution"][range_key] += 1
        
        return analysis


    ###############################################################################
    # SECTION 3: OUTPUT & DISPLAY
    ###############################################################################  
    def display_analysis(self, analysis: Dict, target_entities: List[str] = None):
        if "entities" not in analysis:
            print("No entities detected.")
            return
    
        print("\nDetected Entities:")
        for entity in analysis["entities"]:
            validation = analysis["validation"].get(entity.text, {})
            print(f"- {entity.text}")
            print(f"  Confidence: {entity.confidence:.2f}")
            print(f"  Detector: {validation.get('detector', 'unknown')}")
            if validation.get('validation_rules'):
                print(f"  Rules: {len(validation['validation_rules'])} applied")
    
        stats = analysis.get("statistics", {})
        if stats:
            print(f"\nTotal detected: {len(analysis['entities'])}")
    
        
    def display_detailed_analysis(self, detailed_metrics: Dict, target_entities: List[str] = None):
        """Display detailed entity-specific analysis"""
        print("\nDetailed Entity Analysis")
        print("======================")
        
        # Display metrics for targeted entities only
        if target_entities:
            for entity_type in target_entities:
                if entity_type in detailed_metrics.get("metrics", {}):
                    print(f"\nMetrics for {entity_type}:")
                    print(f"Precision: {detailed_metrics['metrics'].get('precision', 0):.2%}")
                    print(f"Recall: {detailed_metrics['metrics'].get('recall', 0):.2%}")
                    print(f"F1 Score: {detailed_metrics['metrics'].get('f1_score', 0):.2%}")
        else:
            print("\nPerformance Metrics:")
            print(f"Precision: {detailed_metrics['metrics'].get('precision', 0):.2%}")
            print(f"Recall: {detailed_metrics['metrics'].get('recall', 0):.2%}")
            print(f"F1 Score: {detailed_metrics['metrics'].get('f1_score', 0):.2%}")
        
        # Show detector info for relevant entities
        detector_info = detailed_metrics.get("detector_info", {})
        if target_entities and detector_info.get("entity_type") in target_entities:
            print(f"\nDetector: {detector_info.get('detector_type')}")
            print(f"Confidence Threshold: {detector_info.get('confidence_threshold')}")
        
        # Show entity counts
        print("\nEntity Counts:")
        print(f"Ground Truth: {len(detailed_metrics.get('ground_truth', {}).get('entities', []))}")
        print(f"Detected: {len(detailed_metrics.get('detected', {}).get('entities', []))}")
        
        # Display errors for target entities only
        errors = detailed_metrics.get("errors", {})
        if errors.get("missed_entities"):
            missed = [e for e in errors["missed_entities"] 
                     if not target_entities or e.get("entity_type") in target_entities]
            if missed:
                print("\nMissed Entities:")
                for entity in missed:
                    print(f"- {entity['text']} ({entity.get('validation', {}).get('rules_applied', [])})")
                    
        if errors.get("false_positives"):
            false_pos = [e for e in errors["false_positives"] 
                        if not target_entities or e.get("entity_type") in target_entities]
            if false_pos:
                print("\nFalse Positives:")
                for entity in false_pos:
                    print(f"- {entity['text']} (Confidence: {entity.get('confidence', 0):.2f})")
                    if entity.get('likely_causes'):
                        print(f"  Causes: {', '.join(entity['likely_causes'])}")

    ###############################################################################
    # SECTION 4: UTILITY FUNCTIONS
    ###############################################################################


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
    parser.add_argument("stage", choices=["detect", "tune"])
    parser.add_argument("--test", required=True)
    parser.add_argument("--entity", nargs='+', choices=VALID_ENTITIES)
    parser.add_argument("--format", choices=["text", "json"], default="text",
                       help="Output format for metrics")
    
    args = parser.parse_args()
    
    if args.stage == "detect":
        try:
            test_case = TestCase(
                test_id=args.test,
                original_path=Path(f"data/test_suite/originals/{args.test}.pdf"),
                annotation_path=Path(f"data/test_suite/annotations/{args.test}.yaml")
            )
            
            evaluator = StageOneEvaluator()
            metrics = EntityMetrics(config_loader=evaluator.config_loader)
            
            # Run detection with target entities
            detection_results = evaluator.analyze_detection(test_case.test_id, args.entity)
            ground_truth = [
                Entity(
                    text=gt.text,
                    entity_type=gt.entity_type,
                    start_char=0,
                    end_char=len(gt.text),
                    confidence=1.0,
                    sensitivity=SensitivityLevel.LOW
                ) 
                for gt in evaluator._load_ground_truth(test_case.test_id)
            ]
            
            if args.entity:
                detailed_metrics = metrics.analyze_entity_type_metrics(
                    ground_truth=ground_truth,
                    detected=detection_results["detection_analysis"]["entities"],
                    target_entities=args.entity  
                )
                evaluator.display_detailed_analysis(detailed_metrics, args.entity)
            else:
                evaluator.display_analysis(detection_results)
                
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
            
    elif args.stage == "tune":
        print("Parameter tuning not yet implemented")

if __name__ == "__main__":
    main()
