# evaluation/evaluate.py
from pathlib import Path

# Get project root directory (2 levels up from this script)
PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

def validate_project_root():
    """Verify the script is being run from the correct location"""
    required_dirs = ['evaluation', 'redactor', 'data']
    missing_dirs = [d for d in required_dirs if not (PROJECT_ROOT / d).is_dir()]
    
    if missing_dirs:
        raise RuntimeError(
            f"Script must be run from project root. Missing directories: {missing_dirs}\n"
            f"Current working directory: {Path.cwd()}\n"
            f"Expected project root: {PROJECT_ROOT}\n"
            f"Run script using: python -m evaluation.evaluate [args]"
        )

import sys
import argparse
from dataclasses import dataclass
from typing import List, Dict, Set
from pathlib import Path
import yaml
from tabulate import tabulate

from redactor.detectors.ensemble_coordinator import EnsembleCoordinator
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger

@dataclass
class DetectionResult:
    """Results from detection system"""
    text: str
    entity_type: str
    confidence: float
    detector: str
    mapped_type: str = None

class StageOneEvaluator:
    """First stage: Entity detection and mapping analysis"""
    
    def __init__(self):
        self.test_suite_dir = Path("data/test_suite")
        if not self.test_suite_dir.exists():
            raise FileNotFoundError(
                f"Test suite directory not found: {self.test_suite_dir}\n"
                "Make sure you're running from the project root"
            )
        
        # Initialize components we need
        self.logger = RedactionLogger(
            name="evaluation",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )
        
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()
        
        # Initialize ensemble detection system
        self.ensemble = EnsembleCoordinator(
            config_loader=self.config_loader,
            logger=self.logger
        )

    def analyze_detection(self, test_id: str) -> Dict:
        """Analyze entity detection for a single test case"""
        # Get ground truth
        truth = self._load_ground_truth(test_id)
        
        # Get detection results using ensemble system
        text = self._load_test_document(test_id)
        detected_entities = self.ensemble.detect_entities(text)
        
        # Convert ensemble results to our format
        detected = [
            DetectionResult(
                text=entity.text,
                entity_type=entity.entity_type,
                confidence=entity.confidence,
                detector=entity.source,
                mapped_type=entity.metadata.get('mapped_type') if entity.metadata else None
            )
            for entity in detected_entities
        ]
        
        # Analyze results
        analysis = {
            "unmapped_entities": self._find_unmapped_entities(detected),
            "mapping_conflicts": self._find_mapping_conflicts(detected),
            "missed_entities": self._find_missed_entities(truth, detected),
            "detection_analysis": self._analyze_detections(truth, detected)
        }
        
        return analysis

    def _load_test_document(self, test_id: str) -> str:
        """Load document content from PDF"""
        pdf_path = self.test_suite_dir / "originals" / f"{test_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Test document not found: {pdf_path}")
        
        import fitz  # PyMuPDF
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
                confidence=1.0,  # Ground truth has full confidence
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
        detected_texts = {r.text for r in detected}
        return [t for t in truth if t.text not in detected_texts]

    def _analyze_detections(self, truth: List[DetectionResult], 
                          detected: List[DetectionResult]) -> Dict:
        """Analyze detection performance by entity type and detector"""
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
        
        # Analyze by detector source
        for d in detected:
            if d.detector not in analysis["by_detector"]:
                analysis["by_detector"][d.detector] = 0
            analysis["by_detector"][d.detector] += 1
        
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

    def display_analysis(self, analysis: Dict):
        """Display analysis results in readable format"""
        print("\nEntity Detection Analysis")
        print("========================")
        
        # Show unmapped entities
        if analysis["unmapped_entities"]:
            print("\nUnmapped Entity Types:")
            for entity_type in analysis["unmapped_entities"]:
                print(f"- {entity_type}")
        
        # Show mapping conflicts
        if analysis["mapping_conflicts"]:
            print("\nMapping Conflicts:")
            for conflict in analysis["mapping_conflicts"]:
                print(f"Text: {conflict['text']}")
                print(f"Conflicting mappings: {', '.join(conflict['mappings'])}")
        
        # Show detection analysis by type
        print("\nDetection by Entity Type:")
        type_rows = []
        for entity_type, stats in analysis["detection_analysis"]["by_type"].items():
            type_rows.append([
                entity_type,
                stats["total"],
                stats["detected"],
                f"{stats['detection_rate']*100:.1f}%"
            ])
        print(tabulate(type_rows, 
                      headers=["Entity Type", "Total", "Detected", "Detection Rate"]))
        
        # Show detector statistics
        print("\nDetections by Source:")
        detector_rows = [
            [detector, count] 
            for detector, count in analysis["detection_analysis"]["by_detector"].items()
        ]
        print(tabulate(detector_rows, headers=["Detector", "Count"]))
        
        # Show confidence distribution
        print("\nConfidence Distribution:")
        confidence_rows = [
            [range_key, count] 
            for range_key, count in sorted(
                analysis["detection_analysis"]["confidence_distribution"].items()
            )
        ]
        print(tabulate(confidence_rows, headers=["Confidence Range", "Count"]))
        
        # Show missed entities
        if analysis["missed_entities"]:
            print("\nMissed Entities:")
            missed_rows = [
                [e.text, e.entity_type] for e in analysis["missed_entities"]
            ]
            print(tabulate(missed_rows, headers=["Text", "Expected Type"]))


class StageTwoEvaluator:
    """Second stage: Parameter tuning and optimization"""
    def __init__(self):
        # Placeholder for future parameter tuning implementation
        pass

def main():
    """CLI entry point"""
    # Verify we're running from project root
    validate_project_root()
    
    parser = argparse.ArgumentParser(
        description="Redaction system evaluation tool",
        usage="python -m evaluation.evaluate [args]"
    )
    parser.add_argument("stage", choices=["detect", "tune"], 
                       help="Evaluation stage to run")
    parser.add_argument("--test", required=True,
                       help="Test ID to analyze")
    
    args = parser.parse_args()
    
    if args.stage == "detect":
        try:
            evaluator = StageOneEvaluator()
            analysis = evaluator.analyze_detection(args.test)
            evaluator.display_analysis(analysis)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    elif args.stage == "tune":
        print("Parameter tuning not yet implemented")

if __name__ == "__main__":
    main()