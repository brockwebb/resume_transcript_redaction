# /evaluation/test_runner.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from redactor.redactor_logic import RedactionProcessor
from .wrappers import RedactionWrapper
from .comparison.entity_matcher import EntityMatcher
from dataclasses import dataclass, asdict
from evaluation.metrics.entity_metrics import EntityMetrics
from evaluation.models import Entity, SensitivityLevel

###############################################################################
# SECTION 2: TEST CASE DEFINITIONS
###############################################################################

@dataclass
class TestCase:
    """Represents a single test case"""
    test_id: str
    original_path: Path
    annotation_path: Path
    description: Optional[str] = None

@dataclass
class TestRunResults:
    """Contains results for a complete test run"""
    run_id: str
    timestamp: str
    results: Dict[str, Dict]
    summary_metrics: Dict[str, float]

###############################################################################
# SECTION 3: CLASS DEFINITION AND CORE SETUP
###############################################################################

class TestRunner:
    """Orchestrates evaluation of the redaction system"""
    
    def __init__(self, test_suite_dir: str):
        """
        Initialize test runner with test suite directory.
        
        Args:
            test_suite_dir: Path to test suite directory
        """
        self.test_suite_dir = Path(test_suite_dir)
        self.entity_metrics = EntityMetrics()
        self.entity_matcher = EntityMatcher()
        
        # Initialize paths
        self.originals_dir = self.test_suite_dir / "originals"
        self.annotations_dir = self.test_suite_dir / "annotations"
        self.results_dir = self.test_suite_dir / "results"
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)

    # OPTIONS AND PARAMETERS
    def run_entity_evaluation(self, test_cases: List[TestCase], entity_type: str) -> TestRunResults:
        """Run evaluation for a specific entity type."""
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._setup_results_dir(run_id)
        redactor = self._initialize_processor()
        metrics = EntityMetrics(config_loader=self.config_loader)
        
        results = {}
        all_metrics = []
        
        for test_case in test_cases:
            # Load ground truth and run detection
            ground_truth = self._load_annotations(test_case.annotation_path)
            detected = redactor.detect_entities(test_case.original_path)
            
            # Get specific entity metrics
            detailed_metrics = metrics.analyze_entity_type_metrics(
                ground_truth=ground_truth,
                detected=detected,
                entity_type=entity_type
            )
            
            # Store results
            results[test_case.test_id] = detailed_metrics
            all_metrics.append(detailed_metrics)
            
            # Save individual test results
            self._save_test_results(run_dir, test_case.test_id, detailed_metrics)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(all_metrics)
        
        # Create run results
        run_results = TestRunResults(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            results=results,
            summary_metrics=summary_metrics
        )
        
        self._save_run_summary(run_dir, run_results)
        return run_results


    def load_test_cases(self) -> List[TestCase]:
        """
        Load test cases from test suite manifest.
        
        Returns:
            List of TestCase objects
        """
        manifest_path = self.test_suite_dir / "test_manifest.yaml"
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Test manifest not found at {manifest_path}")
            
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
            
        test_cases = []
        for test_info in manifest.get("test_cases", []):
            test_case = TestCase(
                test_id=test_info["id"],
                original_path=self.originals_dir / test_info["original_file"],
                annotation_path=self.annotations_dir / test_info["annotation_file"],
                description=test_info.get("description")
            )
            test_cases.append(test_case)
            
        return test_cases

    def _setup_results_dir(self, run_id: str) -> Path:
        """
        Setup results directory for current test run.
        
        Args:
            run_id: Unique identifier for test run
            
        Returns:
            Path to results directory
        """
        run_dir = self.results_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _load_annotations(self, annotation_path: Path) -> List[Entity]:
        """
        Load ground truth annotations from YAML file.
        
        Args:
            annotation_path: Path to annotation YAML file
            
        Returns:
            List of Entity objects
        """
        with open(annotation_path) as f:
            annotations = yaml.safe_load(f)
            
        entities = []
        for entity_data in annotations.get("entities", []):
            entity = Entity(
                text=entity_data["text"],
                entity_type=entity_data["type"],
                start_char=entity_data["location"]["start_char"],
                end_char=entity_data["location"]["end_char"],
                sensitivity=SensitivityLevel(entity_data["sensitivity"]),
                confidence=1.0,  # Ground truth entities have full confidence
                page=entity_data["location"].get("page", 1)
            )
            entities.append(entity)
            
        return entities

    def _initialize_processor(self) -> RedactionWrapper:
        """
        Initialize detection system with current configuration.
        
        Returns:
            Wrapped RedactionProcessor instance
        """
        processor = RedactionProcessor()
        return RedactionWrapper(processor)

###############################################################################
# SECTION 4: TEST EXECUTION AND RESULTS PROCESSING
###############################################################################

    def run_evaluation(self, test_cases: Optional[List[TestCase]] = None) -> TestRunResults:
        """
        Run evaluation on provided test cases or load from manifest.
        
        Args:
            test_cases: Optional list of test cases to run
            
        Returns:
            TestRunResults containing evaluation metrics
        """
        if test_cases is None:
            test_cases = self.load_test_cases()
           
        # Setup run tracking
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self._setup_results_dir(run_id)
       
        results = {}
        all_metrics = []
       
        # Initialize detection system
        redactor = self._initialize_processor()
       
        for test_case in test_cases:
            # Load ground truth and run detection
            ground_truth = self._load_annotations(test_case.annotation_path)
            detected_entities = redactor.detect_entities(test_case.original_path)
           
            # Match and analyze results
            matches = self.entity_matcher.find_matches(ground_truth, detected_entities)
            match_stats = self.entity_matcher.get_matching_statistics(matches)
           
            # Calculate metrics
            overall_metrics = self.entity_metrics.calculate_metrics(ground_truth, detected_entities)
            type_metrics = self.entity_metrics.get_type_based_metrics(matches)
           
            all_metrics.append({
                "overall": overall_metrics,
                "by_type": type_metrics
            })
           
            # Store results
            json_test_results = self._create_test_results(
                test_case.test_id,
                overall_metrics,
                type_metrics,
                matches,
                match_stats
            )
           
            results[test_case.test_id] = json_test_results
           
            # Save individual test results
            self._save_test_results(run_dir, test_case.test_id, json_test_results)
   
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(all_metrics)
   
        # Create and save run results
        run_results = TestRunResults(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            results=results,
            summary_metrics=summary_metrics
        )
       
        self._save_run_summary(run_dir, run_results)
           
        return run_results

    def _create_test_results(self, test_id: str, overall_metrics: Dict,
                           type_metrics: Dict, matches: Dict,
                           match_stats: Dict) -> Dict:
        """
        Create JSON-serializable test results.
        
        Args:
            test_id: Test case identifier
            overall_metrics: Overall performance metrics
            type_metrics: Metrics broken down by entity type
            matches: Entity matching results
            match_stats: Entity matching statistics
            
        Returns:
            Dictionary of test results
        """
        return {
            "test_id": test_id,
            "summary": {
                "overall_metrics": overall_metrics,
                "type_metrics": type_metrics
            },
            "details": {
                "matches": {
                    k: [e.to_json_dict() if hasattr(e, 'to_json_dict') else str(e)
                        for e in v] if isinstance(v, set) else v
                    for k, v in matches.items()
                },
                "match_statistics": match_stats
            }
        }

    def _save_test_results(self, run_dir: Path, test_id: str, 
                          results: Dict) -> None:
        """
        Save individual test results to file.
        
        Args:
            run_dir: Directory for run results
            test_id: Test case identifier
            results: Test results to save
        """
        result_path = run_dir / f"{test_id}_results.json"
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)

    def _save_run_summary(self, run_dir: Path, run_results: TestRunResults) -> None:
        """
        Save overall run results summary.
        
        Args:
            run_dir: Directory for run results
            run_results: Overall run results to save
        """
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(run_results), f, indent=2)

    @staticmethod
    def _calculate_summary_metrics(metrics_list: List[Dict]) -> Dict:
        """
        Recursively calculate average metrics across all test cases.
        
        Args:
            metrics_list: List of metrics dictionaries from individual test cases
            
        Returns:
            Dictionary of averaged metrics
        """
        if not metrics_list:
            return {}
        
        summary = {}
        # Get all keys from the first metrics dictionary
        keys = metrics_list[0].keys()
        
        for key in keys:
            # Collect values for this key from all metrics
            values = [m.get(key) for m in metrics_list]
            
            # If all values are numeric, calculate average
            if all(isinstance(v, (int, float)) for v in values if v is not None):
                summary[key] = sum(v for v in values if v is not None) / len([v for v in values if v is not None])
            # If all values are dictionaries, recursively average
            elif all(isinstance(v, dict) for v in values if v is not None):
                summary[key] = TestRunner._calculate_summary_metrics(values)
            # If mixed types, keep the first value (or None)
            else:
                summary[key] = values[0]
        
        return summary

###############################################################################
# SECTION 5: COMPARISON AND ANALYSIS
###############################################################################

    def compare_runs(self, run_id1: str, run_id2: str) -> Dict:
        """
        Compare metrics between two test runs.
        
        Args:
            run_id1: ID of first run to compare
            run_id2: ID of second run to compare
            
        Returns:
            Dictionary containing metric differences
        """
        run1_path = self.results_dir / f"run_{run_id1}" / "summary.json"
        run2_path = self.results_dir / f"run_{run_id2}" / "summary.json"
        
        with open(run1_path) as f:
            run1_results = json.load(f)
        with open(run2_path) as f:
            run2_results = json.load(f)
            
        differences = {}
        for metric, value1 in run1_results["summary_metrics"].items():
            value2 = run2_results["summary_metrics"][metric]
            differences[metric] = {
                "run1": value1,
                "run2": value2,
                "difference": value2 - value1,
                "percent_change": ((value2 - value1) / value1 * 100) if value1 != 0 else float("inf")
            }
            
        return differences

    def analyze_type_performance(self, run_id: str) -> Dict:
        """
        Analyze performance by entity type for a specific run.
        
        Args:
            run_id: Run ID to analyze
            
        Returns:
            Dictionary containing per-type analysis
        """
        run_dir = self.results_dir / f"run_{run_id}"
        summary_path = run_dir / "summary.json"
        
        with open(summary_path) as f:
            run_results = json.load(f)
            
        type_analysis = {}
        for test_id, results in run_results["results"].items():
            type_metrics = results["summary"]["type_metrics"]
            
            for entity_type, metrics in type_metrics.items():
                if entity_type not in type_analysis:
                    type_analysis[entity_type] = {
                        "total_detected": 0,
                        "true_positives": 0,
                        "false_positives": 0,
                        "false_negatives": 0,
                        "test_cases": []
                    }
                
                analysis = type_analysis[entity_type]
                
                # Safely get metrics with default values
                analysis["total_detected"] += metrics.get("total_detected", 0)
                analysis["true_positives"] += metrics.get("true_positives", 0)
                analysis["false_positives"] += metrics.get("false_positives", 0)
                analysis["false_negatives"] += metrics.get("false_negatives", 0)
                
                analysis["test_cases"].append({
                    "test_id": test_id,
                    "metrics": metrics
                })
            
        # Calculate aggregate metrics for each type
        for entity_type in type_analysis:
            stats = type_analysis[entity_type]
            total = stats["true_positives"] + stats["false_positives"]
            if total > 0:
                precision = stats["true_positives"] / total
            else:
                precision = 0
                
            total = stats["true_positives"] + stats["false_negatives"]
            if total > 0:
                recall = stats["true_positives"] / total
            else:
                recall = 0
                
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
                
            type_analysis[entity_type]["aggregate_metrics"] = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1
            }
            
        return type_analysis

    def analyze_confidence_distribution(self, run_id: str) -> Dict:
        """
        Analyze distribution of confidence scores for a run.
        
        Args:
            run_id: Run ID to analyze
            
        Returns:
            Dictionary containing confidence score analysis
        """
        run_dir = self.results_dir / f"run_{run_id}"
        
        confidence_ranges = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0
        }
        
        entity_confidences = {
            "true_positives": confidence_ranges.copy(),
            "false_positives": confidence_ranges.copy()
        }
        
        # Process each test case
        for result_file in run_dir.glob("*_results.json"):
            if result_file.name == "summary.json":
                continue
                
            with open(result_file) as f:
                results = json.load(f)
            
            matches = results["details"]["matches"]
            
            # Process true positives
            for entity in matches.get("true_positives", []):
                if isinstance(entity, dict) and "confidence" in entity:
                    confidence = entity["confidence"]
                    range_key = self._get_confidence_range(confidence)
                    entity_confidences["true_positives"][range_key] += 1
            
            # Process false positives
            for entity in matches.get("false_positives", []):
                if isinstance(entity, dict) and "confidence" in entity:
                    confidence = entity["confidence"]
                    range_key = self._get_confidence_range(confidence)
                    entity_confidences["false_positives"][range_key] += 1
        
        return entity_confidences

    @staticmethod
    def _get_confidence_range(confidence: float) -> str:
        """
        Get confidence range key for a confidence value.
        
        Args:
            confidence: Confidence value to categorize
            
        Returns:
            String key for confidence range
        """
        if confidence < 0.2:
            return "0.0-0.2"
        elif confidence < 0.4:
            return "0.2-0.4"
        elif confidence < 0.6:
            return "0.4-0.6"
        elif confidence < 0.8:
            return "0.6-0.8"
        else:
            return "0.8-1.0"

###############################################################################
# SECTION 6: MAIN EXECUTION
###############################################################################

def main():
    """CLI entry point for test runner."""
    try:
        # Initialize test runner
        test_runner = TestRunner("data/test_suite")
        
        # Load test cases
        test_cases = test_runner.load_test_cases()
        
        # Initialize detection system and run evaluation
        results = test_runner.run_evaluation(test_cases)
        
        # Recursively print metrics
        def print_metrics(metrics, indent=""):
            for metric, value in metrics.items():
                if isinstance(value, dict):
                    print(f"{indent}{metric}:")
                    print_metrics(value, indent + "  ")
                elif isinstance(value, (int, float)):
                    print(f"{indent}{metric}: {value:.3f}")
                else:
                    print(f"{indent}{metric}: {value}")
        
        # Print summary metrics
        print("\nTest Run Summary:")
        print("=================")
        print_metrics(results.summary_metrics)
            
        # Analyze type performance
        type_analysis = test_runner.analyze_type_performance(results.run_id)
        
        print("\nPerformance by Entity Type:")
        print("==========================")
        for entity_type, analysis in type_analysis.items():
            metrics = analysis["aggregate_metrics"]
            print(f"\n{entity_type}:")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1 Score: {metrics['f1_score']:.3f}")
            print(f"  Total detected: {analysis['total_detected']}")
            
        # Analyze confidence distribution
        confidence_analysis = test_runner.analyze_confidence_distribution(results.run_id)
        
        print("\nConfidence Score Distribution:")
        print("=============================")
        for category in ["true_positives", "false_positives"]:
            print(f"\n{category.replace('_', ' ').title()}:")
            for range_key, count in confidence_analysis[category].items():
                print(f"  {range_key}: {count}")
                
    except FileNotFoundError as e:
        print(f"\nError: Test suite directory not found - {e}")
        print("Make sure you're running from the project root directory")
    except Exception as e:
        print(f"\nError running tests: {e}")
        raise

if __name__ == "__main__":
    main()