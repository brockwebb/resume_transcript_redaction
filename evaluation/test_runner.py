# /evaluation/test_runner.py

import os
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .metrics.entity_metrics import EntityMetrics, Entity, SensitivityLevel
from .comparison.entity_matcher import EntityMatcher

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

class TestRunner:
    """Orchestrates evaluation of the redaction system"""
    
    def __init__(self, test_suite_dir: str):
        self.test_suite_dir = Path(test_suite_dir)
        self.entity_metrics = EntityMetrics()
        self.entity_matcher = EntityMatcher()
        
        # Initialize paths
        self.originals_dir = self.test_suite_dir / "originals"
        self.annotations_dir = self.test_suite_dir / "annotations"
        self.results_dir = self.test_suite_dir / "results"
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_test_cases(self) -> List[TestCase]:
        """
        Load test cases from test suite manifest
        
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
    
    def run_evaluation(self, redactor, test_cases: Optional[List[TestCase]] = None) -> TestRunResults:
        """
        Run evaluation on specified test cases
        
        Args:
            redactor: Instance of redaction system to evaluate
            test_cases: Optional list of specific test cases to run
                       If None, all test cases are run
                       
        Returns:
            TestRunResults containing evaluation metrics
        """
        if test_cases is None:
            test_cases = self.load_test_cases()
            
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.results_dir / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        
        results = {}
        all_metrics = []
        
        for test_case in test_cases:
            # Load ground truth annotations
            ground_truth = self._load_annotations(test_case.annotation_path)
            
            # Run redaction system
            detected_entities = redactor.detect_entities(test_case.original_path)
            
            # Compare results
            matches = self.entity_matcher.find_matches(ground_truth, detected_entities)
            match_stats = self.entity_matcher.get_matching_statistics(matches)
            
            # Calculate metrics
            metrics = self.entity_metrics.calculate_metrics(ground_truth, detected_entities)
            all_metrics.append(metrics)
            
            # Store detailed results
            test_results = {
                "test_id": test_case.test_id,
                "matches": matches,
                "match_statistics": match_stats,
                "metrics": metrics
            }
            results[test_case.test_id] = test_results
            
            # Save individual test results
            result_path = run_dir / f"{test_case.test_id}_results.json"
            with open(result_path, "w") as f:
                json.dump(test_results, f, indent=2)
        
        # Calculate summary metrics
        summary_metrics = self._calculate_summary_metrics(all_metrics)
        
        # Create run results
        run_results = TestRunResults(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            results=results,
            summary_metrics=summary_metrics
        )
        
        # Save summary results
        summary_path = run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(asdict(run_results), f, indent=2)
            
        return run_results
    
    def _load_annotations(self, annotation_path: Path) -> List[Entity]:
        """
        Load ground truth annotations from YAML file
        
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
    
    @staticmethod
    def _calculate_summary_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calculate average metrics across all test cases
        
        Args:
            metrics_list: List of metrics dictionaries from individual test cases
            
        Returns:
            Dictionary of averaged metrics
        """
        if not metrics_list:
            return {}
            
        summary = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list]
            summary[key] = sum(values) / len(values)
            
        return summary

    def compare_runs(self, run_id1: str, run_id2: str) -> Dict:
        """
        Compare metrics between two test runs
        
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