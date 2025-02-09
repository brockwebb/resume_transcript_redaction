#!/usr/bin/env python
# -------------------
# Section 1: Imports
# -------------------
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List, Dict

from .test_runner import TestRunner, TestCase

# -------------------
# Section 2: Logging Setup 
# -------------------
root_logger = logging.getLogger()
handler = logging.StreamHandler()
handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(message)s')
handler.setFormatter(formatter)
root_logger.addHandler(handler)
root_logger.setLevel(logging.ERROR)

# Disable noisy loggers
for logger_name in ['app.utils.config_loader', 'evaluation', 'presidio']:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# -------------------
# Section 3: Singleton Pattern
# -------------------
class SingletonMeta(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

# -------------------
# Section 4: Config Manager
# -------------------
class ConfigManager(metaclass=SingletonMeta):
    def __init__(self):
        self._config_loader = None
        self._initialized = False

    @property
    def config_loader(self):
        if not self._initialized:
            self.initialize()
        return self._config_loader

    def initialize(self):
        if not self._initialized:
            # Import the global instance from our config_loader module.
            from app.utils.config_loader import global_config_loader
            # Ensure that the configuration has been loaded.
            if not global_config_loader._loaded:
                global_config_loader.load_all_configs()
            self._config_loader = global_config_loader
            self._initialized = True

# -------------------
# Section 5: Focused Test Runner
# -------------------
class FocusedTestRunner(TestRunner):
    def __init__(self, test_suite_dir: str):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.ERROR)
        self._config_manager = ConfigManager()
        super().__init__(test_suite_dir)

    @property
    def config_loader(self):
        return self._config_manager.config_loader

    def _run_single_test(self, test_case: TestCase, entity_type: str,
                         confidence_threshold: float) -> Dict:
        """Run a single test case with minimal output."""
        # Load ground truth and detected entities.
        ground_truth = self._load_annotations(test_case.annotation_path)
        detected_entities = self._initialize_processor().detect_entities(
            test_case.original_path
        )
        # Filter entities by entity type.
        if entity_type:
            ground_truth = [e for e in ground_truth if e.entity_type == entity_type]
            detected_entities = [e for e in detected_entities if e.entity_type == entity_type]
        # Filter by confidence threshold if provided.
        if confidence_threshold is not None:
            detected_entities = [
                e for e in detected_entities 
                if e.confidence >= confidence_threshold
            ]
        # Calculate metrics.
        matches = self.entity_matcher.find_matches(ground_truth, detected_entities)
        match_stats = self.entity_matcher.get_matching_statistics(matches)
        tp = len(matches.get("true_positives", []))
        fp = len(matches.get("false_positives", []))
        fn = len(matches.get("false_negatives", []))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn
            },
            "matches": matches,
            "statistics": match_stats
        }

    def run_focused_evaluation(self, 
                               entity_type: Optional[str] = None,
                               test_ids: Optional[List[str]] = None,
                               confidence_threshold: Optional[float] = None,
                               quiet: bool = False) -> dict:
        """Run evaluation with minimal output."""
        # Load test cases.
        test_cases = self.load_test_cases()
        if test_ids:
            test_cases = [tc for tc in test_cases if tc.test_id in test_ids]
            if not test_cases:
                raise ValueError(f"No test cases found matching IDs: {test_ids}")
        results = {}
        for test_case in test_cases:
            result = self._run_single_test(
                test_case, 
                entity_type,
                confidence_threshold
            )
            results[test_case.test_id] = result

            if not quiet:
                metrics = result["metrics"]
                print(f"{test_case.test_id}:")
                print(f"  Precision: {metrics['precision']:.3f}")
                print(f"  Recall: {metrics['recall']:.3f}")
                print(f"  F1 Score: {metrics['f1_score']:.3f}")
                print(f"  True Positives: {metrics['true_positives']}")
                print(f"  False Positives: {metrics['false_positives']}")
                print(f"  False Negatives: {metrics['false_negatives']}\n")
        return results

def main():
    parser = argparse.ArgumentParser(
        description="Focused entity detection evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --entity-type EMAIL_ADDRESS
  %(prog)s --entity-type PERSON --confidence 0.8
  %(prog)s --test-ids alexis_rivera john_smith
  %(prog)s --entity-type LOCATION --format detailed
"""
    )
    parser.add_argument("--entity-type", "-e",
                        help="Entity type to evaluate (e.g., EMAIL_ADDRESS, PERSON)")
    parser.add_argument("--test-ids", "-t", nargs="+",
                        help="Specific test IDs to evaluate")
    parser.add_argument("--confidence", "-c", type=float,
                        help="Confidence threshold to apply")
    parser.add_argument("--format", "-f", choices=["detailed", "summary"],
                        default="summary", help="Output format")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Minimize output")
    args = parser.parse_args()

    try:
        runner = FocusedTestRunner("data/test_suite")
        results = runner.run_focused_evaluation(
            entity_type=args.entity_type,
            test_ids=args.test_ids,
            confidence_threshold=args.confidence,
            quiet=args.quiet
        )
        if args.format == "detailed" and not args.quiet:
            print("\nDetailed Results:")
            print("================")
            for test_id, result in results.items():
                print(f"\nTest Case: {test_id}")
                for match_type in ["false_positives", "false_negatives"]:
                    entities = result["matches"].get(match_type, [])
                    if entities:
                        print(f"\n{match_type.replace('_', ' ').title()}:")
                        for entity in entities:
                            print(f"  - {entity.text} ({entity.confidence:.2f})")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if not args.quiet:
            raise

if __name__ == "__main__":
    main()
