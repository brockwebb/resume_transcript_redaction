# evaluation/test_validation.py
from pathlib import Path
import pytest
from redactor.detectors.spacy_detector import SpacyDetector
from redactor.detectors.base_detector import Entity
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
import json


class TestEducationalValidation:
    @pytest.fixture
    def setup_detector(self):
        """Setup SpacyDetector with logger"""
        logger = RedactionLogger(
            name="test_validation",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )
        config_loader = ConfigLoader()
        config_loader.load_all_configs()
        return SpacyDetector(config_loader, logger)

    def test_educational_validation(self, setup_detector):
        """Test educational institution validation with different cases"""
        detector = setup_detector

        # Load test cases from JSON
        test_cases_path = Path("evaluation/test_cases/educational_validation.json")
        assert test_cases_path.exists(), f"Test cases file not found: {test_cases_path}"
        with open(test_cases_path) as f:
            test_cases = json.load(f)["educational_validation_tests"]

        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="EDUCATIONAL_INSTITUTION",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="spacy"
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            # Assertions
            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Reasons: {detector.last_validation_reasons}"
            )
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}"
                )

    def test_validation_fallback(self, setup_detector):
        """Test fallback to old validation when new validation fails"""
        detector = setup_detector

        # Temporarily rename validation params to force fallback
        params_path = Path("redactor/config/validation_params.json")
        temp_path = params_path.with_suffix(".json.bak")

        if params_path.exists():
            params_path.rename(temp_path)

        try:
            entity = Entity(
                text="Harvard University",
                entity_type="EDUCATIONAL_INSTITUTION",
                confidence=0.8,
                start=0,
                end=len("Harvard University"),
                source="spacy"
            )

            # Should still validate using old method
            is_valid = detector.validate_detection(entity, "")
            assert is_valid, "Fallback validation failed"

        finally:
            # Restore validation params
            if temp_path.exists():
                temp_path.rename(params_path)
