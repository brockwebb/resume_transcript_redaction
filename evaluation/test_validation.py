# evaluation/test_validation.py
from pathlib import Path
import pytest
from redactor.detectors.spacy_detector import SpacyDetector
from redactor.detectors.presidio_detector import PresidioDetector
from redactor.detectors.base_detector import Entity
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
import json

class BaseEntityValidationTest:
    """Base class for entity validation tests providing common setup"""
    
    @pytest.fixture
    def setup_detector(self):
        """Setup shared detector configuration"""
        self.logger = RedactionLogger(
            name="test_validation",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()
        return self._get_detector()
        
    def _get_detector(self):
        """Override in child classes to return appropriate detector"""
        raise NotImplementedError("Detector must be specified in child class")
        
    def _load_test_cases(self, filename: str) -> list:
        """Load test cases from JSON file"""
        test_cases_path = Path("evaluation/test_cases") / filename
        assert test_cases_path.exists(), f"Test cases file not found: {test_cases_path}"
        with open(test_cases_path) as f:
            return json.load(f)

class TestEducationalValidation(BaseEntityValidationTest):
    """Test validation rules for educational institutions"""
    
    def _get_detector(self):
        return SpacyDetector(self.config_loader, self.logger)
        
    def test_educational_validation(self, setup_detector):
        """Test educational institution validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("educational_validation.json")["educational_validation_tests"]
        
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

class TestPersonValidation(BaseEntityValidationTest):
    """Test validation rules for person names"""
    
    def _get_detector(self):
        return SpacyDetector(self.config_loader, self.logger)
        
    def test_person_validation(self, setup_detector):
        """Test person name validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("person_validation.json")["person_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PERSON",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="spacy"
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )

class TestDateValidation(BaseEntityValidationTest):
    """Test validation rules for dates"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_date_validation(self, setup_detector):
        """Test date validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("date_validation.json")["date_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="DATE_TIME",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "date_pattern"
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )

class TestLocationValidation(BaseEntityValidationTest):
    """Test validation rules for locations"""
    
    def _get_detector(self):
        return SpacyDetector(self.config_loader, self.logger)
        
    def test_location_validation(self, setup_detector):
        """Test location validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("location_validation.json")["location_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="LOCATION",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="spacy"
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )

class TestAddressValidation(BaseEntityValidationTest):
    """Test validation rules for addresses"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)  # Changed to PresidioDetector
        
    def test_address_validation(self, setup_detector):
        """Test address validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("address_validation.json")["address_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="ADDRESS",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio",  # Changed source to presidio
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "address_pattern"
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )

class TestGPAValidation(BaseEntityValidationTest):
    """Test validation rules for GPA entities"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_gpa_validation(self, setup_detector):
        """Test GPA validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("gpa_validation.json")["gpa_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="GPA",
                confidence=0.9,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "gpa_standard_format"
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )

class TestPhoneValidation(BaseEntityValidationTest):
    """Test validation rules for phone number entities"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_phone_validation(self, setup_detector):
        """Test phone validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("phone_validation.json")["phone_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PHONE_NUMBER",
                confidence=0.9,  # Starting with high confidence like GPA
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "us_standard"  # Using default pattern
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, "")
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )
            
            if is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, got {confidence_change}. "
                    f"Validation reasons: {detector.last_validation_reasons}"
                )