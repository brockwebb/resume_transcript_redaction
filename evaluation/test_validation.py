# evaluation/test_validation.py
from pathlib import Path
import pytest
from redactor.detectors.spacy_detector import SpacyDetector
from redactor.detectors.presidio_detector import PresidioDetector 
from redactor.detectors.ensemble_coordinator import EnsembleCoordinator 
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

class TestEmailValidation(BaseEntityValidationTest):
    """Test validation rules for email addresses"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_email_validation(self, setup_detector):
        """Test email validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("email_validation.json")["email_validation_tests"]
        
        for test_case in test_cases:
            text = test_case['text']
            entity = Entity(
                text=text,
                entity_type="EMAIL_ADDRESS",
                start=0,
                end=len(text),
                confidence=0.7,
                source="test"
            )

            is_valid = detector._validate_email_address(entity, text)
            
            if test_case['expected_valid']:
                assert is_valid, f"Failed to validate valid email: {text}"
                if 'min_confidence_adjustment' in test_case:
                    assert entity.confidence >= test_case['min_confidence_adjustment'], \
                        f"Confidence too low for {text}: {entity.confidence}"
            else:
                assert not is_valid, f"Failed to reject invalid email: {text}"

    def test_email_context_boost(self, setup_detector):
        """Test that email validation considers context"""
        detector = setup_detector
        text = "Contact: user@example.com"
        entity = Entity(
            text="user@example.com",
            entity_type="EMAIL_ADDRESS",
            start=9,
            end=24,
            confidence=0.7,
            source="test"
        )

        initial_confidence = entity.confidence
        is_valid = detector._validate_email_address(entity, text)
        assert is_valid
        assert entity.confidence > initial_confidence, "Context boost not applied"


class TestInternetReferenceValidation(BaseEntityValidationTest):
    """Test validation rules for internet references"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_internet_reference_validation(self, setup_detector):
        """Test internet reference validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("internet_reference_validation.json")["internet_reference_validation_tests"]
        
        for test_case in test_cases:
            text = test_case['text']
            entity = Entity(
                text=text,
                entity_type="INTERNET_REFERENCE",
                start=0,
                end=len(text),
                confidence=0.7,
                source="test"
            )

            is_valid = detector._validate_internet_reference(entity, text)
            
            if test_case['expected_valid']:
                assert is_valid, f"Failed to validate valid reference: {text}"
                if 'min_confidence_adjustment' in test_case:
                    assert round(entity.confidence, 2) >= test_case['min_confidence_adjustment'], \
                        f"Confidence too low for {text}: {round(entity.confidence, 2)}"
            else:
                assert not is_valid, f"Failed to reject invalid reference: {text}"

    def test_social_handle_context(self, setup_detector):
        """Test that social handle validation considers context"""
        detector = setup_detector
        text = "Follow me on Twitter: @username"
        entity = Entity(
            text="@username",
            entity_type="INTERNET_REFERENCE",
            start=20,
            end=29,
            confidence=0.7,
            source="test"
        )

        initial_confidence = entity.confidence
        is_valid = detector._validate_internet_reference(entity, text)
        assert is_valid
        assert entity.confidence > initial_confidence, "Context boost not applied"

    def test_url_context(self, setup_detector):
        """Test that URL validation considers context"""
        detector = setup_detector
        text = "Visit our website: https://example.com"
        entity = Entity(
            text="https://example.com",
            entity_type="INTERNET_REFERENCE",
            start=17,
            end=35,
            confidence=0.7,
            source="test"
        )

        initial_confidence = entity.confidence
        is_valid = detector._validate_internet_reference(entity, text)
        assert is_valid
        assert entity.confidence > initial_confidence, "Context boost not applied"

    def test_known_platform_boost(self, setup_detector):
        """Test confidence boost for known platforms"""
        detector = setup_detector
        text = "github.com/username"
        entity = Entity(
            text=text,
            entity_type="INTERNET_REFERENCE",
            start=0,
            end=len(text),
            confidence=0.7,
            source="test"
        )

        initial_confidence = entity.confidence
        is_valid = detector._validate_internet_reference(entity, text)
        assert is_valid
        assert entity.confidence > initial_confidence, "Known platform boost not applied"


class TestProtectedClassValidation(BaseEntityValidationTest):
    """Test validation rules for protected class entities"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_protected_class_validation(self, setup_detector):
        """Test protected class validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("protected_class_validation.json")["protected_class_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PROTECTED_CLASS",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "protected_class_pattern"
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, case.get("context", ""))
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

class TestPHIValidation(BaseEntityValidationTest):
    """Test validation rules for PHI entities"""
    
    def _get_detector(self):
        return PresidioDetector(self.config_loader, self.logger)
        
    def test_phi_validation(self, setup_detector):
        """Test PHI validation with different cases"""
        detector = setup_detector
        test_cases = self._load_test_cases("phi_validation.json")["phi_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PHI",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "phi_pattern"
                    }
                }
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, case.get("context", ""))
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


class TestSpacyProtectedClassValidation(BaseEntityValidationTest):
    """Test spaCy validation for protected class entities"""
    
    def _get_detector(self):
        return SpacyDetector(self.config_loader, self.logger)
        
    def test_spacy_protected_class_validation(self, setup_detector):
        """Test protected class validation with spaCy context detection"""
        detector = setup_detector
        test_cases = self._load_test_cases("protected_class_validation.json")["protected_class_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PROTECTED_CLASS",
                confidence=0.7,
                start=0,
                end=len(case["text"]),
                source="spacy"
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, case.get("context", ""))
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )

class TestSpacyPHIValidation(BaseEntityValidationTest):
    """Test spaCy validation for PHI entities"""
    
    def _get_detector(self):
        return SpacyDetector(self.config_loader, self.logger)
        
    def test_spacy_phi_validation(self, setup_detector):
        """Test PHI validation with spaCy context detection"""
        detector = setup_detector
        test_cases = self._load_test_cases("phi_validation.json")["phi_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PHI",
                confidence=0.7,
                start=0,
                end=len(case["text"]),
                source="spacy"
            )

            initial_confidence = entity.confidence
            is_valid = detector.validate_detection(entity, case.get("context", ""))
            confidence_change = round(entity.confidence - initial_confidence, 2) if is_valid else 0.0

            assert is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, got {is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, Adjustments: {confidence_change}, "
                f"Validation reasons: {detector.last_validation_reasons}"
            )

class TestEnsembleProtectedPHI(BaseEntityValidationTest):
    """Test ensemble validation for protected class and PHI entities"""
    
    def _get_detector(self):
        return EnsembleCoordinator(self.config_loader, self.logger)
        
    def test_ensemble_validation(self, setup_detector):
        """Test ensemble validation combining Presidio and spaCy results"""
        detector = setup_detector
        
        # Test protected class ensemble
        protected_cases = self._load_test_cases("protected_class_validation.json")["protected_class_validation_tests"]
        for case in protected_cases:
            if case["expected_valid"]:
                # Create matching detections from both sources
                presidio_ent = Entity(
                    text=case["text"],
                    entity_type="PROTECTED_CLASS",
                    confidence=0.75,
                    start=0,
                    end=len(case["text"]),
                    source="presidio"
                )
                spacy_ent = Entity(
                    text=case["text"],
                    entity_type="PROTECTED_CLASS",
                    confidence=0.8,
                    start=0,
                    end=len(case["text"]),
                    source="spacy"
                )
                
                combined = detector._combine_protected_phi_entities(presidio_ent, spacy_ent)
                assert combined is not None, f"Failed to combine entities for: {case['description']}"
                assert combined.confidence >= max(presidio_ent.confidence, spacy_ent.confidence), \
                    f"Combined confidence not higher than individual for: {case['description']}"
        
        # Test PHI ensemble
        phi_cases = self._load_test_cases("phi_validation.json")["phi_validation_tests"]
        for case in phi_cases:
            if case["expected_valid"]:
                presidio_ent = Entity(
                    text=case["text"],
                    entity_type="PHI",
                    confidence=0.75,
                    start=0,
                    end=len(case["text"]),
                    source="presidio"
                )
                spacy_ent = Entity(
                    text=case["text"],
                    entity_type="PHI",
                    confidence=0.8,
                    start=0,
                    end=len(case["text"]),
                    source="spacy"
                )
                
                combined = detector._combine_protected_phi_entities(presidio_ent, spacy_ent)
                assert combined is not None, f"Failed to combine entities for: {case['description']}"
                assert combined.confidence >= max(presidio_ent.confidence, spacy_ent.confidence), \
                    f"Combined confidence not higher than individual for: {case['description']}"