# evaluation/test_validation.py

# evaluation/test_validation.py

###############################################################################
# SECTION 1: IMPORTS AND SETUP
###############################################################################

from pathlib import Path
import pytest
from redactor.validation import ValidationCoordinator
from redactor.detectors.base_detector import Entity
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
import json

###############################################################################
# SECTION 2: BASE TEST CLASS
###############################################################################

class BaseValidationTest:
    """Base class for validation tests providing common setup."""
    
    @pytest.fixture
    def setup_validation(self):
        """Setup shared validation configuration."""
        self.logger = RedactionLogger(
            name="test_validation",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()
        
        # Initialize validator
        self.validator = ValidationCoordinator(
            config_loader=self.config_loader,
            logger=self.logger
        )
        return self.validator
        
    def _load_test_cases(self, filename: str) -> list:
        """Load test cases from JSON file."""
        test_cases_path = Path("evaluation/test_cases") / filename
        assert test_cases_path.exists(), f"Test cases file not found: {test_cases_path}"
        with open(test_cases_path) as f:
            return json.load(f)

###############################################################################
# SECTION 3: EDUCATIONAL INSTITUTION TESTS
###############################################################################

class TestEducationalValidation(BaseValidationTest):
    """Test validation rules for educational institutions."""
    
    def test_educational_validation(self, setup_validation):
        """Test educational institution validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}"
                )

###############################################################################
# SECTION 4: PERSON VALIDATION TESTS
###############################################################################

class TestPersonValidation(BaseValidationTest):
    """Test validation rules for person names."""
    
    def test_person_validation(self, setup_validation):
        """Test person name validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )

###############################################################################
# SECTION 5: DATE VALIDATION TESTS
###############################################################################

class TestDateValidation(BaseValidationTest):
    """Test validation rules for dates."""
    
    def test_date_validation(self, setup_validation):
        """Test date validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}. "
                    f"Reasons: {result.reasons}"
                )

###############################################################################
# SECTION 6: LOCATION VALIDATION TESTS
###############################################################################

class TestLocationValidation(BaseValidationTest):
    """Test validation rules for locations."""
    
    def test_location_validation(self, setup_validation):
        """Test location validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}. "
                    f"Reasons: {result.reasons}"
                )
                
    def test_location_formats(self, setup_validation):
        """Test specific location format validations."""
        validator = setup_validation
        
        # Test US city-state format
        entity = Entity(
            text="San Francisco, CA 94105",
            entity_type="LOCATION",
            confidence=0.8,
            start=0,
            end=22,
            source="spacy"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate US city-state format: {result.reasons}"
        assert any("city-state" in reason.lower() for reason in result.reasons), \
            "Missing city-state format validation reason"
        
        # Test international format
        entity = Entity(
            text="München",
            entity_type="LOCATION",
            confidence=0.8,
            start=0,
            end=7,
            source="spacy"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate international format: {result.reasons}"
        assert any("international" in reason.lower() for reason in result.reasons), \
            "Missing international format validation reason"

###############################################################################
# SECTION 7: ADDRESS VALIDATION TESTS
###############################################################################

class TestAddressValidation(BaseValidationTest):
    """Test validation rules for addresses."""
    
    def test_address_validation(self, setup_validation):
        """Test address validation with different cases."""
        validator = setup_validation
        test_cases = self._load_test_cases("address_validation.json")["address_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="ADDRESS",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "address_pattern"
                    }
                }
            )

            initial_confidence = entity.confidence
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}. "
                    f"Reasons: {result.reasons}"
                )
                
    def test_address_patterns(self, setup_validation):
        """Test specific address format patterns."""
        validator = setup_validation
        
        # Test street address format
        entity = Entity(
            text="123 Main Street",
            entity_type="ADDRESS",
            confidence=0.8,
            start=0,
            end=14,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate basic street address: {result.reasons}"
        
        # Test apartment format
        entity = Entity(
            text="456 Oak Ave, Apt 2B",
            entity_type="ADDRESS",
            confidence=0.8,
            start=0,
            end=18,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate address with apartment: {result.reasons}"
        assert any("apartment" in reason.lower() or "unit" in reason.lower() 
                  for reason in result.reasons), \
            "Missing apartment format validation reason"

###############################################################################
# SECTION 8: GPA VALIDATION TESTS
###############################################################################

class TestGPAValidation(BaseValidationTest):
    """Test validation rules for GPA entities."""
    
    def test_gpa_validation(self, setup_validation):
        """Test GPA validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}. "
                    f"Reasons: {result.reasons}"
                )

    def test_gpa_edge_cases(self, setup_validation):
        """Test GPA validation edge cases."""
        validator = setup_validation
        
        # Test exact 4.0
        entity = Entity(
            text="4.0",
            entity_type="GPA",
            confidence=0.9,
            start=0,
            end=3,
            source="presidio"
        )
        result = validator.validate_entity(entity, "GPA: 4.0")
        assert result.is_valid, f"Failed to validate 4.0 GPA: {result.reasons}"
        
        # Test invalid high value
        entity = Entity(
            text="4.1",
            entity_type="GPA",
            confidence=0.9,
            start=0,
            end=3,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert not result.is_valid, "Should not validate GPA > 4.0"
        
        # Test with scale indicator
        entity = Entity(
            text="3.75/4.0",
            entity_type="GPA",
            confidence=0.9,
            start=0,
            end=8,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate GPA with scale: {result.reasons}"
        assert any("scale" in reason.lower() for reason in result.reasons), \
            "Missing scale indicator validation reason"


###############################################################################
# SECTION 9: COMMUNICATION VALIDATION TESTS
###############################################################################

class TestPhoneValidation(BaseValidationTest):
    """Test validation rules for phone numbers."""
    
    def test_phone_validation(self, setup_validation):
        """Test phone number validation with different cases."""
        validator = setup_validation
        test_cases = self._load_test_cases("phone_validation.json")["phone_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="PHONE_NUMBER",
                confidence=0.9,
                start=0,
                end=len(case["text"]),
                source="presidio",
                metadata={
                    "recognition_metadata": {
                        "pattern_name": "us_standard"
                    }
                }
            )

            initial_confidence = entity.confidence
            result = validator.validate_entity(entity, "")
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )
            
            if result.is_valid and "min_confidence_adjustment" in case:
                assert confidence_change >= round(case["min_confidence_adjustment"], 2), (
                    f"Insufficient confidence adjustment for: {case['description']}. "
                    f"Expected at least {round(case['min_confidence_adjustment'], 2)}, "
                    f"got {confidence_change}. "
                    f"Reasons: {result.reasons}"
                )

    def test_phone_formats(self, setup_validation):
        """Test validation of different phone number formats."""
        validator = setup_validation
        
        # Test standard US format
        entity = Entity(
            text="(555) 123-4567",
            entity_type="PHONE_NUMBER",
            confidence=0.8,
            start=0,
            end=14,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate standard US format: {result.reasons}"
        assert any("standard format" in reason.lower() for reason in result.reasons), \
            "Missing standard format validation reason"
        
        # Test with area code
        entity = Entity(
            text="1-555-123-4567",
            entity_type="PHONE_NUMBER",
            confidence=0.8,
            start=0,
            end=14,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate number with area code: {result.reasons}"
        
        # Test with extension
        entity = Entity(
            text="555-123-4567 ext. 890",
            entity_type="PHONE_NUMBER",
            confidence=0.8,
            start=0,
            end=20,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate number with extension: {result.reasons}"
        assert any("extension" in reason.lower() for reason in result.reasons), \
            "Missing extension validation reason"

class TestEmailValidation(BaseValidationTest):
    """Test validation rules for email addresses."""
    
    def test_email_validation(self, setup_validation):
        """Test email validation with different cases."""
        validator = setup_validation
        test_cases = self._load_test_cases("email_validation.json")["email_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="EMAIL_ADDRESS",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio"
            )

            initial_confidence = entity.confidence
            result = validator.validate_entity(entity, case.get("context", ""))
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )

    def test_email_context_validation(self, setup_validation):
        """Test email validation with different contexts."""
        validator = setup_validation
        
        # Test with contact context
        entity = Entity(
            text="user@example.com",
            entity_type="EMAIL_ADDRESS",
            confidence=0.8,
            start=0,
            end=16,
            source="presidio"
        )
        result = validator.validate_entity(entity, "Contact: user@example.com")
        assert result.is_valid, f"Failed to validate email with contact context: {result.reasons}"
        assert any("context" in reason.lower() for reason in result.reasons), \
            "Missing context validation reason"
        
        # Test with known domain
        entity = Entity(
            text="user@gmail.com",
            entity_type="EMAIL_ADDRESS",
            confidence=0.8,
            start=0,
            end=14,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate email with known domain: {result.reasons}"
        assert any("known domain" in reason.lower() for reason in result.reasons), \
            "Missing known domain validation reason"

class TestInternetReferenceValidation(BaseValidationTest):
    """Test validation rules for internet references."""
    
    def test_internet_reference_validation(self, setup_validation):
        """Test internet reference validation with different cases."""
        validator = setup_validation
        test_cases = self._load_test_cases("internet_reference_validation.json")["internet_reference_validation_tests"]
        
        for case in test_cases:
            entity = Entity(
                text=case["text"],
                entity_type="INTERNET_REFERENCE",
                confidence=0.8,
                start=0,
                end=len(case["text"]),
                source="presidio"
            )

            initial_confidence = entity.confidence
            result = validator.validate_entity(entity, case.get("context", ""))
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )

    def test_internet_reference_formats(self, setup_validation):
        """Test validation of different internet reference formats."""
        validator = setup_validation
        
        # Test URL format
        entity = Entity(
            text="https://example.com",
            entity_type="INTERNET_REFERENCE",
            confidence=0.8,
            start=0,
            end=19,
            source="presidio"
        )
        result = validator.validate_entity(entity, "")
        assert result.is_valid, f"Failed to validate URL format: {result.reasons}"
        
        # Test social media handle
        entity = Entity(
            text="@username",
            entity_type="INTERNET_REFERENCE",
            confidence=0.8,
            start=0,
            end=9,
            source="presidio"
        )
        result = validator.validate_entity(entity, "Follow me on Twitter: @username")
        assert result.is_valid, f"Failed to validate social media handle: {result.reasons}"
        assert any("social" in reason.lower() for reason in result.reasons), \
            "Missing social media validation reason"

###############################################################################
# SECTION 10: PROTECTED CLASS AND PHI VALIDATION TESTS
###############################################################################

class TestProtectedClassValidation(BaseValidationTest):
    """Test validation rules for protected class entities."""
    
    def test_protected_class_validation(self, setup_validation):
        """Test protected class validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, case.get("context", ""))
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )

    def test_protected_class_contexts(self, setup_validation):
        """Test protected class validation with different contexts."""
        validator = setup_validation
        
        # Test with identity context
        entity = Entity(
            text="Muslim",
            entity_type="PROTECTED_CLASS",
            confidence=0.8,
            start=0,
            end=6,
            source="presidio"
        )
        result = validator.validate_entity(entity, "Religious identity: Muslim")
        assert result.is_valid, f"Failed to validate protected class with identity context: {result.reasons}"
        assert any("identity" in reason.lower() for reason in result.reasons), \
            "Missing identity context validation reason"
        
        # Test with organizational context
        entity = Entity(
            text="Catholic",
            entity_type="PROTECTED_CLASS",
            confidence=0.8,
            start=0,
            end=8,
            source="presidio"
        )
        result = validator.validate_entity(entity, "Catholic Church organization")
        assert result.is_valid, f"Failed to validate protected class with org context: {result.reasons}"
        assert any("organization" in reason.lower() for reason in result.reasons), \
            "Missing organizational context validation reason"

class TestPHIValidation(BaseValidationTest):
    """Test validation rules for PHI entities."""
    
    def test_phi_validation(self, setup_validation):
        """Test PHI validation with different cases."""
        validator = setup_validation
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
            result = validator.validate_entity(entity, case.get("context", ""))
            
            confidence_change = round(entity.confidence - initial_confidence, 2) if result.is_valid else 0.0

            assert result.is_valid == case["expected_valid"], (
                f"Failed: {case['description']} - Expected {case['expected_valid']}, "
                f"got {result.is_valid}. "
                f"Confidence: {round(entity.confidence, 2)}, "
                f"Adjustments: {confidence_change}, "
                f"Reasons: {result.reasons}"
            )

    def test_phi_contexts(self, setup_validation):
        """Test PHI validation with different medical contexts."""
        validator = setup_validation
        
        # Test with medical condition
        entity = Entity(
            text="Type 2 Diabetes",
            entity_type="PHI",
            confidence=0.8,
            start=0,
            end=15,
            source="presidio"
        )
        result = validator.validate_entity(entity, "Medical condition: Type 2 Diabetes")
        assert result.is_valid, f"Failed to validate PHI with medical condition: {result.reasons}"
        assert any("condition" in reason.lower() for reason in result.reasons), \
            "Missing medical condition validation reason"
        
        # Test with disability context
        entity = Entity(
            text="30% disability rating",
            entity_type="PHI",
            confidence=0.8,
            start=0,
            end=20,
            source="presidio"
        )
        result = validator.validate_entity(entity, "VA disability rating: 30%")
        assert result.is_valid, f"Failed to validate PHI with disability context: {result.reasons}"
        assert any("disability" in reason.lower() for reason in result.reasons), \
            "Missing disability context validation reason"

