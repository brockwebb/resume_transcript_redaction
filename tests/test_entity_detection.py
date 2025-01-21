# tests/test_entity_detection.py
"""
Integration Tests for Resume Redaction Entity Detection System

Purpose:
    Verify core functionality of the entity detection system by testing:
    - Presidio initialization and detection
    - Confidence thresholds
    - Proper routing and validation
    - End-to-end functionality of the detection pipeline

Usage:
    Run all tests:
        python -m tests.test_entity_detection

    Run specific test numbers:
        python -m tests.test_entity_detection --tests 1,3
        (Test numbers start from 0)

    Test specific detector:
        python -m tests.test_entity_detection --detector presidio
        python -m tests.test_entity_detection --detector spacy
        
    Test specific entity type:
        python -m tests.test_entity_detection --entity phone
        python -m tests.test_entity_detection --entity email
        python -m tests.test_entity_detection --entity mixed

    Combine options:
        python -m tests.test_entity_detection --detector presidio --entity phone

Available Tests:
    0: test_spacy_detector_for_location
    1: test_presidio_initialization
    2: test_presidio_email_detection
    3: test_presidio_phone_detection
    4: test_presidio_confidence_thresholds
    5: test_presidio_mixed_content

Entity Types:
    - email
    - phone
    - mixed (multiple entity types)
    - location
"""

import sys
import pytest
from pathlib import Path
import logging
import argparse

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from redactor.detectors.presidio_detector import PresidioDetector
from redactor.detectors.spacy_detector import SpacyDetector

# Test data structures
TEST_TEXTS = {
    "basic": """
    Contact Information:
    Name: John A. Smith
    Email: john.smith@example.com
    Phone: (555) 123-4567
    """,
    "ssn_variations": """
    Personnel Information:
    SSN: 123-45-6789
    Social Security: 234-56-7890
    Social Security Number: 345-67-8901
    SSN Number: 456789012
    Employee SSN: 567-89-0123
    Tax ID / SSN: 678901234
    """,
    "financial": """
    Financial Details:
    Credit Card: 4111-1111-1111-1111
    Bank Account: 12345678
    """,
    "location": """
    Location:
    Address: 123 Main St, Springfield, IL
    IP Address: 192.168.1.1
    """
}

PRESIDIO_TEST_TEXTS = {
    "email_variants": """
    Contact Details:
    Primary: test.user@company.com
    Secondary: test_user+label@sub.domain.co.uk
    Support: support@domain.io
    """,
    "phone_variants": """
    Phone Numbers:
    Office: (555) 123-4567
    Mobile: +1-555-987-6543
    Support: 1.555.234.5678
    International: +44 20 7123 4567
    Extension: (555) 123-4567 ext. 890
    """,
    "mixed_entities": """
    Professional Profile:
    Dr. Jane Smith, PhD
    Research Director
    Email: j.smith@research.org
    Phone: (555) 123-4567
    SSN: 123-45-6789
    Location: 123 Science Park, Cambridge, MA
    """
}

def setup_logging():
    """Setup test logging."""
    logger = RedactionLogger(
        name="test_detection",
        log_level="DEBUG",
        log_dir=str(project_root / "logs"),
        console_output=True
    )
    return logger

@pytest.mark.detector_test
@pytest.mark.spacy
@pytest.mark.entity_type_location
def test_spacy_detector_for_location():
    """Test SpaCy detector specifically for location detection."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = SpacyDetector(config_loader, logger)
    entities = detector.detect_entities(TEST_TEXTS["location"])

    assert len(entities) > 0, "No entities detected in location information test"

    addresses = [e for e in entities if e.entity_type == "LOCATION"]
    logger.info(f"Found {len(addresses)} LOCATION entities")
    
    assert len(addresses) >= 1, "Failed to detect address"
    assert any("123 Main St" in addr.text for addr in addresses), "Incorrect address detection"

@pytest.mark.detector_test
@pytest.mark.presidio
def test_presidio_initialization():
    """Test Presidio detector initialization and configuration."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = PresidioDetector(config_loader, logger)
    
    # Verify configuration loaded properly
    assert detector.analyzer is not None, "Analyzer not initialized"
    assert len(detector.entity_types) > 0, "No entity types configured"
    assert all(0 < threshold <= 1.0 for threshold in detector.entity_types.values()), \
        "Invalid confidence thresholds"

@pytest.mark.detector_test
@pytest.mark.presidio
@pytest.mark.entity_type_email
def test_presidio_email_detection():
    """Test email address detection with various formats."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = PresidioDetector(config_loader, logger)
    entities = detector.detect_entities(PRESIDIO_TEST_TEXTS["email_variants"])

    emails = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
    logger.info(f"Found {len(emails)} email addresses")

    assert len(emails) >= 3, "Failed to detect all email addresses"
    assert any("test.user@company.com" in e.text for e in emails), "Failed to detect basic email"
    assert any("test_user+label@sub.domain.co.uk" in e.text for e in emails), \
        "Failed to detect complex email"

@pytest.mark.detector_test
@pytest.mark.presidio
@pytest.mark.entity_type_phone
def test_presidio_phone_detection():
    """Test phone number detection with various formats."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = PresidioDetector(config_loader, logger)
    entities = detector.detect_entities(PRESIDIO_TEST_TEXTS["phone_variants"])

    phones = [e for e in entities if e.entity_type == "PHONE_NUMBER"]
    logger.info(f"Found {len(phones)} phone numbers")

    assert len(phones) >= 4, "Failed to detect all phone numbers"
    assert any("(555) 123-4567" in e.text for e in phones), "Failed to detect US format"
    assert any("+44 20 7123 4567" in e.text for e in phones), "Failed to detect international format"

@pytest.mark.detector_test
@pytest.mark.presidio
def test_presidio_confidence_thresholds():
    """Test confidence threshold handling."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = PresidioDetector(config_loader, logger)
    
    # Test with clear matches (should have high confidence)
    text = "Email: test@example.com\nPhone: (555) 123-4567"
    entities = detector.detect_entities(text)
    
    for entity in entities:
        assert entity.confidence >= detector.entity_types.get(
            entity.entity_type, detector.confidence_threshold
        ), f"Entity {entity.text} below confidence threshold"

@pytest.mark.detector_test
@pytest.mark.presidio
@pytest.mark.entity_type_mixed
def test_presidio_mixed_content():
    """Test detection in mixed content with multiple entity types."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    detector = PresidioDetector(config_loader, logger)
    entities = detector.detect_entities(PRESIDIO_TEST_TEXTS["mixed_entities"])

    # Group entities by type
    entity_groups = {}
    for entity in entities:
        entity_groups.setdefault(entity.entity_type, []).append(entity)

    # Log detection results
    for entity_type, group in entity_groups.items():
        logger.info(f"Found {len(group)} {entity_type} entities")
        for entity in group:
            logger.info(f"  {entity.text} (confidence: {entity.confidence:.2f})")

    # Verify multiple entity types detected
    assert len(entity_groups) >= 3, "Failed to detect multiple entity types"
    assert any(e.entity_type == "EMAIL_ADDRESS" for e in entities), "No email detected"
    assert any(e.entity_type == "PHONE_NUMBER" for e in entities), "No phone number detected"
    assert any(e.entity_type == "US_SSN" for e in entities), "No SSN detected"

def parse_args():
    """Parse command line arguments for test configuration."""
    parser = argparse.ArgumentParser(
        description='Run entity detection tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--tests', type=str, 
                       help='Comma-separated list of test numbers to run (e.g., 1,3)')
    parser.add_argument('--detector', choices=['spacy', 'presidio', 'all'], 
                       default='all', help='Which detector to test')
    parser.add_argument('--entity', type=str, 
                       help='Test specific entity type (email, phone, mixed, location)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Build pytest args based on command line options
    pytest_args = ['-v']  # Always use verbose output
    
    # Handle test numbers
    if args.tests:
        test_nums = [int(n.strip()) for n in args.tests.split(',')]
        test_funcs = []
        all_tests = [obj for obj in globals().values() 
                    if callable(obj) and obj.__name__.startswith('test_')]
        for num in test_nums:
            if 0 <= num < len(all_tests):
                test_funcs.append(all_tests[num].__name__)
        if test_funcs:
            pytest_args.extend(['-k', ' or '.join(test_funcs)])
    
    # Handle detector selection
    if args.detector != 'all':
        pytest_args.extend(['-m', args.detector])
    
    # Handle entity type selection
    if args.entity:
        # Use -k option instead of -m for entity type
        pytest_args.extend(['-k', f'entity_type_{args.entity}'])
        
    pytest.main(pytest_args)