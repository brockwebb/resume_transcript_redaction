# tests/test_entity_detection.py
"""
Integration Tests for Resume Redaction Entity Detection System

Purpose:
    Verify core functionality of the entity detection system by testing:
    - Configuration loading and validation
    - Individual detector initialization and basic operation
    - Ensemble coordination and result combination
    - End-to-end processing pipeline

These tests focus on system integration rather than detection accuracy.
They verify that:
    - Components initialize correctly
    - Configuration files are loaded and interpreted
    - Data flows properly between components
    - Basic entity detection works for known test cases
    - Results are properly formatted and combined

Note: These tests use simple, hardcoded test cases. They are designed
to verify system functionality, not to evaluate detection accuracy
or performance. Separate evaluation scripts should be used for 
tuning and accuracy assessment.

Usage:
    Can be run directly: python -m tests.test_entity_detection
    Or via pytest: pytest tests/test_entity_detection.py
"""
import sys
from pathlib import Path
import pytest
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from redactor.detectors.presidio_detector import PresidioDetector
from redactor.detectors.spacy_detector import SpacyDetector
from redactor.detectors.ensemble_coordinator import EnsembleCoordinator

# Test data with known entities
TEST_TEXTS = {
    "basic": """
    Contact Information:
    Name: John A. Smith
    Email: john.smith@example.com
    Phone: (555) 123-4567
    """,
    "technical": """
    Technical Skills:
    - Advanced Python programming
    - Tableau dashboard development
    - SQL database management
    """,
    "mixed": """
    Dr. Sarah Johnson, PhD
    Data Science Lead
    TechCorp Industries
    Email: sarah.j@techcorp.com
    Skills: Python, R, Tableau
    Previously employed at MIT and Google
    """,
    "overlap": """
    Contact: James Smith 
    Email: james.smith@company.com
    Organization: Smith Technologies Inc.
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

def test_config_loading():
    """Test configuration loading."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    
    # Test main config loading
    assert config_loader.load_all_configs(), "Failed to load configurations"
    
    # Verify essential configs are present
    assert config_loader.get_config("entity_routing"), "Missing entity routing config"
    assert config_loader.get_config("detection"), "Missing detection patterns config"
    
    logger.info("Configuration loading test passed")

def test_presidio_detector():
    """Test Presidio detector initialization and basic detection."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    
    detector = PresidioDetector(config_loader, logger)
    
    # Test email detection
    logger.debug(f"Testing text: {TEST_TEXTS['basic']}")
    entities = detector.detect_entities(TEST_TEXTS["basic"])
    logger.debug(f"Detected entities: {entities}")
    
    emails = [e for e in entities if e.entity_type == "EMAIL_ADDRESS"]
    logger.debug(f"Found emails: {emails}")
    assert len(emails) > 0, "Failed to detect email"
    assert "john.smith@example.com" in emails[0].text, "Incorrect email detection"
    
    # Test technical term handling
    tech_entities = detector.detect_entities(TEST_TEXTS["technical"])
    logger.debug(f"Technical entities: {tech_entities}")
    assert not any(e.text.lower() == "python" for e in tech_entities), "Technical term not preserved"
    
    logger.info("Presidio detector test passed")

def test_spacy_detector():
    """Test spaCy detector initialization and basic detection."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    
    detector = SpacyDetector(config_loader, logger)
    
    # Test person detection
    entities = detector.detect_entities(TEST_TEXTS["basic"])
    persons = [e for e in entities if e.entity_type == "PERSON"]
    assert len(persons) > 0, "Failed to detect person"
    assert persons[0].text == "John Smith", "Incorrect person detection"
    
    # Test organization detection
    org_entities = detector.detect_entities(TEST_TEXTS["mixed"])
    orgs = [e for e in entities if e.entity_type == "ORG"]
    assert len(orgs) > 0, "Failed to detect organizations"
    
    logger.info("spaCy detector test passed")

def test_ensemble_coordination():
    """Test ensemble coordination and result combination."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    
    coordinator = EnsembleCoordinator(config_loader, logger)
    
    # Test mixed content detection
    entities = coordinator.detect_entities(TEST_TEXTS["mixed"])
    
    # Verify we got results from both detectors
    sources = {e.source for e in entities}
    assert "presidio" in sources, "No Presidio detections"
    assert "spacy" in sources, "No spaCy detections"
    
    # Test overlap handling
    overlap_entities = coordinator.detect_entities(TEST_TEXTS["overlap"])
    
    # Verify no overlapping detections
    positions = [(e.start, e.end) for e in overlap_entities]
    for i, pos1 in enumerate(positions):
        for pos2 in positions[i+1:]:
            assert not (pos1[0] <= pos2[1] and pos1[1] >= pos2[0]), "Overlapping entities found"
    
    logger.info("Ensemble coordination test passed")

def test_end_to_end():
    """Test end-to-end entity detection pipeline."""
    logger = setup_logging()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    
    coordinator = EnsembleCoordinator(config_loader, logger)
    
    # Process all test texts
    for name, text in TEST_TEXTS.items():
        logger.info(f"Processing test text: {name}")
        entities = coordinator.detect_entities(text)
        
        # Basic validation
        assert isinstance(entities, list), f"Invalid result type for {name}"
        if entities:
            for entity in entities:
                assert entity.text, f"Empty entity text in {name}"
                assert 0 <= entity.confidence <= 1, f"Invalid confidence in {name}"
                assert entity.start < entity.end, f"Invalid positions in {name}"
        
        logger.info(f"Found {len(entities)} entities in {name}")
        
    logger.info("End-to-end test passed")

if __name__ == "__main__":
    # Setup logging
    logger = setup_logging()
    logger.info("Starting integration tests")
    
    try:
        # Run tests
        test_config_loading()
        test_presidio_detector()
        test_spacy_detector()
        test_ensemble_coordination()
        test_end_to_end()
        
        logger.info("All tests passed successfully")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise