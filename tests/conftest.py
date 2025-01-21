# tests/conftest.py

import pytest

def pytest_configure(config):
    """Register custom markers."""
    # Core test types
    config.addinivalue_line(
        "markers",
        "detector_test: mark test as a detector test"
    )
    config.addinivalue_line(
        "markers",
        "presidio: mark test as testing Presidio detector"
    )
    config.addinivalue_line(
        "markers",
        "spacy: mark test as testing spaCy detector"
    )
    
    # Entity type markers
    config.addinivalue_line(
        "markers",
        "entity_type_phone: tests phone number detection"
    )
    config.addinivalue_line(
        "markers",
        "entity_type_email: tests email detection"
    )
    config.addinivalue_line(
        "markers",
        "entity_type_mixed: tests mixed entity detection"
    )
    config.addinivalue_line(
        "markers",
        "entity_type_location: tests location detection"
    )