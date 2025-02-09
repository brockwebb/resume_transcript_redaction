# tools/test_date_detection.py

import logging
from pathlib import Path
from app.utils.config_loader import ConfigLoader
from redactor.detectors.presidio_detector import PresidioDetector
from redactor.detectors.spacy_detector import SpacyDetector
from redactor.validation import ValidationCoordinator

def setup_logger():
    logger = logging.getLogger("test_date_detector")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

def test_date_detection():
    """Test date detection using Presidio and spaCy detectors."""
    logger = setup_logger()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    # Initialize detectors
    presidio_detector = PresidioDetector(
        config_loader=config_loader,
        logger=logger
    )
    spacy_detector = SpacyDetector(
        config_loader=config_loader,
        logger=logger
    )

    # Test cases covering different date formats
    test_cases = [
        "Graduated in May 2019 from university",
        "Started working in July 2016",
        "Class of 2018",
        "Employment period: June 2018 - Present",
        "Internship from January 2020 to December 2020"
    ]

    print("\nTesting Date Detection")
    print("=" * 50)

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {text}")
        
        # Presidio detection
        print("\nPresidio Detection:")
        presidio_dates = [e for e in presidio_detector.detect_entities(text) if e.entity_type == "DATE_TIME"]
        for date in presidio_dates:
            print(f"- Found: {date.text}")
            print(f"  Confidence: {date.confidence:.2f}")
            print(f"  Position: {date.start}-{date.end}")

            # Validate using coordinator
            result = presidio_detector.validator.validate_entity(date, text)
            print("\nPresidio Validation Results:")
            print(f"  Valid: {result.is_valid}")
            print(f"  Confidence Adjustment: {result.confidence_adjustment}")
            if result.reasons:
                print("  Reasons:")
                for reason in result.reasons:
                    print(f"    - {reason}")

        # SpaCy detection
        print("\nSpaCy Detection:")
        spacy_dates = [e for e in spacy_detector.detect_entities(text) if e.entity_type == "DATE_TIME"]
        for date in spacy_dates:
            print(f"- Found: {date.text}")
            print(f"  Confidence: {date.confidence:.2f}")
            print(f"  Position: {date.start}-{date.end}")

            # Validate using coordinator
            result = spacy_detector.validator.validate_entity(date, text)
            print("\nSpaCy Validation Results:")
            print(f"  Valid: {result.is_valid}")
            print(f"  Confidence Adjustment: {result.confidence_adjustment}")
            if result.reasons:
                print("  Reasons:")
                for reason in result.reasons:
                    print(f"    - {reason}")

if __name__ == "__main__":
    test_date_detection()