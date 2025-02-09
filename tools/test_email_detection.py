# tools/test_email_detection.py

import logging
from pathlib import Path
from app.utils.config_loader import ConfigLoader
from redactor.detectors.presidio_detector import PresidioDetector
from redactor.validation import ValidationCoordinator

def setup_logger():
    logger = logging.getLogger("test_detector")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger

def test_email_detection():
    """Test email detection with the simplified Presidio detector."""
    logger = setup_logger()
    config_loader = ConfigLoader()
    config_loader.load_all_configs()

    # Initialize detector
    detector = PresidioDetector(
        config_loader=config_loader,
        logger=logger
    )

    # Test cases
    test_cases = [
        "Contact me at test.user@example.com for information",
        "Send your resume to careers@company.org",
        "My email is invalid@email",  # Invalid
        "Email me: alexis.rivera.tech@example.com",
        "user.name+filter@company.co.uk",
    ]

    print("\nTesting Email Detection")
    print("=" * 50)

    for i, text in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {text}")
        
        # Get detector results
        detected = detector.detect_entities(text)
        emails = [e for e in detected if e.entity_type == "EMAIL_ADDRESS"]
        
        if emails:
            print("\nDetector Results:")
            for email in emails:
                print(f"- Found: {email.text}")
                print(f"  Confidence: {email.confidence:.2f}")
                print(f"  Position: {email.start}-{email.end}")

                # Now validate using coordinator
                result = detector.validator.validate_entity(email, text)
                print("\nValidation Results:")
                print(f"  Valid: {result.is_valid}")
                print(f"  Confidence Adjustment: {result.confidence_adjustment}")
                if result.reasons:
                    print("  Reasons:")
                    for reason in result.reasons:
                        print(f"    - {reason}")
        else:
            print("No emails detected")

if __name__ == "__main__":
    test_email_detection()