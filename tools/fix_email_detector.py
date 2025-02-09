# tools/fix_email_detector.py

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import EmailRecognizer
import re
from typing import List, Dict, Any
import logging

class EnhancedEmailRecognizer(EmailRecognizer):
    """Enhanced email recognizer with custom validation rules"""
    
    def __init__(
        self,
        supported_language: str = "en",
        supported_entity: str = "EMAIL_ADDRESS",
        context: List[str] = None,
        validation_rules: Dict[str, Any] = None
    ):
        super().__init__(supported_language=supported_language)
        self.validation_rules = validation_rules or {}
        
        # Add additional context beyond base Presidio
        extra_context = [
            "email", "e-mail", "mail", "contact", 
            "reach me", "reach out", "contact me at",
            "send", "write", "@"
        ]
        if context:
            extra_context.extend(context)
        self.context = extra_context

    def validate_result(self, pattern_text: str) -> bool:
        """Enhanced validation using our custom rules"""
        # First apply base Presidio validation
        if not super().validate_result(pattern_text):
            return False
            
        # Apply our custom validation rules
        try:
            # Length checks
            if len(pattern_text) < self.validation_rules.get("min_chars", 5):
                return False
            if len(pattern_text) > self.validation_rules.get("max_length", 254):
                return False
                
            # Split into local and domain parts
            if '@' not in pattern_text:
                return False
            local_part, domain = pattern_text.rsplit('@', 1)
            
            # Local part validation
            if not local_part or local_part[-1] == '.':
                return False
            if '..' in local_part:
                return False
                
            # Domain validation
            if not domain or '.' not in domain:
                return False
            if domain.endswith('.'):
                return False
                
            return True
            
        except Exception as e:
            logging.warning(f"Error in enhanced email validation: {str(e)}")
            return False

def setup_enhanced_email_detection(analyzer: AnalyzerEngine, validation_rules: Dict) -> None:
    """Setup enhanced email detection with custom rules"""
    
    # Remove existing email recognizer if present
    try:
        analyzer.registry.remove_recognizer("EmailRecognizer")
    except Exception:
        pass
        
    # Create and register enhanced recognizer
    enhanced_recognizer = EnhancedEmailRecognizer(
        validation_rules=validation_rules
    )
    analyzer.registry.add_recognizer(enhanced_recognizer)
    
    # Verify registration
    registered = [r.__class__.__name__ for r in analyzer.registry.recognizers]
    if "EnhancedEmailRecognizer" not in registered:
        raise RuntimeError("Failed to register enhanced email recognizer")

def test_enhanced_detection():
    """Test the enhanced email detection"""
    print("\nTesting Enhanced Email Detection")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AnalyzerEngine()
    
    # Test validation rules
    validation_rules = {
        "min_chars": 5,
        "max_length": 254,
        "require_at_symbol": True,
        "require_domain": True
    }
    
    # Setup enhanced detection
    setup_enhanced_email_detection(analyzer, validation_rules)
    
    # Test cases
    test_cases = [
        {
            "text": "Contact me at test.user@example.com for more information",
            "should_detect": True
        },
        {
            "text": "My email is user.name+filter@company.co.uk",
            "should_detect": True
        },
        {
            "text": "Send resumes to careers@company.org",
            "should_detect": True
        },
        {
            "text": "Invalid@email",  # Missing domain
            "should_detect": False
        },
        {
            "text": "a@b.c",  # Too short
            "should_detect": False
        }
    ]
    
    # Run tests
    for case in test_cases:
        text = case["text"]
        expected = case["should_detect"]
        
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS"]
        )
        
        detected = len(results) > 0
        status = "✅" if detected == expected else "❌"
        
        print(f"\nTest: {text}")
        print(f"Expected detection: {expected}")
        print(f"Actual detection: {detected} {status}")
        
        if results:
            for result in results:
                print(f"Score: {result.score}")
                print(f"Position: {result.start}-{result.end}")

if __name__ == "__main__":
    test_enhanced_detection()