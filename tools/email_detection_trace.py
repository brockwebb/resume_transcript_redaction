# tools/email_detection_trace.py
from evaluation.evaluate import StageOneEvaluator
from redactor.detectors.presidio_detector import PresidioDetector
import yaml
import re
from pathlib import Path
import traceback

def trace_email_detection(test_id):
    try:
        # Initialize evaluator and get configuration
        evaluator = StageOneEvaluator()
        config_loader = evaluator.config_loader
        
        # Load document
        text = evaluator._load_test_document(test_id)
        
        # Load ground truth
        annotation_path = Path(f"data/test_suite/annotations/{test_id}.yaml")
        with open(annotation_path, 'r') as f:
            ground_truth = yaml.safe_load(f)
        
        print("Ground Truth Emails:")
        ground_emails = [
            e for e in ground_truth.get('entities', []) 
            if e['type'] == 'EMAIL_ADDRESS'
        ]
        for email in ground_emails:
            print(f"- {email['text']} (pos: {email['location']['start_char']}-{email['location']['end_char']})")
        
        # Manual email regex detection
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        manual_emails = re.findall(email_pattern, text, re.IGNORECASE)
        
        print("\nManual Email Detection:")
        for email in manual_emails:
            print(f"Found email: {email}")
            start = text.index(email)
            
            print(f"\nDetailed Email Context:")
            # Show context around the email
            context_start = max(0, start - 50)
            context_end = min(len(text), start + len(email) + 50)
            print(text[context_start:context_end])
        
        # Initialize Presidio detector
        presidio_detector = PresidioDetector(
            config_loader=config_loader,
            logger=evaluator.logger
        )
        
        print("\nPresidio Detection Configuration:")
        print(f"Email validation rules: {config_loader.get_config('validation_params')['EMAIL_ADDRESS']}")
        
        print("\nDetection Process Details:")
        print("Analyzers:", [r.name for r in presidio_detector.analyzer.registry.recognizers])
        
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        print(traceback.format_exc())

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m tools.email_detection_trace <test_id>")
        sys.exit(1)
    
    trace_email_detection(sys.argv[1])

if __name__ == "__main__":
    main()