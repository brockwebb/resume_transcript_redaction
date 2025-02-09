# tools/presidio_debug.py

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import EmailRecognizer
import fitz
from pathlib import Path

def debug_presidio_email(test_id: str):
    print("\nPresidio Email Detection Debug")
    print("=" * 50)
    
    # Load the PDF
    pdf_path = Path("data/test_suite/originals") / f"{test_id}.pdf"
    print(f"\nLoading document: {pdf_path}")
    
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        print(f"Document loaded, length: {len(text)} chars")
        
        # Initialize Presidio
        analyzer = AnalyzerEngine()
        
        # First test: Full document
        print("\nAnalyzing full document...")
        results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS"]
        )
        
        print(f"Found {len(results)} email addresses in full document:")
        for result in results:
            print(f"\nEmail: {text[result.start:result.end]}")
            print(f"Score: {result.score}")
            print(f"Position: {result.start}-{result.end}")
            print(f"Context: ...{text[max(0, result.start-20):result.start]}")
            print(f"{text[result.start:result.end]}")
            print(f"{text[result.end:min(len(text), result.end+20)]}...")
            
        # Second test: Analyze each line separately
        print("\nAnalyzing line by line...")
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for line in lines:
            if '@' in line:  # Only check lines that might contain emails
                results = analyzer.analyze(
                    text=line,
                    language="en",
                    entities=["EMAIL_ADDRESS"]
                )
                if results:
                    print(f"\nLine: {line}")
                    for result in results:
                        print(f"Found: {line[result.start:result.end]}")
                        print(f"Score: {result.score}")
                
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python tools/presidio_debug.py <test_id>")
        sys.exit(1)
        
    test_id = sys.argv[1]
    debug_presidio_email(test_id)