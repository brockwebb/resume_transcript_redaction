# tools/email_presidio_debug.py
from presidio_analyzer import AnalyzerEngine
import fitz  # PyMuPDF
import sys

def load_document(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def debug_email_detection(pdf_path):
    # Load document
    text = load_document(pdf_path)
    
    # Presidio analysis
    analyzer = AnalyzerEngine()
    
    # Analyze entire document
    results = analyzer.analyze(
        text=text,
        language='en',
        entities=['EMAIL_ADDRESS']
    )
    
    print("\nPresidio Email Detection Results:")
    for result in results:
        email = text[result.start:result.end]
        print(f"Email: {email}")
        print(f"Score: {result.score}")
        print(f"Position: {result.start}-{result.end}")
        
        # Context around the email
        start = max(0, result.start - 50)
        end = min(len(text), result.end + 50)
        print(f"\nContext:\n{text[start:end]}\n")
        print("-" * 50)

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m tools.email_presidio_debug <pdf_path>")
        sys.exit(1)
    
    debug_email_detection(sys.argv[1])

if __name__ == "__main__":
    main()