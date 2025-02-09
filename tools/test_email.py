# tools/test_email.py

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import EmailRecognizer
import spacy
import sys
from pathlib import Path
import fitz  # PyMuPDF

def load_document(pdf_path):
    """Load text from PDF"""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

def test_email_detection(document_path=None):
    # Test standalone email
    email = "alexis.rivera.tech@example.com"
    
    # Test with Presidio
    print("\n1. Presidio Standalone Email Detection:")
    analyzer = AnalyzerEngine()
    results = analyzer.analyze(
        text=email,
        language="en",
        entities=["EMAIL_ADDRESS"]
    )
    
    for result in results:
        print(f"Text: {email[result.start:result.end]}")
        print(f"Score: {result.score}")
        print(f"Position: {result.start}-{result.end}")
    
    # Test with spaCy
    print("\n2. spaCy Standalone Email Detection:")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(email)
    
    for ent in doc.ents:
        print(f"Text: {ent.text}")
        print(f"Label: {ent.label_}")
        print(f"Position: {ent.start_char}-{ent.end_char}")
    
    # Test in document context if path provided
    if document_path:
        print(f"\n3. Document Email Detection: {document_path}")
        text = load_document(document_path)
        
        print("\nPresidio Document Detection:")
        doc_results = analyzer.analyze(
            text=text,
            language="en",
            entities=["EMAIL_ADDRESS"]
        )
        
        for result in doc_results:
            email_text = text[result.start:result.end]
            print(f"Email: {email_text}")
            print(f"Score: {result.score}")
            print(f"Position: {result.start}-{result.end}")
        
        print("\nspaCy Document Detection:")
        doc = nlp(text)
        
        for ent in doc.ents:
            print(f"Email: {ent.text}")
            print(f"Label: {ent.label_}")
            print(f"Position: {ent.start_char}-{ent.end_char}")

def main():
    if len(sys.argv) > 1:
        # If a PDF path is provided
        pdf_path = Path(sys.argv[1])
        if pdf_path.exists():
            test_email_detection(str(pdf_path))
        else:
            print(f"Error: File not found {pdf_path}")
    else:
        # Default standalone testing
        test_email_detection()

if __name__ == "__main__":
    main()