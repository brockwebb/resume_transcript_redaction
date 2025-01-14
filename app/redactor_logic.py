import re  # Regular expressions
import fitz  # PyMuPDF
from pytesseract import image_to_string
from pdf2image import convert_from_path
import torch
from transformers import AutoTokenizer, AutoModel
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
from typing import Dict, Any
import yaml


# Function to set up Presidio Analyzer with custom patterns
def setup_presidio_analyzer(custom_patterns):
    """Set up Presidio Analyzer with custom patterns."""
    analyzer = AnalyzerEngine()
    for pattern in custom_patterns:
        recognizer = PatternRecognizer(
            supported_entity=pattern["entity_type"],
            patterns=[Pattern(
                name=pattern["name"],
                regex=pattern["regex"],
                score=pattern["score"]
            )]
        )
        analyzer.registry.add_recognizer(recognizer)

    return analyzer


def _normalize_word(word, preserve_pattern=False):
    """Normalize a word by optionally preserving characters for pattern matching."""
    import string

    if preserve_pattern:
        # Allow specific characters (e.g., for phone numbers)
        allowed_chars = set("0123456789()+-.")
        normalized = "".join(c for c in word if c.isalnum() or c in allowed_chars).lower()
    else:
        # General normalization: Remove all punctuation and lowercase
        normalized = word.strip().lower().translate(str.maketrans("", "", string.punctuation))

    return normalized


class RedactionProcessor:
    def __init__(self, custom_patterns, keep_words_path=None, redact_color=(0, 0, 0), redact_opacity=1.0, 
                 pii_types=None, confidence_threshold=0.75, model_directory=None):
        # Set up Presidio Analyzer
        self.analyzer = setup_presidio_analyzer(custom_patterns)
        self.redact_color = redact_color
        self.redact_opacity = redact_opacity
        self.pii_types = pii_types or []
        self.confidence_threshold = confidence_threshold
        self.custom_terms = []
        self.keep_words = []
        self.discard_words = []
        self.validate_pdfs = False
        self.ocr_enabled = False

        # Load keep and discard words from YAML file if provided
        if keep_words_path:
            self.load_custom_word_filters(keep_words_path)

        # Load JobBERT model and tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = AutoTokenizer.from_pretrained(model_directory)
        self.model = AutoModel.from_pretrained(model_directory).to(self.device)

    def load_custom_word_filters(self, file_path):
        """Load custom word filters (keep and discard) from a YAML file."""
        try:
            with open(file_path, "r") as file:
                filters = yaml.safe_load(file).get("custom_word_filters", {})
                # Ensure all entries are strings
                self.keep_words = [str(word).lower() for word in filters.get("keep_words", []) if isinstance(word, str)]
                self.discard_words = [str(word).lower() for word in filters.get("discard_words", []) if isinstance(word, str)]
                print(f"Loaded {len(self.keep_words)} keep words and {len(self.discard_words)} discard words successfully.")
        except Exception as e:
            print(f"Error loading custom word filters: {e}")

    def add_custom_terms(self, terms):
        """Add custom terms to the list for redaction."""
        self.custom_terms.extend(terms)

    def validate_pdf(self, file_path):
        """Check if the file is a valid PDF."""
        try:
            with fitz.open(file_path) as _:
                return True
        except Exception:
            return False

    def apply_ocr(self, file_path):
        """Apply OCR to non-OCR PDFs."""
        images = convert_from_path(file_path)
        text = "\n".join(image_to_string(image) for image in images)
        return text

    def _validate_with_jobbert(self, texts):
        """Validate text using JobBERT."""
        if not texts:
            return []

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        confidence_scores = [torch.norm(emb).item() for emb in embeddings]

        return confidence_scores

    def _detect_redactions(self, wordlist):
        """Detect sensitive information."""
        redact_areas = []
    
        # Normalize keep and discard words
        normalized_keep_words = set(_normalize_word(word) for word in self.keep_words)
        normalized_discard_words = set(_normalize_word(word) for word in self.discard_words)
    
        for word in wordlist:
            text = word["text"] if isinstance(word, dict) else word[4]
            bbox = word["bbox"] if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])
    
            # Normalize for filtering logic
            normalized_text = _normalize_word(text)
    
            # Skip keep words
            if normalized_text in normalized_keep_words:
                continue
    
            # Automatically redact discard words
            if normalized_text in normalized_discard_words:
                redact_areas.append(bbox)
                continue
    
            # Use raw text for Presidio analysis
            results = self.analyzer.analyze(text=text, language="en")
            for res in results:
                # Check against dynamic entity types and confidence
                if res.entity_type in self.pii_types and res.score >= self.confidence_threshold:
                    redact_areas.append(bbox)
    
            # Check custom terms
            if any(term.lower() in text.lower() for term in self.custom_terms):
                redact_areas.append(bbox)
    
        return redact_areas


    def process_pdf(self, input_path: str, output_path: str, highlight_only=False) -> Dict[str, Any]:
        """Process a single PDF file."""
        if self.validate_pdfs and not self.validate_pdf(input_path):
            raise ValueError(f"Invalid PDF: {input_path}")

        doc = fitz.open(input_path)
        total_words = 0
        redacted_words = 0

        for page_index, page in enumerate(doc, start=1):
            wordlist = page.get_text("words")
            if self.ocr_enabled and not wordlist:
                # Apply OCR if the page has no text
                ocr_text = self.apply_ocr(input_path)
                wordlist = [{"bbox": None, "text": word} for word in ocr_text.split()]

            total_words += len(wordlist)
            redact_areas = self._detect_redactions(wordlist)
            redacted_words += len(redact_areas)

            for rect in redact_areas:
                if highlight_only:
                    rect_annot = page.add_rect_annot(rect)
                    rect_annot.set_colors(stroke=None, fill=(1, 0, 0))  # Red fill
                    rect_annot.set_opacity(self.redact_opacity)  # Semi-transparent
                    rect_annot.update()
                else:
                    try:
                        annot = page.add_redact_annot(rect)
                        annot.set_colors(fill=self.redact_color)
                        annot.update()
                    except Exception as e:
                        print(f"Error applying redaction annotation on Page {page_index}: {e}")

            if not highlight_only:
                try:
                    page.apply_redactions()
                except Exception as e:
                    print(f"Error applying redactions on Page {page_index}: {e}")

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return {"total_words": total_words, "redacted_words": redacted_words}
