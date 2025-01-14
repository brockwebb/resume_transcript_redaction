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

    # Add the GeneralMedicalDisclosureRecognizer
    analyzer.registry.add_recognizer(GeneralMedicalDisclosureRecognizer())

    return analyzer


class GeneralMedicalDisclosureRecognizer(PatternRecognizer):
    """Custom recognizer for generalized medical disclosures."""
    def __init__(self):
        patterns = [
            Pattern(
                name="medical_disclosure_general",
                regex=r"\b(medical condition|disability|health issue|requires accommodation|chronic illness|diagnosed with)\b",
                score=0.9
            )
        ]
        context = [
            "medical", "health", "condition", "disability", "chronic", "diagnosed", "accommodation", "treatment"
        ]
        super().__init__(
            supported_entity="MEDICAL_DISCLOSURE",
            patterns=patterns,
            context=context
        )


class RedactionProcessor:
    def __init__(self, custom_patterns, keep_words_path=None, redact_color=(0, 0, 0), redact_opacity=1.0, pii_types=None, confidence_threshold=0.75, model_directory=None):
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
                print(f"Loaded {len(self.keep_words)} keep words and {len(self.discard_words)} discard words.")
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

    def _validate_with_jobbert(self, texts, reference_phrases=None):
        """Validate text using JobBERT and optional semantic similarity."""
        if not texts:
            return []

        # Reference medical disclosure phrases for validation
        reference_phrases = reference_phrases or [
            "medical condition",
            "chronic illness",
            "requires accommodation",
            "health issue",
            "diagnosed with a disability"
        ]

        inputs = self.tokenizer(texts + reference_phrases, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Separate embeddings for texts and reference phrases
        text_embeddings = outputs.last_hidden_state[:len(texts)].mean(dim=1)
        reference_embeddings = outputs.last_hidden_state[len(texts):].mean(dim=1)

        # Compute cosine similarity
        cosine_similarities = torch.nn.functional.cosine_similarity(
            text_embeddings.unsqueeze(1), reference_embeddings.unsqueeze(0), dim=2
        )
        max_similarities = torch.max(cosine_similarities, dim=1).values

        return max_similarities.tolist()

    def _detect_redactions(self, wordlist):
        """Detect sensitive information."""
        redact_areas = []
        batch_texts = []
        batch_coords = []

        # Combine words into phrases for pattern matching
        combined_text = " ".join(word["text"] if isinstance(word, dict) else word[4] for word in wordlist)
        combined_results = self.analyzer.analyze(text=combined_text, language="en")

        for word in wordlist:
            text = word["text"] if isinstance(word, dict) else word[4]
            bbox = word["bbox"] if isinstance(word, dict) else fitz.Rect(word[0], word[1], word[2], word[3])

            # Skip keep words
            if text.lower() in self.keep_words:
                continue

            # Automatically redact discard words
            if text.lower() in self.discard_words:
                redact_areas.append(bbox)
                continue

            # Check combined results for PII
            for res in combined_results:
                if res.entity_type in self.pii_types and res.score >= self.confidence_threshold:
                    if text in combined_text[res.start:res.end]:
                        batch_texts.append(text)
                        batch_coords.append(bbox)

            # Check individual word results from Presidio
            results = self.analyzer.analyze(text=text, language="en")
            if any(res.entity_type in self.pii_types and res.score >= self.confidence_threshold for res in results):
                batch_texts.append(text)
                batch_coords.append(bbox)

            # Check custom terms
            if any(term.lower() in text.lower() for term in self.custom_terms):
                redact_areas.append(bbox)

        # Validate flagged text with JobBERT
        if batch_texts:
            confidence_scores = self._validate_with_jobbert(batch_texts)
            for bbox, confidence in zip(batch_coords, confidence_scores):
                if confidence > self.confidence_threshold:
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
