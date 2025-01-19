import fitz
from typing import Dict, Any, List, Tuple
import os
from pytesseract import image_to_string
from pdf2image import convert_from_path

class RedactFileProcessor:
    def __init__(self, redact_color=(0, 0, 0), redact_opacity=1.0, ocr_enabled=False, validate_pdfs=False):
        """
        Handles PDF processing operations, including OCR and redaction.

        :param redact_color: Color for redaction/highlight (RGB tuple).
        :param redact_opacity: Opacity for highlights.
        :param ocr_enabled: Enable OCR if PDF lacks text.
        :param validate_pdfs: Enable PDF validation before processing.
        """
        self.redact_color = redact_color
        self.redact_opacity = redact_opacity
        self.ocr_enabled = ocr_enabled
        self.validate_pdfs = validate_pdfs

    def set_redact_color(self, color: Tuple[float, float, float]):
        """Dynamically update the redaction color."""
        self.redact_color = color

    def set_redact_opacity(self, opacity: float):
        """Dynamically update the redaction opacity."""
        self.redact_opacity = opacity

    def validate_pdf(self, file_path: str) -> bool:
        """Validate if the file is a valid PDF."""
        try:
            doc = fitz.open(file_path)
            doc.close()
            return True
        except Exception:
            return False

    def apply_ocr(self, file_path: str) -> str:
        """Apply OCR to extract text from a PDF file."""
        images = convert_from_path(file_path)
        ocr_text = ""
        for image in images:
            ocr_text += image_to_string(image)
        return ocr_text

    def process_pdf(self, input_path: str, output_path: str, detect_redactions) -> Dict[str, Any]:
        """
        Process a PDF for redaction or highlighting.

        :param input_path: Path to the input PDF.
        :param output_path: Path to save the processed PDF.
        :param detect_redactions: Function to detect redaction areas.
        :return: Dictionary with statistics about the processing.
        """
        if self.validate_pdfs and not self.validate_pdf(input_path):
            raise ValueError(f"Invalid PDF: {input_path}")

        doc = fitz.open(input_path)
        total_words = 0
        redacted_words = 0

        for page_index, page in enumerate(doc, start=1):
            wordlist = page.get_text("words")
            if self.ocr_enabled and not wordlist:
                ocr_text = self.apply_ocr(input_path)
                wordlist = [{"bbox": None, "text": word} for word in ocr_text.split()]

            total_words += len(wordlist)
            redact_areas = detect_redactions(wordlist)
            redacted_words += len(redact_areas)

            for rect in redact_areas:
                try:
                    if self.redact_opacity < 1.0:  # Highlight mode
                        rgb = [c if 0 <= c <= 1 else c / 255 for c in self.redact_color]
                        annot = page.add_highlight_annot(rect)
                        annot.set_colors(stroke=rgb)
                        annot.set_opacity(self.redact_opacity)
                        annot.update()
                    else:  # Standard redaction mode
                        annot = page.add_redact_annot(rect)
                        rgb = [c if 0 <= c <= 1 else c / 255 for c in self.redact_color]
                        annot.set_colors(stroke=rgb, fill=rgb)
                        annot.update()
                        page.apply_redactions(images=fitz.PDF_REDACT_IMAGE_NONE)

                except Exception as e:
                    print(f"Error applying annotation on Page {page_index}: {e}")

        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        return {"total_words": total_words, "redacted_words": redacted_words}

# Example usage of RedactFileProcessor class in other modules
# from redactor_file_processing import RedactFileProcessor
# processor = RedactFileProcessor(redact_color=(1, 1, 0), redact_opacity=0.5)
# stats = processor.process_pdf("input.pdf", "output.pdf", detect_redactions_function)
