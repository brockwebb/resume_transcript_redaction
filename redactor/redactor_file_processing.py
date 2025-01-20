# redactor/redactor_file_processing.py
import fitz
from typing import Callable, Dict, List, Tuple, Optional, Any
from pathlib import Path

class RedactFileProcessor:
    """Handles the file-level operations for PDF redaction."""

    def __init__(self, 
                 redact_color: Tuple[float, float, float] = (0, 0, 0),
                 redact_opacity: float = 1.0,
                 ocr_enabled: bool = False,
                 validate_pdfs: bool = False,
                 logger: Optional[Any] = None):
        """
        Initialize the file processor.

        Args:
            redact_color: RGB color tuple for redactions
            redact_opacity: Opacity level for redactions (0.0 to 1.0)
            ocr_enabled: Whether to use OCR for processing
            validate_pdfs: Whether to validate PDF files
            logger: Optional logger instance
        """
        self.redact_color = redact_color
        self.redact_opacity = redact_opacity
        self.ocr_enabled = ocr_enabled
        self.validate_pdfs = validate_pdfs

        # Setup logging
        self.logger = logger
        self.log = self._create_log_functions()

    def _create_log_functions(self):
        """Create logging functions that either use logger or print."""
        class LogWrapper:
            def __init__(self, logger):
                self.logger = logger

            def info(self, msg):
                if self.logger:
                    self.logger.info(msg)
                else:
                    print(msg)

            def debug(self, msg):
                if self.logger:
                    self.logger.debug(msg)
                else:
                    print(msg)

            def warning(self, msg):
                if self.logger:
                    self.logger.warning(msg)
                else:
                    print(f"WARNING: {msg}")

            def error(self, msg):
                if self.logger:
                    self.logger.error(msg)
                else:
                    print(f"ERROR: {msg}")

        return LogWrapper(self.logger)

    def process_pdf(self, 
                   input_path: str, 
                   output_path: str,
                   detect_redactions: Callable) -> Dict[str, int]:
        """
        Process a PDF file and apply redactions.
    
        Args:
            input_path: Path to input PDF
            output_path: Path to save redacted PDF
            detect_redactions: Function to detect areas needing redaction
    
        Returns:
            Dict with processing statistics
        """
        stats = {"total_words": 0, "redacted_words": 0}
    
        self.log.info(f"Opening PDF: {input_path}")
        try:
            doc = fitz.open(input_path)
    
            for page_num, page in enumerate(doc):
                self.log.debug(f"Processing page {page_num + 1}/{len(doc)}")
    
                # Get words from page
                words = page.get_text("words")
                stats["total_words"] += len(words)
    
                # Detect areas to redact
                redact_areas = detect_redactions(words)
                stats["redacted_words"] += len(redact_areas)
    
                # Apply redactions
                shape = page.new_shape()
                for rect in redact_areas:
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
    
                shape.commit()
                
            # Save the redacted document
            self.log.info(f"Saving redacted PDF to: {output_path}")
            doc.save(output_path)
            doc.close()
    
            self.log.info(f"Processing complete. Stats: {stats}")
            return stats
    
        except Exception as e:
            self.log.error(f"Error processing PDF: {str(e)}")
            raise

    def validate_pdf(self, file_path: str) -> bool:
        """
        Validate if a file is a valid PDF.

        Args:
            file_path: Path to PDF file

        Returns:
            bool: Whether file is valid
        """
        try:
            doc = fitz.open(file_path)
            is_valid = doc.is_pdf
            doc.close()
            if is_valid:
                self.log.debug(f"PDF validation successful: {file_path}")
            else:
                self.log.warning(f"Invalid PDF file: {file_path}")
            return is_valid
        except Exception as e:
            self.log.error(f"Error validating PDF {file_path}: {str(e)}")
            return False