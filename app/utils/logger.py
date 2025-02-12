# app/utils/logger.py

###############################################################################
# SECTION 1: IMPORTS AND CONSTANTS
###############################################################################
import logging
import logging.handlers  # Added this import for RotatingFileHandler
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Union, Any, Dict

###############################################################################
# SECTION 2: CLASS DEFINITION AND CONSTANTS
###############################################################################
class RedactionLogger:
    """
    Centralized logging system for the redaction application.

    Features:
    - Configurable logging levels
    - File and console output options
    - Specialized logging methods for redaction operations
    - Automatic log rotation
    - Structured logging for detailed debugging
    """

    DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    CONSOLE_FORMAT = '%(levelname)s: %(message)s'


###############################################################################
# SECTION 3: INITIALIZATION AND SETUP
###############################################################################
    def __init__(
        self,
        name: str = 'redaction',
        log_level: str = "INFO",
        log_dir: Optional[Union[str, Path]] = None,
        max_log_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True
    ):
        """Initialize the logger with specified configuration."""
        self.logger = logging.getLogger(name)
        numeric_level = getattr(logging, log_level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        self.logger.propagate = False  # Prevent duplicate logging

        # Clear any existing handlers
        self.logger.handlers = []

        # Configure formatters
        self.file_formatter = logging.Formatter(self.DEFAULT_LOG_FORMAT)
        self.console_formatter = logging.Formatter(self.CONSOLE_FORMAT)

        # Setup handlers
        if log_dir:
            self._setup_file_handler(Path(log_dir), max_log_size, backup_count)

        if console_output:
            self._setup_console_handler()

        self.info(f"Logger initialized: {name} at {log_level} level")

    @property
    def level(self):
        return self.logger.getEffectiveLevel()

###############################################################################
# SECTION 4: HANDLER SETUP METHODS
###############################################################################
    def _setup_file_handler(self, log_dir: Path, max_size: int, backup_count: int) -> None:
        """Configure rotating file handler."""
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / f"redaction_{datetime.now():%Y%m%d}.log"

            handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=max_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            handler.setFormatter(self.file_formatter)
            self.logger.addHandler(handler)

        except Exception as e:
            print(f"Failed to setup file logging: {e}", file=sys.stderr)

    def _setup_console_handler(self) -> None:
        """Configure console handler."""
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(self.console_formatter)
        self.logger.addHandler(handler)

###############################################################################
# SECTION 5: CORE LOGGING METHODS
###############################################################################
    def setLevel(self, level: Union[str, int]) -> None:
        """Set logging level, accepting both string and numeric levels."""
        if isinstance(level, str):
            numeric_level = getattr(logging, level.upper(), None)
            if not isinstance(numeric_level, int):
                raise ValueError(f"Invalid log level string: {level}")
            self.logger.setLevel(numeric_level)
        elif isinstance(level, int):
            self.logger.setLevel(level)
        else:
            raise ValueError(f"Invalid log level type: {type(level)}")

    def debug(self, msg: str, *args, **kwargs: Any) -> None:
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs: Any) -> None:
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs: Any) -> None:
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs: Any) -> None:
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs: Any) -> None:
        self.logger.critical(msg, *args, **kwargs)

###############################################################################
# SECTION 6: SPECIALIZED REDACTION LOGGING METHODS
###############################################################################
    def log_redaction_start(self, file_path: Union[str, Path], config: Dict[str, Any]) -> None:
        """Log the start of a redaction operation."""
        self.info(
            f"Starting redaction for: {file_path}",
            extra={'file': str(file_path), 'config': config}
        )

    def log_redaction_complete(self, file_path: Union[str, Path], stats: Dict[str, Any]) -> None:
        """Log the completion of a redaction operation."""
        self.info(
            f"Completed redaction for: {file_path}",
            extra={'file': str(file_path), 'stats': stats}
        )

    def log_pattern_match(
        self,
        pattern_name: str,
        text: str,
        matched: bool,
        score: Optional[float] = None,
        entity_type: Optional[str] = None
    ) -> None:
        """Log pattern matching results."""
        result = "matched" if matched else "did not match"
        score_info = f" (score: {score:.2f})" if score is not None else ""
        entity_info = f" [{entity_type}]" if entity_type else ""

        self.debug(
            f"Pattern '{pattern_name}'{entity_info} {result} text '{text}'{score_info}",
            extra={
                'pattern': pattern_name,
                'text': text,
                'matched': matched,
                'score': score,
                'entity_type': entity_type
            }
        )

###############################################################################
# SECTION 7: ENTITY DETECTION AND ERROR LOGGING
###############################################################################
    def log_entity_detection(
        self,
        detector: str,
        text: str,
        entities: list,
        processing_time: Optional[float] = None
    ) -> None:
        """Log entity detection results."""
        time_info = f" (took {processing_time:.3f}s)" if processing_time else ""
        self.debug(
            f"{detector} analysis complete{time_info}: '{text}'",
            extra={
                'detector': detector,
                'text': text,
                'entity_count': len(entities)
            }
        )

        for entity in entities:
            self.debug(
                f"- Found {entity.entity_type}: '{entity.text}' "
                f"(confidence: {getattr(entity, 'confidence', 'N/A')})"
            )

    def log_error_context(
        self,
        error: Exception,
        context: Dict[str, Any],
        critical: bool = False
    ) -> None:
        """Log an error with additional context."""
        log_method = self.critical if critical else self.error
        log_method(
            f"{type(error).__name__}: {str(error)}",
            extra={
                'error_type': type(error).__name__,
                'error_message': str(error),
                **context
            }
        )