import yaml
import logging
import re
from typing import Dict, Any, List, Tuple

class RedactorBase:
    """
    Base class providing shared functionality and utilities for redaction operations.
    """

    def __init__(self, config_path: str, log_level: int = logging.INFO):
        """
        Initialize the base class with configurations and logging.
        Args:
            config_path (str): Path to the configuration file (YAML/JSON).
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        self.config = self._load_config(config_path)
        self._setup_logging(log_level)

    @staticmethod
    def _load_config(path: str) -> Dict[str, Any]:
        """
        Load configuration from a YAML or JSON file.
        Args:
            path (str): Path to the configuration file.
        Returns:
            Dict[str, Any]: Parsed configuration data.
        """
        try:
            with open(path, "r") as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {path}: {e}")

    def _setup_logging(self, log_level: int):
        """
        Set up logging for the application.
        Args:
            log_level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        """
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=log_level
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        self.logger.info("Logging initialized.")

    @staticmethod
    def normalize_text(text: str, domain_specific: bool = False) -> str:
        """
        Normalize text for analysis, optionally retaining domain-specific characters.
        Args:
            text (str): Input text to normalize.
            domain_specific (bool): If True, retain characters like '.', '/', '@', etc.
        Returns:
            str: Normalized text.
        """
        if domain_specific:
            return re.sub(r"[^a-zA-Z0-9@./:_-]", "", text).strip().lower()
        return text.translate(str.maketrans("", "", r"""!"#$%&'()*+,;<=>?[\]^`{|}~""")).strip().lower()

    @staticmethod
    def validate_regex(regex: str) -> bool:
        """
        Validate if a regex is correctly formatted.
        Args:
            regex (str): The regex string to validate.
        Returns:
            bool: True if the regex is valid, False otherwise.
        """
        try:
            re.compile(regex)
            return True
        except re.error:
            return False

    def log_debug(self, message: str):
        """
        Log a debug message if logging is enabled at the debug level.
        Args:
            message (str): The message to log.
        """
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(message)

    def log_error(self, message: str):
        """
        Log an error message.
        Args:
            message (str): The message to log.
        """
        self.logger.error(message)

    def log_info(self, message: str):
        """
        Log an informational message.
        Args:
            message (str): The message to log.
        """
        self.logger.info(message)

    @staticmethod
    def combine_bounding_boxes(bboxes: List[Any]) -> Tuple[float, float, float, float]:
        """
        Combine multiple bounding boxes into one unified rectangle.
        Args:
            bboxes (List[Any]): List of bounding boxes to combine.
        Returns:
            Tuple[float, float, float, float]: Combined bounding box.
        """
        x0 = min(bbox[0] for bbox in bboxes if bbox)
        y0 = min(bbox[1] for bbox in bboxes if bbox)
        x1 = max(bbox[2] for bbox in bboxes if bbox)
        y1 = max(bbox[3] for bbox in bboxes if bbox)
        return x0, y0, x1, y1
