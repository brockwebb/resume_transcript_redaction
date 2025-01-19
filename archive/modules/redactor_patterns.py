import re
from presidio_analyzer import PatternRecognizer, Pattern
from typing import Dict, Any, List

class RedactorPatterns:
    """
    A class for managing and utilizing regex-based patterns in redaction workflows.
    """

    def __init__(self, detection_config: Dict[str, Any]):
        """
        Initialize the pattern manager with detection configurations.
        Args:
            detection_config (Dict[str, Any]): The YAML or JSON configuration containing patterns.
        """
        self.detection_config = detection_config
        self.analyzer_engine = self._setup_presidio_analyzer()

    def _setup_presidio_analyzer(self) -> Dict[str, PatternRecognizer]:
        """
        Set up Presidio Analyzer with regex patterns.
        Returns:
            Dict[str, PatternRecognizer]: A dictionary of pattern recognizers.
        """
        pattern_recognizers = {}
        for entity_type, patterns in self.detection_config.items():
            if isinstance(patterns, list):
                for pattern in patterns:
                    recognizer = PatternRecognizer(
                        supported_entity=entity_type,
                        patterns=[Pattern(
                            name=pattern["name"],
                            regex=pattern["regex"],
                            score=pattern.get("score", 0.7)
                        )]
                    )
                    pattern_recognizers[entity_type] = recognizer
        return pattern_recognizers

    def detect_patterns(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect patterns in the provided text using Presidio Analyzer.
        Args:
            text (str): The input text to analyze.
        Returns:
            List[Dict[str, Any]]: A list of detected entities and their positions.
        """
        detections = []
        for recognizer in self.analyzer_engine.values():
            matches = recognizer.analyze(
                text=text,
                entities=recognizer.supported_entities,
                language="en"
            )
            for match in matches:
                detections.append({
                    "entity_type": match.entity_type,
                    "start": match.start,
                    "end": match.end,
                    "score": match.score
                })
        return detections

    def load_patterns_from_config(self, config_path: str):
        """
        Load detection patterns from a YAML or JSON configuration file.
        Args:
            config_path (str): The path to the configuration file.
        """
        try:
            with open(config_path, "r") as file:
                self.detection_config = yaml.safe_load(file)
            self.analyzer_engine = self._setup_presidio_analyzer()
        except Exception as e:
            raise ValueError(f"Failed to load patterns from {config_path}: {e}")

    @staticmethod
    def create_pattern(name: str, regex: str, score: float) -> Pattern:
        """
        Create a Presidio Pattern object.
        Args:
            name (str): The name of the pattern.
            regex (str): The regex string.
            score (float): Confidence score for the pattern.
        Returns:
            Pattern: A Presidio Pattern object.
        """
        return Pattern(name=name, regex=regex, score=score)

    @staticmethod
    def validate_pattern(regex: str) -> bool:
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
