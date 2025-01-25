from typing import List, Optional, Any, Dict
from presidio_analyzer import AnalyzerEngine
from .base_detector import BaseDetector, Entity
from .custom_recognizers import (
    ConfigDrivenPhoneRecognizer,
    ConfigDrivenSsnRecognizer,
    ConfigDrivenAddressRecognizer,
    ConfigDrivenEducationalInstitutionRecognizer,
    ConfigDrivenGpaRecognizer,
    ConfigDrivenLocationRecognizer,
    ConfigDrivenSocialMediaRecognizer
)
import re


class PresidioDetector(BaseDetector):
    def __init__(self, config_loader: Any, logger: Optional[Any] = None, language: str = "en"):
        """Initialize Presidio detector with configuration."""
        self.language = language
        self.analyzer = None
        self.entity_types = {}
        self.ignore_date_words = []
        self.ignore_phrases = []
        self.social_media_patterns = []
        self.social_context_words = []
        
        # Initialize base class
        super().__init__(config_loader, logger)
    
        # Initialize Presidio analyzer and load entity types
        self._init_presidio()
        self._load_entity_types()
    
        # Load configurations
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            presidio_primary = routing_config.get("routing", {}).get("presidio_primary", {})
            validation_config = presidio_primary.get("validation", {})
            
            # General configurations
            self.ignore_phrases = validation_config.get("ignore_phrases", [])
            self.ignore_date_words = validation_config.get("date_ignore_words", [])
            
            # Social media patterns from core_patterns.yaml
            core_patterns = self.config_loader.get_config("core_patterns")
            self.social_media_patterns = core_patterns.get("social_media_patterns", [])
            self.social_context_words = core_patterns.get("context_words", {}).get("social_media", [])
            
            self.logger.debug(f"Loaded ignore phrases: {self.ignore_phrases}")
            self.logger.debug(f"Loaded date ignore words: {self.ignore_date_words}")
            self.logger.debug(f"Loaded social media patterns: {self.social_media_patterns}")
            self.logger.debug(f"Loaded social media context words: {self.social_context_words}")
        except Exception as e:
            self.logger.error(f"Failed to load configurations: {str(e)}")

            
    def _init_presidio(self) -> None:
        """Initialize Presidio analyzer with default and custom recognizers."""
        try:
            self.analyzer = AnalyzerEngine()

            # Remove default recognizers that are replaced by custom ones
            self.analyzer.registry.remove_recognizer("PhoneRecognizer")
            self.analyzer.registry.remove_recognizer("UsSsnRecognizer")

            # Add custom recognizers
            self.analyzer.registry.add_recognizer(ConfigDrivenAddressRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenLocationRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenPhoneRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenEducationalInstitutionRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenGpaRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenSocialMediaRecognizer(self.config_loader, self.logger))
            self.analyzer.registry.add_recognizer(ConfigDrivenSsnRecognizer(self.config_loader, self.logger))

            recognizers = self.analyzer.get_recognizers()
            self.logger.debug(f"Initialized Presidio with recognizers: {[r.name for r in recognizers]}")
        except Exception as e:
            self.logger.error(f"Error initializing Presidio analyzer: {str(e)}")
            raise

    def _load_entity_types(self) -> None:
        """Load entity types and their thresholds from configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            base_threshold = presidio_config.get("confidence_threshold", 0.75)
            entities = presidio_config.get("entities", [])

            for entity in entities:
                threshold = presidio_config.get("thresholds", {}).get(entity, base_threshold)
                self.entity_types[entity] = threshold

            self.logger.debug("Loaded Presidio entity types with thresholds:")
            for entity, threshold in self.entity_types.items():
                self.logger.debug(f"  {entity}: {threshold}")
        except Exception as e:
            self.logger.error(f"Error loading entity types: {str(e)}")
            raise

    def _validate_config(self) -> bool:
        """Validate the configuration for PresidioDetector."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                self.logger.error("Missing routing configuration for PresidioDetector.")
                return False

            presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
            if not presidio_config:
                self.logger.error("Missing 'presidio_primary' configuration in routing.")
                return False

            if "entities" not in presidio_config:
                self.logger.error("'entities' is missing in 'presidio_primary' configuration.")
                return False

            self.logger.debug("PresidioDetector configuration validated successfully.")
            return True

        except Exception as e:
            self.logger.error(f"Error validating PresidioDetector configuration: {str(e)}")
            return False

    def detect_entities(self, text: str) -> List[Entity]:
        """Detect entities using Presidio's built-in recognizers."""
        if not text or not self.analyzer or not self.entity_types:
            return []

        try:
            self.logger.debug(f"Analyzing text with Presidio: {text[:100]}...")

            results = self.analyzer.analyze(
                text=text,
                language=self.language,
                entities=list(self.entity_types.keys())
            )

            self.logger.debug("Raw Presidio results before filtering:")
            for result in results:
                self.logger.debug(
                    f"  Found: '{text[result.start:result.end]}' "
                    f"Type: {result.entity_type} "
                    f"Score: {result.score:.3f} "
                    f"Start: {result.start} End: {result.end} "
                    f"Recognition data: {result.recognition_metadata}"
                )

            entities = []
            for result in results:
                entity = Entity(
                    text=text[result.start:result.end],
                    entity_type=result.entity_type,
                    confidence=result.score,
                    start=result.start,
                    end=result.end,
                    source="presidio",
                    metadata={
                        "recognition_metadata": result.recognition_metadata,
                        "analysis_explanation": result.analysis_explanation
                    }
                )

                # Reclassify URL as SOCIAL_MEDIA if applicable
                if entity.entity_type == "URL":
                    reclassified_type = self._reclassify_url_as_social_media(entity.text, text)
                    if reclassified_type:
                        self.logger.debug(
                            f"Entity '{entity.text}' reclassified from 'URL' to '{reclassified_type}'."
                        )
                        entity.entity_type = reclassified_type

                # Validate detection
                if self.validate_detection(entity, text):
                    entities.append(entity)
                    self.log_detection(entity)
                else:
                    self.logger.debug(
                        f"Entity failed validation: {entity.entity_type} "
                        f"({entity.confidence:.3f} < "
                        f"{self.entity_types.get(entity.entity_type, self.confidence_threshold)}) "
                        f"Text: '{entity.text}'"
                    )

            return entities

        except Exception as e:
            self.logger.error(f"Error during Presidio detection: {str(e)}")
            return []

    def validate_detection(self, entity: Entity, full_text: str) -> bool:
        """Validate detection against thresholds and ignore phrases."""
        try:
            if self._should_ignore(entity.text, full_text, entity.entity_type):
                self.logger.debug(f"Entity '{entity.text}' ignored due to ignore phrases.")
                return False

            threshold = self.entity_types.get(entity.entity_type, self.confidence_threshold)
            if entity.confidence < threshold:
                self.logger.debug(f"Entity '{entity.text}' confidence below threshold: {entity.confidence:.2f} < {threshold}")
                return False

            if not entity.text.strip():
                self.logger.debug(f"Entity has invalid text: '{entity.text}'")
                return False

            return True
        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False

    def _should_ignore(self, entity_text: str, full_text: str, entity_type: str) -> bool:
        """Determine if an entity should be ignored based on ignore phrases and type-specific logic."""
        self.logger.debug(f"Checking ignore for entity: '{entity_text}' with type '{entity_type}'.")

        for phrase in self.ignore_phrases:
            if phrase.lower() in entity_text.lower():
                self.logger.debug(f"Ignored due to phrase match: '{phrase}' in '{entity_text}'.")
                return True

        if entity_type == "DATE_TIME":
            for phrase in self.ignore_date_words:
                if phrase.lower() in entity_text.lower():
                    self.logger.debug(f"Ignored DATE_TIME due to ignore word: '{phrase}' in '{entity_text}'.")
                    return True

        return False

    def _reclassify_url_as_social_media(self, url: str, full_text: str) -> Optional[str]:
        """Reclassify a URL as SOCIAL_MEDIA if it matches patterns or context."""
        try:
            # First check if URL matches any social media patterns
            for pattern in self.social_media_patterns:
                if not pattern.get("regex"):
                    continue
                    
                try:
                    if re.search(pattern["regex"], url, re.IGNORECASE):
                        self.logger.debug(
                            f"URL '{url}' matches social media pattern: {pattern.get('name', 'Unnamed')}"
                        )
                        return "SOCIAL_MEDIA"
                except re.error as e:
                    self.logger.error(f"Invalid regex pattern {pattern['regex']}: {str(e)}")
                    continue
    
            # Then check context words in surrounding text
            text_lower = full_text.lower()
            for word in self.social_context_words:
                word_lower = word.lower()
                # Look for word boundaries to avoid partial matches
                pattern = r'\b' + re.escape(word_lower) + r'\b'
                if re.search(pattern, text_lower):
                    # Check if context word is near the URL (within 100 characters)
                    url_pos = text_lower.find(url.lower())
                    if url_pos != -1:
                        # Get surrounding text window
                        start = max(0, url_pos - 100)
                        end = min(len(text_lower), url_pos + len(url) + 100)
                        window = text_lower[start:end]
                        
                        if re.search(pattern, window):
                            self.logger.debug(
                                f"URL '{url}' reclassified as SOCIAL_MEDIA due to nearby context word: '{word}'"
                            )
                            return "SOCIAL_MEDIA"
    
            return None
    
        except Exception as e:
            self.logger.error(f"Error reclassifying URL '{url}': {str(e)}")
            return None


