# redactor/detectors/spacy_detector.py

from typing import List, Dict, Optional, Any, Set
import spacy
from .base_detector import BaseDetector, Entity

class SpacyDetector(BaseDetector):
    def __init__(self, config_loader, logger):
        """
        Initializes the SpaCy Detector with a loaded configuration and logger.

        Args:
            config_loader (ConfigLoader): Handles loading configurations.
            logger (RedactionLogger): Logger for debugging and error reporting.
        """
        super().__init__(config_loader, logger)
        self.nlp = None
        self.entity_types = set()
        self.entity_mappings = {}
        self.confidence_thresholds = {}
        self._validate_config()
        self._load_spacy_model()

    def _load_spacy_model(self):
        """
        Loads the SpaCy NLP model specified in the configuration or defaults to `en_core_web_lg`.
        """
        try:
            import spacy

            model_name = self.config_loader.get_config("main").get("language_model", "en_core_web_lg")
            self.nlp = spacy.load(model_name)
            self.logger.info(f"SpaCy model {model_name} loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading SpaCy model: {e}")
            raise

    def _validate_config(self) -> bool:
        """Validates and loads the routing configuration for SpaCy."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                self.logger.error("Missing routing configuration for SpacyDetector")
                return False
        
            spacy_config = routing_config["routing"].get("spacy_primary", {})
            self.entity_types = set(spacy_config.get("entities", []))
            self.entity_mappings = spacy_config.get("mappings", {})
        
            # Get entity-specific thresholds from config
            self.confidence_thresholds = {}
            thresholds = spacy_config.get("thresholds", {})
            for entity in self.entity_types:
                # Map the entity if needed
                mapped_entity = self.entity_mappings.get(entity, entity)
                # Get threshold or use a default
                threshold = thresholds.get(mapped_entity, 0.6)
                self.confidence_thresholds[mapped_entity] = threshold
    
            self.logger.debug(f"Loaded spaCy entity types: {self.entity_types}")
            self.logger.debug(f"Entity mappings: {self.entity_mappings}")
            self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")
    
            return True
    
        except Exception as e:
            self.logger.error(f"Error validating spaCy config: {str(e)}")
            return False
            
    def _merge_adjacent_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merges adjacent entities that are likely part of the same address or location.
        """
        if not entities:
            return entities

        merged = []
        current = None

        for entity in entities:
            if not current:
                current = entity
                continue

            # Check if entities should be merged
            if (entity.start - current.end <= 2  # Allow for spaces between entities
                and entity.entity_type == current.entity_type
                and self._is_mergeable_entity(current, entity)):
                
                # Merge entities
                current.text = f"{current.text} {entity.text}".strip()
                current.end = entity.end
                current.confidence = min(current.confidence, entity.confidence)
            else:
                merged.append(current)
                current = entity

        if current:
            merged.append(current)

        return merged

    def _is_mergeable_entity(self, ent1: Entity, ent2: Entity) -> bool:
        """
        Determines if two entities should be merged based on their content.
        """
        # Check if entities form a typical address pattern
        text1 = ent1.text.lower()
        text2 = ent2.text.lower()
        
        address_indicators = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'lane', 'ln', 'drive', 'dr'}
        has_number = any(c.isdigit() for c in text1 + text2)
        has_street = any(indicator in text1.split() or indicator in text2.split() 
                        for indicator in address_indicators)
        
        return has_number or has_street

    def detect_entities(self, text: str):
        """
        Detect entities in the provided text using SpaCy NLP.
    
        Args:
            text (str): Input text to analyze.
    
        Returns:
            list[Entity]: List of detected entities.
        """
        if not text:
            self.logger.warning("Empty text provided for entity detection.")
            return []
    
        try:
            doc = self.nlp(text)
            entities = []
    
            for ent in doc.ents:
                if ent.label_ in self.entity_types:
                    # Map entity type if needed
                    mapped_type = self.entity_mappings.get(ent.label_, ent.label_)
                    confidence = self.confidence_thresholds.get(ent.label_, 0.85)
    
                    entity = Entity(
                        text=ent.text,
                        entity_type=mapped_type,
                        confidence=confidence,
                        start=ent.start_char,
                        end=ent.end_char,
                        source="spacy",
                    )
    
                    if self.validate_detection(entity):
                        entities.append(entity)
    
            # Merge adjacent entities that might be part of the same address
            merged_entities = self._merge_adjacent_entities(entities)
            return merged_entities

        except Exception as e:
            self.logger.error(f"Error detecting entities with SpaCy: {e}")
            return []

    def validate_detection(self, entity: Entity):
        """
        Validates if the detected entity passes confidence and formatting checks.

        Args:
            entity (Entity): The entity to validate.

        Returns:
            bool: True if valid, False otherwise.
        """
        if entity.confidence < self.confidence_thresholds.get(entity.entity_type, 0.75):
            self.logger.debug(
                f"Entity '{entity.text}' confidence below threshold: {entity.confidence}"
            )
            return False

        if not entity.text or len(entity.text.strip()) == 0:
            self.logger.debug(f"Entity '{entity.text}' has invalid text.")
            return False

        return True