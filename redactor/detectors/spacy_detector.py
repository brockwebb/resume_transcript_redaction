# redactor/detectors/spacy_detector.py

from typing import List, Dict, Optional, Any, Set
import spacy
from .base_detector import BaseDetector, Entity

class SpacyDetector(BaseDetector):
    """Detector implementation using spaCy for entity detection."""
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 model: str = "en_core_web_lg"):
        """Initialize spaCy detector."""
        super().__init__(config_loader, logger)
        
        self.model_name = model
        self.nlp = None
        self.allowed_entities = set()
        self.confidence_thresholds = {}
        
        # Initialize spaCy
        self._init_spacy()

    def _validate_config(self) -> bool:
        """Validate spaCy-specific configuration."""
        try:
            # Get entity routing config
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                self.logger.error("Missing entity routing configuration")
                return False

            # Get spaCy section
            spacy_section = routing_config.get("routing", {}).get("spacy_primary", {})
            if not spacy_section:
                self.logger.error("Missing spacy_primary configuration")
                return False

            # Store allowed entities and their thresholds
            self.allowed_entities = set(spacy_section.get("entities", []))
            base_threshold = spacy_section.get("confidence_threshold", 0.75)
            
            # Store thresholds for each entity type
            self.confidence_thresholds = {
                entity: spacy_section.get("thresholds", {}).get(entity, base_threshold)
                for entity in self.allowed_entities
            }

            self.logger.debug(
                f"Configured for entities: {self.allowed_entities} "
                f"with thresholds: {self.confidence_thresholds}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Error validating spaCy config: {str(e)}")
            return False

    def _init_spacy(self) -> None:
        """Initialize spaCy model."""
        try:
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"spaCy model {self.model_name} loaded successfully")
            
            # Log available entity types
            self.logger.debug(
                f"Available spaCy entity types: {self.nlp.pipe_labels['ner']}"
            )
            
        except Exception as e:
            self.logger.error(f"Error loading spaCy model: {str(e)}")
            raise

    def detect_entities(self, text: str) -> List[Entity]:
        """Detect entities using spaCy."""
        if not text or not self.nlp:
            return []
            
        try:
            # Process text with spaCy
            self.logger.debug(f"Processing text: {text[:100]}...")
            doc = self.nlp(text)
            
            entities = []
            for ent in doc.ents:
                # Skip if not in allowed entities
                if ent.label_ not in self.allowed_entities:
                    self.logger.debug(f"Skipping entity type: {ent.label_}")
                    continue
                
                # Calculate confidence based on vector norm
                confidence = float(ent.vector_norm) / 10.0  # Normalize to 0-1 range
                
                # Check against entity-specific threshold
                threshold = self.confidence_thresholds.get(ent.label_, 0.75)
                if confidence < threshold:
                    self.logger.debug(
                        f"Entity {ent.text} ({ent.label_}) below threshold: "
                        f"{confidence} < {threshold}"
                    )
                    continue
                
                entity = Entity(
                    text=ent.text,
                    entity_type=ent.label_,
                    confidence=confidence,
                    start=ent.start_char,
                    end=ent.end_char,
                    source="spacy",
                    metadata={
                        "dep": ent.root.dep_,  # Dependency parsing info
                        "pos": ent.root.pos_,  # Part of speech
                        "is_title": ent.root.is_title  # Capitalization check
                    }
                )
                
                # Validate before adding
                if self.validate_detection(entity):
                    entities.append(entity)
                    self.log_detection(entity)
                else:
                    self.logger.debug(f"Entity failed validation: {entity}")
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error in spaCy detection: {str(e)}")
            return []

    def validate_detection(self, entity: Entity) -> bool:
        """Validate spaCy detection."""
        try:
            # Entity type validation already done in detect_entities
            
            # Additional validation based on entity type
            if entity.entity_type == "PERSON":
                return self._validate_person(entity)
            elif entity.entity_type == "ORG":
                return self._validate_organization(entity)
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False

    def _validate_person(self, entity: Entity) -> bool:
        """Additional validation for person entities."""
        # Skip single-word persons unless they're titles
        if len(entity.text.split()) == 1:
            return entity.metadata.get("is_title", False)
        return True

    def _validate_organization(self, entity: Entity) -> bool:
        """Additional validation for organization entities."""
        # Require proper noun for organizations
        return entity.metadata.get("pos") == "PROPN"