from typing import List, Dict, Optional, Any, Set
import spacy
from .base_detector import BaseDetector, Entity
from redactor.validation.validation_rules import EntityValidationRules
import re  # for regex patterns
import json


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

        # Add validation rules while preserving existing initialization
        try:
            self.validation_rules = EntityValidationRules(logger)
            if logger:
                logger.debug("Validation rules initialized successfully")
        except Exception as e:
            if logger:
                logger.warning(f"Failed to initialize validation rules: {e}. Will continue with default validation.")
            self.validation_rules = None

        # Keep existing initialization
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

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """Validate entity detections with comprehensive checks."""
        # Skip validation for empty/invalid entities
        if not entity or not entity.text or len(entity.text.strip()) == 0:
            self.logger.debug(f"Empty or invalid entity text")
            return False

        # Initialize reasons tracker
        self.last_validation_reasons = []

        # Check confidence threshold
        threshold = self.confidence_thresholds.get(entity.entity_type, 0.75)
        if entity.confidence < threshold:
            reason = f"Entity '{entity.text}' confidence below threshold: {entity.confidence} < {threshold}"
            self.logger.debug(reason)
            self.last_validation_reasons.append(reason)
            return False

        # Entity-specific validation
        is_valid = False
        if entity.entity_type == "PERSON":
            is_valid = self._validate_person(entity, text)

        elif entity.entity_type == "EDUCATIONAL_INSTITUTION":
            # Try new validation first if available
            if hasattr(self, 'validation_rules') and self.validation_rules:
                try:
                    result = self.validation_rules.validate_educational_institution(entity.text, text)
                    if result.is_valid:
                        confidence_adj = result.confidence_adjustment
                        entity.confidence += confidence_adj
                        self.last_validation_reasons.append(
                            f"Validation passed with confidence adjustment: {confidence_adj}"
                        )
                        self.logger.debug(
                            f"Validation passed for '{entity.text}'. Confidence adjusted by {confidence_adj} to {entity.confidence}."
                        )
                        return True
                    else:
                        reason = result.reason
                        self.logger.debug(f"New education validation failed: {reason}")
                        self.last_validation_reasons.append(reason)
                        return False
                except Exception as e:
                    self.logger.warning(f"New validation failed, using fallback: {e}")

            # Fall back to existing validation
            is_valid = self._validate_educational_institution(entity, text)
            if not is_valid:
                self.last_validation_reasons.append("Fallback validation failed")
            return is_valid

        elif entity.entity_type == "LOCATION":
            # Require at least one word character
            if not any(c.isalpha() for c in entity.text):
                self.last_validation_reasons.append("No alphabetic characters in location")
                return False
            # Require proper capitalization for location names
            if not entity.text[0].isupper():
                self.last_validation_reasons.append("Location not properly capitalized")
                return False

        elif entity.entity_type == "SOCIAL_MEDIA":
            # Validate social media handles
            if entity.text.startswith('@'):
                # Twitter handle validation
                if not re.match(r'@[\w_]{1,15}$', entity.text):
                    self.last_validation_reasons.append("Invalid Twitter handle format")
                    return False

        elif entity.entity_type == "GPA":
            # Validate GPA format and range
            gpa_match = re.search(r'\d+\.\d+', entity.text)
            if gpa_match:
                gpa_value = float(gpa_match.group())
                if gpa_value < 0 or gpa_value > 4.0:
                    self.last_validation_reasons.append("GPA out of valid range")
                    return False

        # Reject single-word entities for certain types
        if entity.entity_type in {"EDUCATIONAL_INSTITUTION", "ORGANIZATION"}:
            if len(entity.text.split()) < 2:
                self.last_validation_reasons.append(f"Rejected single-word entity: {entity.text}")
                return False

        # All validations passed
        return True

    def _validate_person(self, entity: Entity, text: str) -> bool:
        """Validate PERSON entities with additional checks."""
        name_parts = entity.text.split()

        # Exclude single-word entities unless they are proper nouns
        if len(name_parts) == 1:
            if not name_parts[0][0].isupper():
                return False
            # Further context validation
            start = max(0, entity.start - 20)
            end = min(len(text), entity.end + 20)
            context = text[start:end].lower()
            name_indicators = {'mr', 'mrs', 'ms', 'dr', 'prof', 'professor'}
            if not any(indicator in context for indicator in name_indicators):
                return False

        # Validate multi-word names
        if len(name_parts) >= 2 and all(part[0].isupper() for part in name_parts):
            return True

        return False

    def _validate_educational_institution(self, entity: Entity, text: str) -> bool:
        """Validate educational institution entities using JSON configuration."""
        edu_config = self.config_loader.get_config("validation_params").get("educational", {})
        core_terms = set(edu_config.get("core_terms", []))
        known_acronyms = set(edu_config.get("known_acronyms", []))
        confidence_boosts = edu_config.get("confidence_boosts", {})
        confidence_penalties = edu_config.get("confidence_penalties", {})
        min_words = edu_config.get("min_words", 2)
        generic_terms = set(edu_config.get("generic_terms", []))

        text_lower = entity.text.lower()
        words = text_lower.split()
        is_valid = True

        # Apply penalties and boosts
        if len(words) < min_words:
            penalty = confidence_penalties.get("single_word", -0.3)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for single word: {penalty}")
            is_valid = False

        if any(term in text_lower for term in generic_terms):
            penalty = confidence_penalties.get("generic_term", -1.0)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for generic term: {penalty}")
            is_valid = False

        if not any(term in text_lower for term in core_terms):
            penalty = confidence_penalties.get("no_core_terms", -1.0)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for missing core terms: {penalty}")
            is_valid = False

        if entity.text.isupper() and entity.text not in known_acronyms:
            penalty = confidence_penalties.get("improper_caps", -0.2)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for improper caps: {penalty}")
            is_valid = False

        if entity.text in known_acronyms:
            boost = confidence_boosts.get("known_acronym", 0.3)
            entity.confidence += boost
            self.last_validation_reasons.append(f"Boost for known acronym: {boost}")
            is_valid = True

        if any(term in text_lower for term in core_terms) and len(words) >= min_words:
            boost = confidence_boosts.get("full_name", 0.2)
            entity.confidence += boost
            self.last_validation_reasons.append(f"Boost for full name: {boost}")
            is_valid = True

        return is_valid

    def detect_entities(self, text: str):
        """Detect entities in the provided text using SpaCy NLP."""
        if not text:
            self.logger.warning("Empty text provided for entity detection.")
            return []

        try:
            doc = self.nlp(text)
            entities = []

            for ent in doc.ents:
                if ent.label_ in self.entity_types:
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

                    if self.validate_detection(entity, text):
                        entities.append(entity)

            self.logger.debug("Entities before merging:")
            for entity in entities:
                self.logger.debug(f"{entity.entity_type}: '{entity.text}' ({entity.start}-{entity.end})")

            merged_entities = self._merge_adjacent_entities(entities)

            self.logger.debug("Entities after merging:")
            for entity in merged_entities:
                self.logger.debug(f"{entity.entity_type}: '{entity.text}' ({entity.start}-{entity.end})")

            return merged_entities

        except Exception as e:
            self.logger.error(f"Error detecting entities with SpaCy: {e}")
            return []

    def _merge_adjacent_entities(self, entities: List[Entity]) -> List[Entity]:
        """
        Merges adjacent entities that are likely part of the same address, location, or educational institution.
        """
        if not entities:
            return entities

        merged = []
        current = None

        for entity in entities:
            if not current:
                current = entity
                continue

            # Check if entities are mergeable
            if (entity.start - current.end <= 2  # Allow for small gaps (e.g., spaces)
                and self._is_mergeable_entity(current, entity)):

                # Merge entities: Concatenate text, update end position, adjust confidence
                current.text = f"{current.text} {entity.text}".strip()
                current.end = entity.end
                current.confidence = min(current.confidence, entity.confidence)
            else:
                # Append the current entity if no merge occurs
                merged.append(current)
                current = entity

        # Append the last processed entity
        if current:
            merged.append(current)

        return merged

    def _is_mergeable_entity(self, ent1: Entity, ent2: Entity) -> bool:
        """
        Determines if two entities should be merged based on their content.
        """
        # Only merge entities of the same type
        if ent1.entity_type != ent2.entity_type:
            return False

        # Special handling for EDUCATIONAL_INSTITUTION: Always merge adjacent instances
        if ent1.entity_type == "EDUCATIONAL_INSTITUTION":
            return True

        # Original logic for merging ADDRESS entities
        text1 = ent1.text.lower()
        text2 = ent2.text.lower()
        address_indicators = {'street', 'st', 'avenue', 'ave', 'road', 'rd', 'lane', 'ln', 'drive', 'dr'}
        has_number = any(c.isdigit() for c in text1 + text2)
        has_street = any(indicator in text1.split() or indicator in text2.split() 
                        for indicator in address_indicators)

        return has_number or has_street
