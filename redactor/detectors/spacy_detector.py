###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

from typing import List, Dict, Optional, Any, Set
import spacy
import re
import json
from .base_detector import BaseDetector, Entity
from redactor.validation.validation_rules import EntityValidationRules

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class SpacyDetector(BaseDetector):
    """
    Uses spaCy for context-aware named entity recognition.
    Focuses on linguistic analysis for entity validation.
    """
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None,
                 confidence_threshold: float = 0.7):
        """
        Initialize SpacyDetector with configuration and NLP model.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance
            confidence_threshold: Minimum confidence threshold
        """
        # Initialize base detector
        super().__init__(config_loader, logger, confidence_threshold)
        
        self.nlp = None
        self.entity_types = set()
        self.entity_mappings = {}
        self.confidence_thresholds = {}

        # Initialize validation rules
        try:
            self.validation_rules = EntityValidationRules(logger)
            if self.debug_mode:
                self.logger.debug("Validation rules initialized successfully")
        except Exception as e:
            if self.logger:
                self.logger.warning(
                    f"Failed to initialize validation rules: {e}. "
                    "Will continue with default validation."
                )
            self.validation_rules = None

        # Initialize core components
        self._validate_config()
        self._load_spacy_model()

    def _load_spacy_model(self) -> None:
        """Load the spaCy NLP model specified in configuration."""
        try:
            # Try to load large model first, fall back to small if needed
            model_name = self.config_loader.get_config("main").get(
                "language_model", 
                "en_core_web_lg"
            )
            try:
                if self.debug_mode:
                    self.logger.debug(f"Loading spaCy model: {model_name}")
                self.nlp = spacy.load(model_name)
            except OSError:
                if self.debug_mode:
                    self.logger.debug("Large model failed, falling back to en_core_web_sm")
                self.nlp = spacy.load("en_core_web_sm")

            if self.debug_mode:
                self.logger.debug(f"SpaCy model {self.nlp.meta['name']} loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading SpaCy model: {e}")
            raise

    
    def _validate_config(self) -> bool:
        """
        Validate routing configuration for SpacyDetector.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config or "routing" not in routing_config:
                self.logger.error("Missing routing configuration for SpacyDetector")
                return False
    
            spacy_config = routing_config["routing"].get("spacy_primary", {})
            ensemble_config = routing_config["routing"].get("ensemble_required", {})
            
            # Load basic entities from spacy_primary
            self.entity_types = set(spacy_config.get("entities", []))
            self.entity_mappings = spacy_config.get("mappings", {})
            
            # Get base thresholds
            base_thresholds = spacy_config.get("thresholds", {})
            ensemble_thresholds = ensemble_config.get("confidence_thresholds", {})
            
            # Initialize and populate thresholds
            self.confidence_thresholds = {}
            
            # Add base thresholds
            for entity, threshold in base_thresholds.items():
                self.confidence_thresholds[entity] = threshold
                
            # Add ensemble thresholds - lower for contextual detection
            for entity in ensemble_thresholds:
                self.confidence_thresholds[entity] = 0.6  # Base confidence for contextual
    
            if self.debug_mode:
                self.logger.debug(f"Loaded spaCy entity types: {self.entity_types}")
                self.logger.debug(f"Entity mappings: {self.entity_mappings}")
                self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")
    
            return True
    
        except Exception as e:
            self.logger.error(f"Error validating spaCy config: {str(e)}")
            return False


###############################################################################
# SECTION 3: CORE ENTITY DETECTION AND PROCESSING
###############################################################################

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

            if self.debug_mode:
                self.logger.debug("Entities before merging:")
                for entity in entities:
                    self.logger.debug(f"{entity.entity_type}: '{entity.text}' ({entity.start}-{entity.end})")

            merged_entities = self._merge_adjacent_entities(entities)

            if self.debug_mode:
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

    ###############################################################################
# SECTION 4: VALIDATION METHODS
###############################################################################

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate entity detection using type-specific rules.

        Args:
            entity: Entity to validate
            text: Optional context text for validation
            
        Returns:
            bool indicating if entity is valid
        """
        if not entity or not entity.text or len(entity.text.strip()) == 0:
            self.logger.debug(f"Empty or invalid entity text")
            return False
    
        self.last_validation_reasons = []
        threshold = self.confidence_thresholds.get(entity.entity_type, 0.75)
        
        # Check base confidence threshold
        if entity.confidence < threshold:
            reason = f"Entity '{entity.text}' confidence below threshold: {entity.confidence} < {threshold}"
            self.logger.debug(reason)
            self.last_validation_reasons.append(reason)
            return False
    
        is_valid = False
        # Route to type-specific validation
        if entity.entity_type == "PERSON":
            is_valid = self._validate_person(entity, text)
            if not is_valid:
                return False
                
        elif entity.entity_type == "EDUCATIONAL_INSTITUTION":
            if hasattr(self, 'validation_rules') and self.validation_rules:
                try:
                    result = self.validation_rules.validate_educational_institution(entity, text)
                    if result.is_valid:
                        entity.confidence += result.confidence_adjustment
                        self.last_validation_reasons.append(
                            f"Validation passed with confidence adjustment: {result.confidence_adjustment}"
                        )
                        return True
                    else:
                        self.last_validation_reasons.append(result.reason)
                        return False
                except Exception as e:
                    self.logger.warning(f"New validation failed, using fallback: {e}")
    
            is_valid = self._validate_educational_institution(entity, text)
            if not is_valid:
                self.last_validation_reasons.append("Fallback validation failed")
            return is_valid
            
        elif entity.entity_type == "LOCATION":
            is_valid = self._validate_location(entity, text)
            if not is_valid:
                return False
                
        elif entity.entity_type == "PROTECTED_CLASS":
            is_valid = self._validate_protected_class(entity, text)
            if not is_valid:
                return False
                
        elif entity.entity_type == "PHI":
            is_valid = self._validate_phi(entity, text)
            if not is_valid:
                return False
                
        elif entity.entity_type == "GPA":
            gpa_match = re.search(r'\d+\.\d+', entity.text)
            if gpa_match:
                gpa_value = float(gpa_match.group())
                if gpa_value < 0 or gpa_value > 4.0:
                    self.last_validation_reasons.append("GPA out of valid range")
                    return False
    
        # Additional validation for organizations and educational institutions
        if entity.entity_type in {"EDUCATIONAL_INSTITUTION", "ORGANIZATION"}:
            if len(entity.text.split()) < 2:
                self.last_validation_reasons.append(f"Rejected single-word entity: {entity.text}")
                return False
    
        return True

    ###############################################################################
# SECTION 5: PERSON AND LOCATION VALIDATION
###############################################################################

    def _validate_person(self, entity: Entity, text: str) -> bool:
        """
        Validate PERSON entities using configuration-based rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool: Whether entity is valid
        """
        if not entity.text:
            return False
    
        person_config = self.config_loader.get_config("validation_params").get("person", {})
        title_markers = set(person_config.get("title_markers", []))
        confidence_boosts = person_config.get("confidence_boosts", {})
        confidence_penalties = person_config.get("confidence_penalties", {})
        validation_rules = person_config.get("validation_rules", {})
        
        name_parts = entity.text.split()
        text_lower = entity.text.lower()
        reasons = []
    
        # Length validation
        if len(entity.text) < validation_rules.get("min_chars", 2):
            reasons.append(f"Name too short: {len(entity.text)} chars")
            self.last_validation_reasons = reasons
            return False
    
        if len(name_parts) < validation_rules.get("min_words", 1):
            reasons.append("Insufficient word count")
            self.last_validation_reasons = reasons
            return False
    
        # Single word validation - must be capitalized or have title
        if len(name_parts) == 1:
            if not name_parts[0][0].isupper() and not any(marker in text_lower for marker in title_markers):
                reasons.append("Single word must be capitalized or have title")
                self.last_validation_reasons = reasons
                return False
    
        # Check for title markers
        has_title = any(marker in text_lower for marker in title_markers)
        if has_title:
            boost = confidence_boosts.get("has_title", 0.2)
            entity.confidence += boost
            reasons.append(f"Title marker found: +{boost}")
    
        # Handle ALL CAPS names
        is_all_caps = entity.text.isupper()
        if is_all_caps and validation_rules.get("allow_all_caps", True):
            # ALL CAPS is allowed, check if it's a valid name format
            if len(name_parts) >= 2 or has_title:
                boost = confidence_boosts.get("multiple_words", 0.1)
                entity.confidence += boost
                reasons.append(f"Valid ALL CAPS name: +{boost}")
            else:
                # Single word ALL CAPS without title
                reasons.append("Single word ALL CAPS without title")
                self.last_validation_reasons = reasons
                return False
        else:
            # Regular capitalization check
            if not all(part[0].isupper() for part in name_parts):
                reasons.append("Name parts must be properly capitalized")
                self.last_validation_reasons = reasons
                return False
            boost = confidence_boosts.get("proper_case", 0.1)
            entity.confidence += boost
            reasons.append(f"Proper capitalization: +{boost}")
    
        if len(name_parts) >= 2:
            boost = confidence_boosts.get("multiple_words", 0.1)
            entity.confidence += boost
            reasons.append(f"Multiple words: +{boost}")
    
        self.last_validation_reasons = reasons
        return True

    def _validate_location(self, entity: Entity, text: str) -> bool:
        """
        Validate location entities using configuration-based rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool: Whether entity is valid
        """
        if not entity.text:
            return False
    
        location_config = self.config_loader.get_config("validation_params").get("location", {})
        us_states = location_config.get("us_states", {})
        confidence_boosts = location_config.get("confidence_boosts", {})
        confidence_penalties = location_config.get("confidence_penalties", {})
        validation_rules = location_config.get("validation_rules", {})
        format_patterns = location_config.get("format_patterns", {})
        
        text_lower = entity.text.lower()
        words = entity.text.split()
        reasons = []
        
        # Basic validation rules
        if len(entity.text) < validation_rules.get("min_chars", 2):
            reasons.append("Location too short")
            self.last_validation_reasons = reasons
            return False
                
        # Check capitalization requirements
        if validation_rules.get("require_capitalization", True):
            # Modified to handle non-ASCII uppercase characters
            first_char = entity.text[0]
            if not (first_char.isupper() or first_char != first_char.lower()):
                reasons.append("Location not properly capitalized")
                self.last_validation_reasons = reasons
                return False
    
        # Track matched formats for proper confidence adjustment
        formats_matched = []
                    
        # Apply format-based confidence adjustments
        self._check_location_formats(entity, formats_matched, format_patterns, 
                                   confidence_boosts, reasons)
                
        # Handle single-word locations
        if len(words) == 1:
            if not validation_rules.get("allow_single_word", True):
                reasons.append("Single word locations not allowed")
                self.last_validation_reasons = reasons
                return False
                    
            # Modified to better handle international cities
            if entity.text.isupper() and validation_rules.get("allow_all_caps", True):
                boost = confidence_boosts.get("proper_case", 0.1)
                entity.confidence += boost
                reasons.append(f"Valid ALL CAPS location: +{boost}")
            elif not formats_matched:
                # Only apply single word penalty if no other format was matched
                penalty = confidence_penalties.get("single_word", -0.2)
                entity.confidence += penalty
                reasons.append(f"Single word penalty: {penalty}")
                    
        # Multi-word boost (if not already counted in format matching)
        elif not formats_matched:
            boost = confidence_boosts.get("multi_word", 0.1)
            entity.confidence += boost
            reasons.append(f"Multi-word location: +{boost}")
                
        # Check for US state presence (if not already counted)
        if "us_city_state" not in formats_matched:
            state_lists = [
                s.lower() for s in us_states.get("full", []) + us_states.get("abbrev", [])
            ]
            if any(state.lower() in text_lower for state in state_lists):
                boost = confidence_boosts.get("has_state", 0.2)
                entity.confidence += boost
                reasons.append(f"Contains US state: +{boost}")
                    
        # If no format was matched but basic validation passed
        if not formats_matched and not reasons:
            reasons.append("Basic validation passed but no specific format matched")
                
        # Store validation reasons
        self.last_validation_reasons = reasons
        return True

    def _check_location_formats(self, entity: Entity, formats_matched: List[str],
                              format_patterns: Dict, confidence_boosts: Dict, 
                              reasons: List[str]) -> None:
        """
        Check location entity against format patterns.
        
        Args:
            entity: Entity to validate
            formats_matched: List to track matched formats
            format_patterns: Format patterns to check
            confidence_boosts: Confidence boost values
            reasons: List to store validation reasons
        """
        # Check US city-state format
        us_pattern = format_patterns.get("us_city_state", {}).get("regex", "")
        if us_pattern and re.match(us_pattern, entity.text):
            formats_matched.append("us_city_state")
            base_boost = confidence_boosts.get("known_format", 0.1)
            state_boost = confidence_boosts.get("has_state", 0.2)
            total_boost = base_boost + state_boost
            entity.confidence += total_boost
            reasons.append(f"Matches US city-state format: +{total_boost}")
                
            # Extra boost for having ZIP code
            if re.search(r'\d{5}(?:-\d{4})?$', entity.text):
                zip_boost = confidence_boosts.get("has_zip", 0.1)
                entity.confidence += zip_boost
                reasons.append(f"Includes ZIP code: +{zip_boost}")
                formats_matched.append("zip_code")
                    
        # Check international format
        intl_pattern = format_patterns.get("international", {}).get("regex", "")
        if intl_pattern and not formats_matched:
            # Modified international check to handle single words and diacritics
            if re.match(r'^[A-ZÀ-ÿ][A-Za-zÀ-ÿ\s-]*$', entity.text):
                formats_matched.append("international")
                boost = confidence_boosts.get("known_format", 0.1)
                entity.confidence += boost
                reasons.append(f"Matches international format: +{boost}")
                
        # Check metropolitan area format if no other format matched
        if not formats_matched:
            metro_pattern = format_patterns.get("metro_area", {}).get("regex", "")
            if metro_pattern and re.match(metro_pattern, entity.text):
                formats_matched.append("metro_area")
                boost = confidence_boosts.get("known_format", 0.1)
                entity.confidence += boost
                reasons.append(f"Matches metropolitan area format: +{boost}")

  ###############################################################################
# SECTION 6: EDUCATIONAL INSTITUTION VALIDATION
###############################################################################

    def _validate_educational_institution(self, entity: Entity, text: str) -> bool:
        """
        Validate educational institution entities using JSON configuration.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool: Whether the entity is valid
        """
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
    
        # 1) Single-word check: Exempt known acronyms, penalize others
        if len(words) < min_words:
            # Trim whitespace, check uppercase
            clean_text = entity.text.strip().upper()
            self.logger.warning("ACRONYM DEBUG - TEXT: '%s', CLEAN: '%s', known_acronyms=%s",
                    entity.text, clean_text, known_acronyms)
            # If it's a known acronym (MIT, UCLA, etc.), skip penalty
            if clean_text in known_acronyms:
                boost = confidence_boosts.get("known_acronym", 0.4)
                entity.confidence += boost
                self.last_validation_reasons.append(f"Boost for known acronym: {boost}")
                return True  # is_valid
            else:
                # Otherwise penalize
                penalty = confidence_penalties.get("single_word", -0.3)
                entity.confidence += penalty
                self.last_validation_reasons.append(f"Penalty for single word: {penalty}")
                return False  # is_valid

        # 2) Generic terms => penalty
        if any(term in text_lower for term in generic_terms):
            penalty = confidence_penalties.get("generic_term", -1.0)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for generic term: {penalty}")
            is_valid = False
    
        # 3) Missing core terms => penalty
        if not any(term in text_lower for term in core_terms):
            penalty = confidence_penalties.get("no_core_terms", -1.0)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for missing core terms: {penalty}")
            is_valid = False
    
        # 4) All caps => penalty if not a known acronym
        if entity.text.isupper() and entity.text not in known_acronyms:
            penalty = confidence_penalties.get("improper_caps", -0.2)
            entity.confidence += penalty
            self.last_validation_reasons.append(f"Penalty for improper caps: {penalty}")
            is_valid = False
    
        # 5) If we do have core terms AND enough words => boost
        #    (makes multi-word "University of X" get a nice bump)
        if any(term in text_lower for term in core_terms) and len(words) >= min_words:
            boost = confidence_boosts.get("full_name", 0.2)
            entity.confidence += boost
            self.last_validation_reasons.append(f"Boost for full name: {boost}")
            is_valid = True
    
        return is_valid

    def _check_edu_acronym(self, text: str, known_acronyms: Set[str]) -> bool:
        """
        Check if text is a valid educational institution acronym.
        
        Args:
            text: Text to check
            known_acronyms: Set of known valid acronyms
            
        Returns:
            bool: Whether text is a valid acronym
        """
        # Clean up text for comparison
        clean_text = text.strip().upper()
        
        if self.debug_mode:
            self.logger.debug(f"Checking acronym: '{clean_text}'")
            self.logger.debug(f"Known acronyms: {known_acronyms}")
        
        # Direct match with known acronyms
        if clean_text in known_acronyms:
            if self.debug_mode:
                self.logger.debug(f"Found direct acronym match: {clean_text}")
            return True
            
        return False

    def _process_edu_context(self, text: str, entity_text: str) -> Dict[str, Any]:
        """
        Process educational institution context for additional validation.
        
        Args:
            text: Full context text
            entity_text: Entity text to analyze
            
        Returns:
            Dict containing context analysis results
        """
        results = {
            "has_degree": False,
            "has_year": False,
            "has_department": False,
            "confidence_adjustment": 0.0
        }
        
        # Common degree indicators
        degree_terms = {
            "phd", "ph.d", "bachelor", "master", "bs", "ba", "ms", "ma", 
            "doctorate", "mba", "jd", "md"
        }
        
        # Department indicators
        department_terms = {
            "department", "dept", "school of", "college of", "faculty of",
            "institute of", "center for"
        }
        
        text_lower = text.lower()
        
        # Check for degree terms
        if any(term in text_lower for term in degree_terms):
            results["has_degree"] = True
            results["confidence_adjustment"] += 0.1
            
        # Check for years (rough pattern for years)
        if re.search(r'\b(19|20)\d{2}\b', text):
            results["has_year"] = True
            results["confidence_adjustment"] += 0.1
            
        # Check for department terms
        if any(term in text_lower for term in department_terms):
            results["has_department"] = True
            results["confidence_adjustment"] += 0.1
            
        if self.debug_mode:
            self.logger.debug(f"Education context results: {results}")
            
        return results

###############################################################################
# SECTION 7: PROTECTED CLASS VALIDATION 
###############################################################################

    def _validate_protected_class(self, entity: Entity, text: str) -> bool:
        """
        Validate protected class entities using linguistic analysis and configuration rules.
        
        Args:
            entity: Entity to validate
            text: Full text context
            
        Returns:
            bool: Whether the entity is valid
        """
        if not entity or not entity.text:
            return False
    
        # Load configurations
        protected_config = self.config_loader.get_config("validation_params").get("protected_class", {})
        validation_rules = protected_config.get("validation_rules", {})
        text_lower = entity.text.lower()
        context_lower = text.lower() if text else ""
        reasons = []
        score = 0
    
        # Get scoring values from config
        term_match_score = validation_rules.get("term_match_score", 0.3)
        context_match_score = validation_rules.get("context_match_score", 0.3)
        threshold = validation_rules.get("validation_threshold", 0.3)
    
        # First check pronoun patterns - these are self-validating
        pronoun_patterns = protected_config.get("categories", {}).get("pronoun_patterns", {})
        for pattern_name, pattern in pronoun_patterns.items():
            try:
                self.logger.debug(f"Checking pronoun pattern: {pattern} against text: {entity.text}")
                if re.search(pattern, entity.text, re.IGNORECASE):
                    score += term_match_score
                    reasons.append(f"Pronoun pattern match ({pattern_name}): +{term_match_score}")
                    entity.confidence += score
                    self.last_validation_reasons = reasons
                    self.logger.debug(f"Matched pronoun pattern: {pattern_name}")
                    return True
            except re.error:
                self.logger.warning(f"Invalid regex pattern: {pattern}")
                continue
    
        # Process with spaCy for non-pronoun cases
        doc = self.nlp(text)
        entity_doc = self.nlp(entity.text)
    
        # Debug full spaCy parse
        self._log_spacy_parse_debug(doc)
    
        # Load all identity terms from config
        gender_patterns = protected_config.get("categories", {}).get("gender_patterns", {})
        religious_patterns = protected_config.get("categories", {}).get("religious_patterns", {})
        orientation_terms = protected_config.get("categories", {}).get("orientation", [])
        race_terms = protected_config.get("categories", {}).get("race_ethnicity", [])
        
        all_identity_terms = (
            gender_patterns.get("gender_terms", []) +
            religious_patterns.get("religious_terms", []) +
            orientation_terms +
            race_terms
        )
    
        # Find matched identity term
        matched_term = self._find_matched_identity_term(entity_doc, all_identity_terms)
    
        if matched_term:
            # Get valid contexts from config
            valid_contexts = self._get_valid_contexts(gender_patterns, religious_patterns)
            
            self.logger.debug(f"Looking for valid contexts: {valid_contexts}")
            self.logger.debug(f"In text: {text}")
    
            # Check noun chunks first
            if self._validate_noun_chunks(doc, matched_term, valid_contexts, term_match_score, context_match_score, reasons, entity):
                return True
    
            # If not found in chunks, check dependency chains
            if self._validate_dependency_chains(doc, matched_term, valid_contexts, term_match_score, context_match_score, reasons, entity):
                return True
    
        # Apply penalty if no valid matches found
        penalty = protected_config.get("confidence_penalties", {}).get("no_context", -0.4)
        entity.confidence += penalty
        reasons.append(f"No protected class context: {penalty}")
        self.last_validation_reasons = reasons
        return False

    def _log_spacy_parse_debug(self, doc) -> None:
        """Log detailed spaCy parse information for debugging."""
        self.logger.debug("\nFull SpaCy Parse:")
        self.logger.debug("Text tokens:")
        for token in doc:
            self.logger.debug(f"Token: {token.text}")
            self.logger.debug(f"  Dep: {token.dep_}")
            self.logger.debug(f"  Head: {token.head.text}")
            self.logger.debug(f"  Children: {[child.text for child in token.children]}")
            self.logger.debug(f"  Left children: {[child.text for child in token.lefts]}")
            self.logger.debug(f"  Right children: {[child.text for child in token.rights]}")
            self.logger.debug(f"  Ancestors: {[ancestor.text for ancestor in token.ancestors]}")
    
        self.logger.debug("\nNoun chunks:")
        for chunk in doc.noun_chunks:
            self.logger.debug(f"Chunk: {chunk.text}")
            self.logger.debug(f"  Root: {chunk.root.text}")
            self.logger.debug(f"  Root dep: {chunk.root.dep_}")
            self.logger.debug(f"  Root head: {chunk.root.head.text}")
            
            # Log compounds and modifiers
            compounds = [t.text for t in chunk.root.children if t.dep_ == 'compound']
            modifiers = [t.text for t in chunk.root.children if t.dep_ in ['amod', 'advmod']]
            self.logger.debug(f"  Compounds: {compounds}")
            self.logger.debug(f"  Modifiers: {modifiers}")

    def _find_matched_identity_term(self, entity_doc, all_identity_terms: List[str]) -> Optional[str]:
        """Find matching identity term in entity text."""
        matched_term = None
        for token in entity_doc:
            term_text = token.text.lower()
            lemma_text = token.lemma_.lower()
            self.logger.debug(f"Checking term: {term_text} (lemma: {lemma_text})")
            if term_text in all_identity_terms or lemma_text in all_identity_terms:
                matched_term = term_text
                self.logger.debug(f"Matched identity term: {matched_term}")
                break
        return matched_term

    def _get_valid_contexts(self, gender_patterns: Dict, religious_patterns: Dict) -> List[str]:
        """Get list of valid context terms from configuration."""
        return (
            gender_patterns.get("identity_context", []) +
            religious_patterns.get("identity_context", []) +
            religious_patterns.get("practice_context", []) +
            gender_patterns.get("org_context", []) +
            religious_patterns.get("org_context", []) +
            religious_patterns.get("leadership_roles", [])
        )

    def _validate_noun_chunks(self, doc, matched_term: str, valid_contexts: List[str],
                            term_match_score: float, context_match_score: float,
                            reasons: List[str], entity: Entity) -> bool:
        """Validate entity using noun chunk analysis."""
        for chunk in doc.noun_chunks:
            chunk_text = chunk.text.lower()
            if matched_term in chunk_text:
                self.logger.debug(f"Found matched term in noun chunk: {chunk_text}")
                self.logger.debug(f"Chunk root: {chunk.root.text}")
                self.logger.debug(f"Chunk root dep: {chunk.root.dep_}")
                self.logger.debug(f"Chunk root head: {chunk.root.head.text}")
    
                # Log all relationships in the chunk
                for token in chunk:
                    self.logger.debug(f"Token in chunk: {token.text}")
                    self.logger.debug(f"  Dep to root: {token.dep_}")
                    self.logger.debug(f"  Head: {token.head.text}")
                    self.logger.debug(f"  Dependencies: {[(child.text, child.dep_) for child in token.children]}")
    
                if self._check_chunk_context(chunk, valid_contexts, term_match_score,
                                          context_match_score, reasons, entity):
                    return True
        return False

    def _check_chunk_context(self, chunk, valid_contexts: List[str],
                            term_match_score: float, context_match_score: float,
                            reasons: List[str], entity: Entity) -> bool:
        """Check context of a noun chunk for validation."""
        if chunk.root.dep_ in ['dobj', 'pobj', 'nsubj', 'attr']:
            head_text = chunk.root.head.text.lower()
            if head_text in valid_contexts:
                self.logger.debug(f"Found context match through chunk head: {head_text}")
                score = term_match_score + context_match_score
                reasons.append(f"Found context through head: +{score}")
                entity.confidence += score
                self.last_validation_reasons = reasons
                return True
    
        # Check children of the chunk root
        for token in chunk.root.children:
            token_text = token.text.lower()
            if token_text in valid_contexts:
                self.logger.debug(f"Found context match through chunk child: {token_text}")
                score = term_match_score + context_match_score
                reasons.append(f"Found context through child: +{score}")
                entity.confidence += score
                self.last_validation_reasons = reasons
                return True
        return False

    def _validate_dependency_chains(self, doc, matched_term: str, valid_contexts: List[str],
                                  term_match_score: float, context_match_score: float,
                                  reasons: List[str], entity: Entity) -> bool:
        """Validate entity using dependency chain analysis."""
        for token in doc:
            token_text = token.text.lower()
            if token_text in valid_contexts:
                self.logger.debug(f"Found context match: {token_text}")
                
                # Build dependency chain
                token_chain = []
                current = token
                while current and len(token_chain) < 5:
                    token_chain.append(current)
                    current = current.head
                self.logger.debug(f"Dependency chain: {[t.text for t in token_chain]}")
                
                if self._check_chain_matches(token_chain, matched_term, term_match_score,
                                           context_match_score, reasons, entity):
                    return True
        return False

    def _check_chain_matches(self, token_chain: List[Any], matched_term: str,
                            term_match_score: float, context_match_score: float,
                            reasons: List[str], entity: Entity) -> bool:
        """Check for matches in a dependency chain."""
        for chain_token in token_chain:
            chain_text = chain_token.text.lower()
            if chain_text == matched_term:
                self.logger.debug(f"Found match in chain at {chain_token.text}")
                score = term_match_score + context_match_score
                reasons.append(f"Found context through chain: +{score}")
                entity.confidence += score
                self.last_validation_reasons = reasons
                return True

            # Check immediate children and grandchildren
            for child in chain_token.children:
                child_text = child.text.lower()
                if child_text == matched_term:
                    self.logger.debug(f"Found match in child of {chain_token.text}")
                    score = term_match_score + context_match_score
                    reasons.append(f"Found context through child: +{score}")
                    entity.confidence += score
                    self.last_validation_reasons = reasons
                    return True
                
                for grandchild in child.children:
                    grandchild_text = grandchild.text.lower()
                    if grandchild_text == matched_term:
                        self.logger.debug(f"Found match in grandchild of {chain_token.text}")
                        score = term_match_score + context_match_score
                        reasons.append(f"Found context through grandchild: +{score}")
                        entity.confidence += score
                        self.last_validation_reasons = reasons
                        return True
        return False

###############################################################################
# SECTION 8: PHI AND MEDICAL VALIDATION
###############################################################################

    def _validate_phi(self, entity: Entity, text: str) -> bool:
        """
        Validate PHI entities using advanced spaCy linguistic analysis and configuration rules.
        
        Args:
            entity: Entity to validate
            text: Original text context
            
        Returns:
            bool: Whether entity is valid
        """
        if not entity or not entity.text:
            self.logger.debug("Empty entity or text")
            return False
    
        # Load PHI configuration
        phi_config = self.config_loader.get_config("validation_params").get("phi", {})
        text_lower = entity.text.lower()
        reasons = []
        total_boost = 0
    
        self.logger.debug(f"Starting PHI validation for entity: '{entity.text}'")
        self.logger.debug(f"Context text: '{text}'")
        
        # Process with spaCy for linguistic analysis
        doc = self.nlp(text)
        entity_doc = self.nlp(entity.text)
        entity_start = text.find(entity.text)
        if entity_start == -1:  # Handle case-insensitive match
            entity_start = text.lower().find(text_lower)
            self.logger.debug(f"Used case-insensitive match to find entity at position {entity_start}")
    
        # Find the entity tokens
        entity_tokens = self._get_entity_tokens(doc, entity_start, entity.text)
    
        if not entity_tokens:
            self.logger.warning(f"Could not find entity tokens for '{entity.text}'")
            return False
    
        self.logger.debug("Entity tokens found:")
        for token in entity_tokens:
            self.logger.debug(
                f"  Token: '{token.text}', POS: {token.pos_}, "
                f"Dep: {token.dep_}, Head: '{token.head.text}'"
            )

        # Run all validation checks
        dep_match = self._check_dependency_patterns(doc, entity_tokens, phi_config, total_boost, reasons)
        chunk_match = self._analyze_noun_chunks(doc, entity_tokens, phi_config, total_boost, reasons)
        treatment_match = self._check_treatment_patterns(entity_tokens, phi_config, total_boost, reasons)
        context_match = self._analyze_medical_context(entity_tokens, phi_config, total_boost, reasons)
        veteran_match = self._analyze_veteran_disability(doc, entity_tokens, phi_config, total_boost, reasons)
        ada_match = self._analyze_ada_accommodations(doc, phi_config, total_boost, reasons)

        # Validate based on combined linguistic evidence
        is_valid = any([dep_match, chunk_match, treatment_match, context_match, 
                       veteran_match, ada_match])
        
        self.logger.debug(f"\nCombined validation result: is_valid={is_valid}")
        self.logger.debug(f"Total confidence boost: {total_boost}")
        
        # Handle generic terms with strict requirements
        if text_lower in phi_config.get("generic_reject_terms", []):
            self.logger.debug("\nGeneric term detected, applying strict validation")
            if not treatment_match:  # Skip strict validation if it's a clear treatment reference
                is_valid = self._validate_generic_term(doc, entity_tokens, phi_config, 
                                                     total_boost, reasons, entity)

        # Apply final confidence adjustment
        if is_valid:
            entity.confidence += total_boost
            self.logger.debug(f"Validation successful - Final confidence: {entity.confidence}")
        else:
            penalty = phi_config.get("confidence_penalties", {}).get("no_context", -0.7)
            entity.confidence += penalty
            reasons.append(f"No valid medical context found: {penalty}")
            self.logger.debug(
                f"Validation failed - Applied penalty: {penalty}, "
                f"Final confidence: {entity.confidence}"
            )
    
        self.last_validation_reasons = reasons
        self.logger.debug("\nValidation reasons:")
        for reason in reasons:
            self.logger.debug(f"  {reason}")
            
        return is_valid

    def _get_entity_tokens(self, doc, entity_start: int, entity_text: str) -> List[Any]:
        """Find tokens corresponding to entity in document."""
        entity_tokens = []
        for token in doc:
            if token.idx >= entity_start and token.idx < entity_start + len(entity_text):
                entity_tokens.append(token)
        return entity_tokens

    def _check_dependency_patterns(self, doc, entity_tokens: List[Any], phi_config: Dict,
                                 total_boost: float, reasons: List[str]) -> bool:
        """Check medical dependency patterns."""
        medical_verbs = set(phi_config.get("medical_verbs", {
            'treat', 'diagnose', 'prescribe', 'administer', 'receive',
            'manage', 'suffer', 'experience', 'undergo', 'require'
        }))
        medical_context_found = False
        
        self.logger.debug("Checking dependency patterns...")
        
        for token in entity_tokens:
            self.logger.debug(f"\nAnalyzing token: '{token.text}'")
            self.logger.debug(f"Head: '{token.head.text}', Dependency: {token.dep_}")
            
            # Check ancestors
            ancestors = list(token.ancestors)
            self.logger.debug(f"Ancestors: {[t.text for t in ancestors]}")
            if self._check_ancestor_patterns(ancestors, medical_verbs, phi_config, 
                                          total_boost, reasons):
                medical_context_found = True

            # Check children
            children = list(token.children)
            self.logger.debug(f"Children: {[t.text for t in children]}")
            if self._check_child_patterns(children, token, phi_config, total_boost, reasons):
                medical_context_found = True

        self.logger.debug(f"Dependency check result: medical_context_found={medical_context_found}")
        return medical_context_found

    def _check_ancestor_patterns(self, ancestors: List[Any], medical_verbs: Set[str],
                               phi_config: Dict, total_boost: float, 
                               reasons: List[str]) -> bool:
        """Check medical patterns in ancestor tokens."""
        for ancestor in ancestors:
            self.logger.debug(f"Checking ancestor: '{ancestor.text}', lemma: '{ancestor.lemma_}'")
            if ancestor.lemma_ in medical_verbs:
                boost = phi_config.get("confidence_boosts", {}).get("treatment_context_match", 0.3)
                reason = f"Medical verb dependency ({ancestor.lemma_}): +{boost}"
                self.logger.debug(f"Found medical verb ancestor: {reason}")
                reasons.append(reason)
                total_boost += boost
                return True
        return False

    def _check_child_patterns(self, children: List[Any], token: Any, phi_config: Dict,
                            total_boost: float, reasons: List[str]) -> bool:
        """Check medical patterns in child tokens."""
        context_words = set(phi_config.get("context_words", []))
        for child in children:
            self.logger.debug(f"Checking child: '{child.text}', dep: {child.dep_}")
            if child.dep_ in {'prep', 'dobj', 'pobj'} and child.text.lower() in context_words:
                boost = phi_config.get("confidence_boosts", {}).get("has_medical_context", 0.2)
                reason = f"Medical context dependency ({token.text} → {child.text}): +{boost}"
                self.logger.debug(f"Found medical context child: {reason}")
                reasons.append(reason)
                total_boost += boost
                return True
        return False

    def _validate_generic_term(self, doc, entity_tokens: List[Any], phi_config: Dict,
                             total_boost: float, reasons: List[str], entity: Entity) -> bool:
        """Validate generic medical terms with strict requirements."""
        required_context = phi_config.get("generic_term_context_requirements", {})
        required_words = set(required_context.get("required_context_words", []))
        min_required = required_context.get("minimum_context_words", 2)
        
        # Use spaCy's dependency parsing to find related context words
        context_matches = sum(1 for token in doc if (
            token.text.lower() in required_words and
            any(t in entity_tokens for t in list(token.children) + list(token.ancestors))
        ))
        
        self.logger.debug(
            f"Context word matches found: {context_matches}, "
            f"minimum required: {min_required}"
        )
        
        if context_matches < min_required:
            penalty = phi_config.get("confidence_penalties", {}).get("generic_medical_term", -0.7)
            total_boost += penalty
            reasons.append(f"Generic term without sufficient context: {penalty}")
            self.logger.debug(f"Insufficient context - applying penalty: {penalty}")
            self.last_validation_reasons = reasons
            entity.confidence += penalty
            return False
            
        return True

    def _analyze_medical_context(self, entity_tokens: List[Any], phi_config: Dict,
                               total_boost: float, reasons: List[str]) -> bool:
        """Analyze medical context using linguistic features."""
        medical_conditions = phi_config.get("categories", {}).get("medical_conditions", {})
        condition_found = False
        
        self.logger.debug("\nAnalyzing medical context...")
        
        # Flatten condition lists from config
        all_conditions = {
            condition.lower() 
            for category in medical_conditions.values() 
            for condition in category
        }
        self.logger.debug(f"Loaded {len(all_conditions)} medical conditions from config")
    
        for token in entity_tokens:
            # Get token's neighborhood in dependency tree
            context_tokens = list(token.children) + list(token.ancestors)
            
            self.logger.debug(f"\nAnalyzing context for token: '{token.text}'")
            self.logger.debug(f"Context tokens: {[t.text for t in context_tokens]}")

            for context_token in context_tokens:
                if self._check_medical_token(context_token, all_conditions, phi_config,
                                          total_boost, reasons):
                    condition_found = True
                    break
    
        self.logger.debug(f"Medical context analysis result: condition_found={condition_found}")
        return condition_found

    def _check_medical_token(self, token: Any, all_conditions: Set[str], phi_config: Dict,
                           total_boost: float, reasons: List[str]) -> bool:
        """Check individual token for medical context."""
        self.logger.debug(f"\nChecking context token: '{token.text}'")
        self.logger.debug(f"POS: {token.pos_}, Dep: {token.dep_}, Lemma: {token.lemma_}")
        
        # Check for condition mentions
        if token.text.lower() in all_conditions:
            boost = phi_config.get("confidence_boosts", {}).get("condition_specific_match", 0.4)
            reason = f"Medical condition context ({token.text}): +{boost}"
            self.logger.debug(f"Found medical condition: {reason}")
            reasons.append(reason)
            total_boost += boost
            return True

        # Check for diagnostic/symptomatic language patterns
        diagnostic_patterns = [
            (token.lemma_ in {'diagnose', 'present', 'show'} and 
             any(c.dep_ == 'prep' for c in token.children)),
            (token.dep_ == 'prep' and token.head.lemma_ in {'suffer', 'experience'})
        ]
        
        for i, is_match in enumerate(diagnostic_patterns):
            if is_match:
                boost = phi_config.get("confidence_boosts", {}).get("diagnostic_precision", 0.3)
                reason = f"Diagnostic context ({token.lemma_}): +{boost}"
                self.logger.debug(f"Found diagnostic pattern {i+1}: {reason}")
                reasons.append(reason)
                total_boost += boost
                return True
                
        return False

###############################################################################
# SECTION 9: UTILITY METHODS
###############################################################################

    def _analyze_noun_chunks(self, doc, entity_tokens: List[Any], phi_config: Dict,
                           total_boost: float, reasons: List[str]) -> bool:
        """
        Analyze noun chunks for medical phrases.
        
        Args:
            doc: spaCy Doc object
            entity_tokens: List of entity tokens
            phi_config: PHI configuration dictionary
            total_boost: Running total of confidence adjustments
            reasons: List to store validation reasons
            
        Returns:
            bool: Whether valid medical context was found
        """
        medical_terms = set(phi_config.get("context_words", []))
        chunk_match_found = False
        
        self.logger.debug("\nAnalyzing noun chunks...")
        
        for chunk in doc.noun_chunks:
            self.logger.debug(f"\nExamining chunk: '{chunk.text}'")
            self.logger.debug(f"Root: '{chunk.root.text}', Root dep: {chunk.root.dep_}")
            
            # Check if our entity is part of this chunk
            chunk_tokens = list(chunk)
            entity_in_chunk = any(token in entity_tokens for token in chunk_tokens)
            self.logger.debug(f"Entity in chunk: {entity_in_chunk}")
            
            if entity_in_chunk:
                # Look for medical modifiers
                modifiers = [token for token in chunk if token.dep_ in {'amod', 'compound'}]
                self.logger.debug(f"Modifiers found: {[(t.text, t.dep_) for t in modifiers]}")
                
                medical_modifiers = [mod for mod in modifiers if mod.text.lower() in medical_terms]
                if medical_modifiers:
                    boost = phi_config.get("confidence_boosts", {}).get("specialized_terminology", 0.3)
                    reason = f"Medical noun phrase ({chunk.text}): +{boost}"
                    self.logger.debug(f"Found medical modifier: {reason}")
                    reasons.append(reason)
                    total_boost += boost
                    chunk_match_found = True
    
        self.logger.debug(f"Noun chunk analysis result: chunk_match_found={chunk_match_found}")
        return chunk_match_found

    def _analyze_ada_accommodations(self, doc, phi_config: Dict, total_boost: float,
                                  reasons: List[str]) -> bool:
        """
        Analyze ADA-related accommodation patterns using dependency parsing.
        
        Args:
            doc: spaCy Doc object
            phi_config: PHI configuration dictionary
            total_boost: Running total of confidence adjustments
            reasons: List to store validation reasons
            
        Returns:
            bool: Whether valid ADA accommodation was found
        """
        accommodation_found = False
        
        self.logger.debug("\nAnalyzing ADA accommodation patterns...")
        
        # Get disability patterns from config
        disabilities = phi_config.get("categories", {}).get("disabilities", {})
        ada_patterns = disabilities.get("ada_related", [])
        
        self.logger.debug(f"Loaded ADA accommodation patterns: {ada_patterns}")
        
        # Check for basic accommodation terms first
        accommodation_terms = {'accommodation', 'accommodations', 'ada', 'disability'}
        context_terms = {'need', 'needed', 'require', 'required', 'requesting', 'disability'}
        
        # Check full context
        full_context = ' '.join(t.text.lower() for t in doc)
        
        # First check exact patterns
        for pattern in ada_patterns:
            if pattern in full_context:
                boost = phi_config.get("confidence_boosts", {}).get("has_medical_context", 0.4)
                reason = f"ADA accommodation pattern ({pattern}): +{boost}"
                self.logger.debug(f"Found ADA pattern: {reason}")
                reasons.append(reason)
                total_boost += boost
                accommodation_found = True
                break
        
        # Then check for accommodation term + context term combinations
        if not accommodation_found:
            words = set(t.text.lower() for t in doc)
            if any(a in words for a in accommodation_terms) and any(c in words for c in context_terms):
                boost = phi_config.get("confidence_boosts", {}).get("has_medical_context", 0.4)
                reason = f"Accommodation context pattern: +{boost}"
                self.logger.debug(f"Found accommodation context pattern")
                reasons.append(reason)
                total_boost += boost
                accommodation_found = True
        
        self.logger.debug(f"ADA accommodation analysis result: accommodation_found={accommodation_found}")
        return accommodation_found

    def _normalize_text(self, text: str) -> str:
        """
        Normalize text for comparison.
        
        Args:
            text: Text to normalize
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
            
        # Basic normalization - lowercase and strip whitespace
        normalized = text.lower().strip()
        
        # Remove all punctuation except .-_ and space
        normalized = ''.join(
            c for c in normalized 
            if c.isalnum() or c in '.-_ '
        )
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized

    def _check_term_match(self, text: str, terms: List[str]) -> bool:
        """
        Check if text matches any terms in list.
        
        Args:
            text: Text to check
            terms: List of terms to match against
            
        Returns:
            bool: Whether a match was found
        """
        text_lower = text.lower()
        for term in terms:
            if term.lower() in text_lower:
                return True
        return False

    def _get_context_window(self, text: str, entity_start: int, entity_end: int,
                           window_size: int = 50) -> str:
        """
        Get text context window around entity.
        
        Args:
            text: Full text
            entity_start: Start position of entity
            entity_end: End position of entity
            window_size: Size of context window
            
        Returns:
            str: Context window text
        """
        start = max(0, entity_start - window_size)
        end = min(len(text), entity_end + window_size)
        return text[start:end]

    def is_generic_medical_term(self, text_lower: str, doc) -> bool:
        """
        Check if text is a generic medical term requiring more context.
        
        Args:
            text_lower: Lowercase text to check
            doc: spaCy Doc object
            
        Returns:
            bool: Whether text is a generic medical term
        """
        # List of terms that require strong medical context
        strict_generic_terms = {'health', 'medical', 'condition', 'treatment'}
        
        # If the entire term is a strict generic term
        if text_lower in strict_generic_terms:
            # Count strong medical context words
            context_words = {
                'diagnosed', 'treatment', 'condition', 'illness', 
                'therapy', 'prescription', 'medical', 'healthcare', 
                'clinical', 'diagnosis', 'disability'
            }
            
            # Check tokens and their lemmas
            context_matches = sum(
                1 for token in doc 
                if (token.text.lower() in context_words or 
                    token.lemma_.lower() in context_words)
            )
            
            # Require multiple context words
            return context_matches < 2
        
        return False

