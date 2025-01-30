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
            ensemble_config = routing_config["routing"].get("ensemble_required", {})
            
            # Basic entities from spacy_primary
            self.entity_types = set(spacy_config.get("entities", []))
            self.entity_mappings = spacy_config.get("mappings", {})
            
            # Get thresholds from configs
            base_thresholds = spacy_config.get("thresholds", {})
            ensemble_thresholds = ensemble_config.get("confidence_thresholds", {})
            
            # Initialize thresholds
            self.confidence_thresholds = {}
            
            # Add base thresholds
            for entity, threshold in base_thresholds.items():
                self.confidence_thresholds[entity] = threshold
                
            # Add ensemble thresholds - lower for contextual detection
            for entity, config in ensemble_thresholds.items():
                self.confidence_thresholds[entity] = 0.6  # Base confidence for contextual
    
            self.logger.debug(f"Loaded spaCy entity types: {self.entity_types}")
            self.logger.debug(f"Entity mappings: {self.entity_mappings}")
            self.logger.debug(f"Confidence thresholds: {self.confidence_thresholds}")
    
            return True
    
        except Exception as e:
            self.logger.error(f"Error validating spaCy config: {str(e)}")
            return False

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        if not entity or not entity.text or len(entity.text.strip()) == 0:
            self.logger.debug(f"Empty or invalid entity text")
            return False
    
        self.last_validation_reasons = []
        threshold = self.confidence_thresholds.get(entity.entity_type, 0.75)
        
        if entity.confidence < threshold:
            reason = f"Entity '{entity.text}' confidence below threshold: {entity.confidence} < {threshold}"
            self.logger.debug(reason)
            self.last_validation_reasons.append(reason)
            return False
    
        is_valid = False
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
    
        if entity.entity_type in {"EDUCATIONAL_INSTITUTION", "ORGANIZATION"}:
            if len(entity.text.split()) < 2:
                self.last_validation_reasons.append(f"Rejected single-word entity: {entity.text}")
                return False
    
        return True

    def _validate_person(self, entity: Entity, text: str) -> bool:
            """Validate PERSON entities using configuration-based rules."""
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
            """Validate location entities using configuration-based rules."""
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
            
            # 1. Basic validation rules
            if len(entity.text) < validation_rules.get("min_chars", 2):
                reasons.append("Location too short")
                self.last_validation_reasons = reasons
                return False
                
            # 2. Check capitalization requirements
            if validation_rules.get("require_capitalization", True):
                # Modified to handle non-ASCII uppercase characters
                first_char = entity.text[0]
                if not (first_char.isupper() or first_char != first_char.lower()):
                    reasons.append("Location not properly capitalized")
                    self.last_validation_reasons = reasons
                    return False
    
            # Track matched formats for proper confidence adjustment
            formats_matched = []
                    
            # 3. Apply format-based confidence adjustments
            import re
            
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
                
            # 4. Handle single-word locations
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
                    
            # 5. Multi-word boost (if not already counted in format matching)
            elif not formats_matched:
                boost = confidence_boosts.get("multi_word", 0.1)
                entity.confidence += boost
                reasons.append(f"Multi-word location: +{boost}")
                
            # 6. Check for US state presence (if not already counted)
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
    
        # 1) Single-word check: Exempt known acronyms, penalize others
        if len(words) < min_words:
            # Trim whitespace, check uppercase
            clean_text = entity.text.strip().upper()
            self.logger.warning("ACRONYM DEBUG - TEXT: '%s', CLEAN: '%s', known_acronyms=%s",
                    entity.text, clean_text, known_acronyms)
            # If it’s a known acronym (MIT, UCLA, etc.), skip penalty
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

##########Validate_Protected_Class##############    

    def _validate_protected_class(self, entity: Entity, text: str) -> bool:
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
        matched_term = None
        for token in entity_doc:
            term_text = token.text.lower()
            lemma_text = token.lemma_.lower()
            self.logger.debug(f"Checking term: {term_text} (lemma: {lemma_text})")
            if term_text in all_identity_terms or lemma_text in all_identity_terms:
                matched_term = term_text
                self.logger.debug(f"Matched identity term: {matched_term}")
                break
    
        if matched_term:
            # Get valid contexts from config
            valid_contexts = (
                gender_patterns.get("identity_context", []) +
                religious_patterns.get("identity_context", []) +
                religious_patterns.get("practice_context", []) +
                gender_patterns.get("org_context", []) +
                religious_patterns.get("org_context", []) +
                religious_patterns.get("leadership_roles", [])
            )
            
            self.logger.debug(f"Looking for valid contexts: {valid_contexts}")
            self.logger.debug(f"In text: {text}")
    
            # Check noun chunks first
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
    
                    # Check context words connected to this chunk
                    if chunk.root.dep_ in ['dobj', 'pobj', 'nsubj', 'attr']:
                        head_text = chunk.root.head.text.lower()
                        if head_text in valid_contexts:
                            self.logger.debug(f"Found context match through chunk head: {head_text}")
                            score += (term_match_score + context_match_score)
                            reasons.append(f"Found context through head: +{term_match_score + context_match_score}")
                            entity.confidence += score
                            self.last_validation_reasons = reasons
                            return True
    
                    # Check children of the chunk root
                    for token in chunk.root.children:
                        token_text = token.text.lower()
                        if token_text in valid_contexts:
                            self.logger.debug(f"Found context match through chunk child: {token_text}")
                            score += (term_match_score + context_match_score)
                            reasons.append(f"Found context through child: +{term_match_score + context_match_score}")
                            entity.confidence += score
                            self.last_validation_reasons = reasons
                            return True
    
            # If not found in chunks, check dependency chains
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
    
                    # Check each token in chain and its children
                    for chain_token in token_chain:
                        chain_text = chain_token.text.lower()
                        if chain_text == matched_term:
                            self.logger.debug(f"Found match in chain at {chain_token.text}")
                            score += (term_match_score + context_match_score)
                            reasons.append(f"Found context through chain: +{term_match_score + context_match_score}")
                            entity.confidence += score
                            self.last_validation_reasons = reasons
                            return True
    
                        # Check immediate children
                        for child in chain_token.children:
                            child_text = child.text.lower()
                            if child_text == matched_term:
                                self.logger.debug(f"Found match in child of {chain_token.text}")
                                score += (term_match_score + context_match_score)
                                reasons.append(f"Found context through child: +{term_match_score + context_match_score}")
                                entity.confidence += score
                                self.last_validation_reasons = reasons
                                return True
                            
                            # Check grandchildren
                            for grandchild in child.children:
                                grandchild_text = grandchild.text.lower()
                                if grandchild_text == matched_term:
                                    self.logger.debug(f"Found match in grandchild of {chain_token.text}")
                                    score += (term_match_score + context_match_score)
                                    reasons.append(f"Found context through grandchild: +{term_match_score + context_match_score}")
                                    entity.confidence += score
                                    self.last_validation_reasons = reasons
                                    return True
    
        # Apply penalty if no valid matches found
        penalty = protected_config.get("confidence_penalties", {}).get("no_context", -0.4)
        entity.confidence += penalty
        reasons.append(f"No protected class context: {penalty}")
        self.last_validation_reasons = reasons
        return False


##########Validate_PHI############## 
    
    def _validate_phi(self, entity: Entity, text: str) -> bool:
        """
        Validate PHI entities using configuration-based rules with improved context handling.
        """
        if not entity or not entity.text:
            return False
    
        # Load PHI configuration
        phi_config = self.config_loader.get_config("validation_params").get("phi", {})
        text_lower = entity.text.lower()
        context_lower = text.lower() if text else ""
        reasons = []
        total_boost = 0
        is_valid = False
    
        # 1. Check if it's an always-valid condition first
        always_valid = phi_config.get("validation_rules", {}).get("always_valid_conditions", [])
        if any(condition.lower() in text_lower for condition in always_valid):
            boost = phi_config.get("confidence_boosts", {}).get("condition_specific_match", 0.4)
            total_boost += boost
            reasons.append(f"Always valid condition match: +{boost}")
            is_valid = True
        
        # 2. Process with spaCy for linguistic analysis
        doc = self.nlp(text)
        entity_doc = self.nlp(entity.text)
    
        # 3. Check for medical conditions categories
        medical_conditions = phi_config.get("categories", {}).get("medical_conditions", {})
        for category, conditions in medical_conditions.items():
            if any(condition.lower() in text_lower for condition in conditions):
                boost = phi_config.get("confidence_boosts", {}).get("condition_specific_match", 0.4)
                total_boost += boost
                reasons.append(f"Medical condition match ({category}): +{boost}")
                is_valid = True
                break
    
        # 4. Check for treatment matches
        treatments = phi_config.get("categories", {}).get("treatments", {})
        for category, treatment_list in treatments.items():
            if any(treatment.lower() in text_lower for treatment in treatment_list):
                boost = phi_config.get("confidence_boosts", {}).get("treatment_context_match", 0.4)
                total_boost += boost
                reasons.append(f"Treatment match ({category}): +{boost}")
                is_valid = True
                break
    
        # 5. Check for disability-related content
        disabilities = phi_config.get("categories", {}).get("disabilities", {})
        for category, disability_list in disabilities.items():
            if any(disability.lower() in text_lower for disability in disability_list):
                boost = phi_config.get("confidence_boosts", {}).get("condition_specific_match", 0.4)
                total_boost += boost
                reasons.append(f"Disability match ({category}): +{boost}")
                is_valid = True
                break
    
        # 6. Pattern matching
        patterns = phi_config.get("patterns", {})
        for pattern_type, pattern_config in patterns.items():
            regex_patterns = pattern_config.get("regex", [])
            if not isinstance(regex_patterns, list):
                regex_patterns = [regex_patterns]
            
            for pattern in regex_patterns:
                try:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        boost = pattern_config.get("boost", 0.3)
                        total_boost += boost
                        reasons.append(f"Pattern match ({pattern_type}): +{boost}")
                        is_valid = True
                except re.error:
                    self.logger.warning(f"Invalid regex pattern: {pattern}")
                    continue
    
        # 7. Context word validation
        context_words = set(phi_config.get("context_words", []))
        context_matches = sum(1 for token in doc if token.text.lower() in context_words)
        
        if context_matches >= phi_config.get("validation_rules", {}).get("min_context_words", 0):
            boost = phi_config.get("confidence_boosts", {}).get("has_medical_context", 0.3)
            total_boost += boost
            reasons.append(f"Context words match ({context_matches}): +{boost}")
            is_valid = True
    
        # 8. Check for generic terms that require additional context
        # Step 8: Check for generic terms that require additional context
        #
        # 1) Only proceed if this entity text is in generic_reject_terms
        #    e.g., "medication", "treatment", "condition", "health", etc.
        
        generic_terms = set(phi_config.get("generic_reject_terms", []))
        entity_text_lower = entity.text.lower()
        
        if entity_text_lower in generic_terms:
            # Load config for required context
            required_config = phi_config.get("generic_term_context_requirements", {})
            required_words = set(required_config.get("required_context_words", []))
            min_required = required_config.get("minimum_context_words", 2)
        
            penalty = phi_config.get("confidence_penalties", {}).get("generic_medical_term", -0.7)
            
            # 2) If your context has been loaded into 'doc', 
            #    ensure it's from the *entire* 'context' string, NOT just the entity.text
            #    e.g., doc = self.nlp(context_string)
        
            # 3) Attempt to find the spaCy sentence containing this entity.
            entity_span = None
            for sent in doc.sents:
                # If this sentence covers the entity's char range:
                if sent.start_char <= entity.start and sent.end_char >= entity.end:
                    entity_span = sent
                    break
            
            # If spaCy didn't split sentences or we didn't find one, fallback to entire doc
            if not entity_span:
                entity_span = doc
            
            # 4) Count how many "required context words" appear in that sentence (or doc).
            #    We do BOTH text.lower() & lemma_.lower() to catch morphological variants.
            context_matches = 0
            for token in entity_span:
                t_lower = token.text.lower()
                l_lower = token.lemma_.lower()
        
                if t_lower in required_words or l_lower in required_words:
                    context_matches += 1
        
            # 5) If we found fewer than min_required matches, penalize & set is_valid=False
            if context_matches < min_required:
                total_boost += penalty
                reasons.append(f"Generic term without sufficient context: {penalty}")
                is_valid = False


    
        9. # Apply final confidence adjustment
        if is_valid:
            entity.confidence += total_boost
        else:
            penalty = phi_config.get("confidence_penalties", {}).get("no_context", -0.7)
            entity.confidence += penalty
            reasons.append(f"No valid medical context found: {penalty}")
    
        self.last_validation_reasons = reasons
        return is_valid



    # Specific handling for overly generic medical terms
    def is_generic_medical_term(text_lower, doc):
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
