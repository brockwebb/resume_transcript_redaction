# redactor/detectors/validation_pattern_matcher.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

import re
from typing import Dict, List, Optional, Any, Set
from .base_detector import BaseDetector, Entity
import logging

###############################################################################
# SECTION 2: CLASS DEFINITION AND INITIALIZATION
###############################################################################

class ValidationPatternMatcher(BaseDetector):
    """
    Pattern matcher that uses validation_params.json patterns.
    Ensures consistency between detection and validation rules.
    """
    
    def __init__(self, config_loader: Any, logger: Optional[Any] = None):
        """
        Initialize with configuration loader.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger instance for debugging
        """
        # Initialize pattern matching components
        self.patterns = {}
        self.validation_params = {}
        
        # Initialize base detector
        super().__init__(config_loader, logger)
        
        # Load patterns using cached configs
        self._load_patterns()
        
        if self.debug_mode:
            self.logger.debug("ValidationPatternMatcher initialized")
            self.logger.debug(f"Loaded patterns for types: {list(self.patterns.keys())}")

    def _validate_config(self) -> bool:
        """
        Validate required configuration is present.
        
        Returns:
            bool indicating if configuration is valid
        """
        try:
            # Get cached configs
            validation_config = self._cached_configs.get("validation_params")
            routing_config = self._cached_configs.get("entity_routing")
            
            if not validation_config or not routing_config:
                if self.logger:
                    self.logger.error("Missing required cached configurations")
                return False

            # Get required types from routing config
            required_types = set()
            routing = routing_config.get("routing", {})
            
            # Only presidio_primary and ensemble_required need pattern validation
            for section in ["presidio_primary", "ensemble_required"]:
                if section in routing:
                    types = routing[section].get("entities", [])
                    if self.debug_mode:
                        self.logger.debug(f"Found {len(types)} entities in {section}")
                    required_types.update(types)

            # Remove entity types that use built-in detectors
            builtin_detectors = {'PHONE_NUMBER', 'EMAIL_ADDRESS', 'INTERNET_REFERENCE'} 
            required_types = required_types - builtin_detectors
            
            if self.debug_mode:
                self.logger.debug(f"Types requiring pattern validation: {required_types}")

            # Verify patterns exist for required types
            for entity_type in required_types:
                entity_config = validation_config.get(entity_type)
                
                if self.debug_mode:
                    self.logger.debug(f"Checking config for {entity_type}")
                
                if not entity_config:
                    if self.logger:
                        self.logger.error(f"Missing config for {entity_type}")
                    return False
                    
                # Check for any valid pattern type
                has_patterns = (
                    'patterns' in entity_config or
                    'common_formats' in entity_config or
                    'format_patterns' in entity_config
                )
                
                if not has_patterns:
                    if self.logger:
                        self.logger.error(f"Missing patterns for {entity_type}")
                    return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating config: {str(e)}")
            return False

    def _load_patterns(self) -> None:
        """Load patterns from validation_params.json."""
        try:
            validation_config = self._cached_configs.get("validation_params")
            routing_config = self._cached_configs.get("entity_routing")
            
            if not validation_config or not routing_config:
                raise ValueError("Required cached configurations not found")
                
            self.validation_params = validation_config
            
            # Only load patterns for relevant sections
            for section in ["presidio_primary", "ensemble_required"]:
                if section in routing_config.get("routing", {}):
                    entities = routing_config["routing"][section].get("entities", [])
                    for entity_type in entities:
                        # Skip builtin detectors
                        if entity_type in {'PHONE_NUMBER', 'EMAIL_ADDRESS', 'INTERNET_REFERENCE'}:
                            continue

                        entity_config = validation_config.get(entity_type)
                        if entity_config:
                            pattern_key = self._determine_pattern_key(entity_config)
                            
                            if pattern_key in entity_config:
                                self.patterns[entity_type] = entity_config[pattern_key]
                                if self.debug_mode:
                                    self.logger.debug(
                                        f"Loaded {len(entity_config[pattern_key])} patterns "
                                        f"for {entity_type}"
                                    )
                            else:
                                if self.debug_mode:
                                    self.logger.debug(
                                        f"No {pattern_key} found for {entity_type}, "
                                        "trying fallback patterns"
                                    )
                                # Try fallback pattern sources
                                for fallback_key in ['patterns', 'common_formats', 'format_patterns']:
                                    if fallback_key in entity_config:
                                        self.patterns[entity_type] = entity_config[fallback_key]
                                        if self.debug_mode:
                                            self.logger.debug(
                                                f"Using fallback {fallback_key} for {entity_type}"
                                            )
                                        break

                        elif self.debug_mode:
                            self.logger.debug(f"No config found for {entity_type}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading patterns: {str(e)}")
            raise

    def _determine_pattern_key(self, config: Dict) -> str:
        """
        Determine the correct pattern key for an entity config.
        
        Args:
            config: Entity configuration dictionary
            
        Returns:
            str: Best matching pattern key for the configuration
        """
        pattern_keys = ['patterns', 'common_formats', 'format_patterns']
        
        for key in pattern_keys:
            if key in config:
                if self.debug_mode:
                    self.logger.debug(f"Selected pattern key: {key}")
                return key
        
        return 'patterns'  # Default

###############################################################################
# SECTION 3: PATTERN LOADING AND MANAGEMENT
###############################################################################

    def _load_patterns(self) -> None:
        """Load patterns from validation_params.json."""
        try:
            validation_config = self._cached_configs.get("validation_params")
            routing_config = self._cached_configs.get("entity_routing")
            
            if not validation_config or not routing_config:
                raise ValueError("Required cached configurations not found")
                
            self.validation_params = validation_config
            
            # Only load patterns for relevant sections
            for section in ["presidio_primary", "ensemble_required"]:
                if section in routing_config.get("routing", {}):
                    entities = routing_config["routing"][section].get("entities", [])
                    for entity_type in entities:
                        # Skip builtin detectors
                        if entity_type in {'PHONE_NUMBER', 'EMAIL_ADDRESS', 'INTERNET_REFERENCE'}:
                            continue

                        entity_config = validation_config.get(entity_type)
                        if entity_config:
                            pattern_key = self._determine_pattern_key(entity_config)
                            
                            if pattern_key in entity_config:
                                self.patterns[entity_type] = entity_config[pattern_key]
                                if self.debug_mode:
                                    self.logger.debug(
                                        f"Loaded {len(entity_config[pattern_key])} patterns "
                                        f"for {entity_type}"
                                    )
                            else:
                                if self.debug_mode:
                                    self.logger.debug(
                                        f"No {pattern_key} found for {entity_type}, "
                                        "trying fallback patterns"
                                    )
                                # Try fallback pattern sources
                                for fallback_key in ['patterns', 'common_formats', 'format_patterns']:
                                    if fallback_key in entity_config:
                                        self.patterns[entity_type] = entity_config[fallback_key]
                                        if self.debug_mode:
                                            self.logger.debug(
                                                f"Using fallback {fallback_key} for {entity_type}"
                                            )
                                        break

                        elif self.debug_mode:
                            self.logger.debug(f"No config found for {entity_type}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading patterns: {str(e)}")
            raise

    def _determine_pattern_key(self, config: Dict) -> str:
        """
        Determine the correct pattern key for an entity config.
        
        Args:
            config: Entity configuration dictionary
            
        Returns:
            str: Best matching pattern key for the configuration
        """
        pattern_keys = ['patterns', 'common_formats', 'format_patterns']
        
        for key in pattern_keys:
            if key in config:
                if self.debug_mode:
                    self.logger.debug(f"Selected pattern key: {key}")
                return key
        
        return 'patterns'  # Default


    
###############################################################################
# SECTION 4: PATTERN LOADING AND MANAGEMENT
###############################################################################

    def _load_patterns(self) -> None:
        """
        Load patterns from validation_params.json.
        Maps patterns to entity types based on configuration and section routing.
        """
        try:
            # Get cached configs
            validation_config = self._cached_configs.get("validation_params")
            routing_config = self._cached_configs.get("entity_routing")
            
            if not validation_config or not routing_config:
                raise ValueError("Required cached configurations not found")
                
            self.validation_params = validation_config
            
            # Map entity types to their configurations
            pattern_mappings = {}
            builtin_detectors = {'PHONE_NUMBER', 'EMAIL_ADDRESS', 'INTERNET_REFERENCE'}
            
            # Only load patterns for relevant sections
            for section in ["presidio_primary", "ensemble_required"]:
                if section in routing_config.get("routing", {}):
                    entities = routing_config["routing"][section].get("entities", [])
                    for entity in entities:
                        if entity not in builtin_detectors:
                            config_key = self._get_config_key(entity)
                            if config_key in validation_config:
                                pattern_key = self._determine_pattern_key(
                                    validation_config[config_key]
                                )
                                pattern_mappings[entity] = (config_key, pattern_key)
                                
                            elif self.debug_mode:
                                self.logger.debug(f"No patterns needed for {entity}")

            # Load mapped patterns
            for entity_type, (config_key, pattern_key) in pattern_mappings.items():
                if config_key in validation_config:
                    entity_config = validation_config[config_key]
                    if pattern_key in entity_config:
                        self.patterns[entity_type] = entity_config[pattern_key]
                        if self.debug_mode:
                            self.logger.debug(
                                f"Loaded {len(entity_config[pattern_key])} patterns "
                                f"for {entity_type}"
                            )
                    else:
                        if self.debug_mode:
                            self.logger.debug(
                                f"No {pattern_key} found for {entity_type}, "
                                "trying fallback patterns"
                            )
                        # Try fallback pattern sources
                        for fallback_key in ['patterns', 'common_formats', 'format_patterns']:
                            if fallback_key in entity_config:
                                self.patterns[entity_type] = entity_config[fallback_key]
                                if self.debug_mode:
                                    self.logger.debug(
                                        f"Using fallback {fallback_key} for {entity_type}"
                                    )
                                break
                                
                if entity_type not in self.patterns and self.debug_mode:
                    self.logger.debug(f"No patterns loaded for {entity_type}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error loading patterns: {str(e)}")
            raise

    def _determine_pattern_key(self, config: Dict) -> str:
        """
        Determine the correct pattern key for an entity config.
        
        Args:
            config: Entity configuration dictionary
            
        Returns:
            str: Best matching pattern key for the configuration
        """
        # Try pattern keys in order of preference
        pattern_keys = ['patterns', 'common_formats', 'format_patterns']
        
        # Return the first pattern key found in the config
        for key in pattern_keys:
            if key in config:
                if self.debug_mode:
                    self.logger.debug(f"Selected pattern key: {key}")
                return key
        
        # Default to 'patterns' if none found
        if self.debug_mode:
            self.logger.debug("No pattern key found, defaulting to 'patterns'")
        return 'patterns'

###############################################################################
# SECTION 5: ENTITY DETECTION
###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities using validation patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List[Entity]: List of detected entities
        """
        if not text:
            return []

        detected_entities = []

        try:
            # Process each entity type with its patterns
            for entity_type, patterns in self.patterns.items():
                # Get entity-specific validation config
                entity_config = self._get_entity_config(entity_type)
                
                if self.debug_mode:
                    self.logger.debug(f"Processing {entity_type} with {len(patterns)} patterns")

                # Find matches for all patterns
                matches = self._find_pattern_matches(text, patterns, entity_type)

                # Validate and filter matches
                for match in matches:
                    if self.validate_detection(match, text):
                        detected_entities.append(match)

            # Sort entities by position
            detected_entities.sort(key=lambda x: (x.start, x.end))
            
            if self.debug_mode:
                self.logger.debug(
                    f"Found {len(detected_entities)} total entities in text"
                )
                for entity in detected_entities:
                    self.logger.debug(
                        f"- {entity.entity_type}: '{entity.text}' "
                        f"({entity.confidence:.2f})"
                    )

            return detected_entities

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error detecting entities: {str(e)}")
            return []

    def _find_pattern_matches(self, text: str, patterns: Dict, entity_type: str) -> List[Entity]:
        """
        Find all pattern matches for an entity type.
        
        Args:
            text: Text to search in
            patterns: Dictionary of patterns to match
            entity_type: Type of entity to detect
            
        Returns:
            List[Entity]: List of matched entities
        """
        matches = []

        try:
            for pattern_name, pattern_info in patterns.items():
                # Handle different pattern formats
                if isinstance(pattern_info, str):
                    regex = pattern_info
                    confidence = 0.8  # Default confidence
                elif isinstance(pattern_info, dict):
                    regex = pattern_info.get('regex', '')
                    confidence = pattern_info.get('confidence', 0.8)
                else:
                    if self.debug_mode:
                        self.logger.debug(f"Skipping invalid pattern format: {pattern_info}")
                    continue

                if not regex:
                    continue

                try:
                    # Apply early preprocessing if needed
                    processed_text = text
                    if entity_type == 'DATE_TIME':
                        processed_text = self._preprocess_text(text, entity_type)
                    elif entity_type == 'LOCATION':
                        processed_text = self._preprocess_text(text, entity_type)

                    # Find all matches
                    for match in re.finditer(regex, processed_text, re.IGNORECASE):
                        # Skip false positives
                        if self._is_false_positive(match.group(0), entity_type):
                            if self.debug_mode:
                                self.logger.debug(
                                    f"Skipping false positive match: '{match.group(0)}'"
                                )
                            continue

                        entity = Entity(
                            text=match.group(0),
                            entity_type=entity_type,
                            confidence=confidence,
                            start=match.start(),
                            end=match.end(),
                            source="pattern_matcher",
                            metadata={
                                "pattern_name": pattern_name,
                                "pattern_confidence": confidence,
                                "validation_source": "validation_params"
                            }
                        )
                        matches.append(entity)
                        
                        if self.debug_mode:
                            self.logger.debug(
                                f"Found {entity_type} match: '{entity.text}' "
                                f"using pattern {pattern_name}"
                            )

                except re.error as e:
                    if self.logger:
                        self.logger.warning(
                            f"Invalid regex pattern for {entity_type} - {pattern_name}: {e}"
                        )
                    continue

            return matches

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error finding pattern matches: {str(e)}")
            return []

    def _preprocess_text(self, text: str, entity_type: str) -> str:
        """
        Preprocess text for specific entity types.
        
        Args:
            text: Text to preprocess
            entity_type: Type of entity for specific preprocessing
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
            
        try:
            entity_config = self._get_entity_config(entity_type)
            preprocessing = entity_config.get('preprocessing', {})
            
            # Get preprocessing rules
            remove_patterns = preprocessing.get('remove_patterns', [])
            normalize_patterns = preprocessing.get('normalize_patterns', {})
            
            processed = text
            
            # Apply removal patterns
            for pattern in remove_patterns:
                try:
                    processed = re.sub(pattern, '', processed)
                except re.error:
                    if self.logger:
                        self.logger.warning(f"Invalid removal pattern: {pattern}")
                        
            # Apply normalization patterns
            for pattern, replacement in normalize_patterns.items():
                try:
                    processed = re.sub(pattern, replacement, processed)
                except re.error:
                    if self.logger:
                        self.logger.warning(f"Invalid normalization pattern: {pattern}")
                    
            if self.debug_mode and processed != text:
                self.logger.debug(f"Preprocessed text for {entity_type}")
                
            return processed
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in text preprocessing: {str(e)}")
            return text

    def _is_false_positive(self, text: str, entity_type: str) -> bool:
        """
        Check if match is likely a false positive based on entity config rules.
        
        Args:
            text: Matched text to check
            entity_type: Type of entity detected
            
        Returns:
            bool: True if match is likely false positive
        """
        try:
            entity_config = self._get_entity_config(entity_type)
            false_positive_rules = entity_config.get('false_positive_rules', {})
            
            # Check false positive patterns
            patterns = false_positive_rules.get('patterns', [])
            for pattern in patterns:
                try:
                    if re.search(pattern, text, re.IGNORECASE):
                        if self.debug_mode:
                            self.logger.debug(
                                f"False positive pattern match: {pattern}"
                            )
                        return True
                except re.error:
                    if self.logger:
                        self.logger.warning(f"Invalid false positive pattern: {pattern}")
            
            # Check false positive terms
            terms = false_positive_rules.get('terms', [])
            text_lower = text.lower()
            if any(term.lower() in text_lower for term in terms):
                if self.debug_mode:
                    self.logger.debug("False positive term match")
                return True
            
            return False
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error checking false positives: {str(e)}")
            return False
            

###############################################################################
# SECTION 6: ENTITY VALIDATION
###############################################################################

    def validate_detection(self, entity: Entity, text: str = "") -> bool:
        """
        Validate entity detection using type-specific rules.
        Implementation of BaseDetector abstract method.
        
        Args:
            entity: Entity to validate
            text: Optional context text
            
        Returns:
            bool: Whether the entity is valid
        """
        try:
            # Get entity-specific config
            entity_config = self._get_entity_config(entity.entity_type)
            if not entity_config:
                if self.debug_mode:
                    self.logger.debug(f"No validation config found for {entity.entity_type}")
                return False
                
            # Skip validation for builtin detectors
            builtin_detectors = {'PHONE_NUMBER', 'EMAIL_ADDRESS', 'INTERNET_REFERENCE'} 
            if entity.entity_type in builtin_detectors:
                return True

            # Get validation rules
            validation_rules = entity_config.get('validation_rules', {})
            
            # Basic text validation
            if not self._validate_text(entity.text, validation_rules):
                return False

            # Pattern-based validation
            if not self._validate_pattern_match(entity, validation_rules):
                return False

            # Type-specific validation
            if not self._validate_type_specific(entity, entity_config, text):
                return False

            # Check confidence threshold
            min_confidence = validation_rules.get(
                'min_confidence', 
                self.confidence_threshold
            )
            if entity.confidence < min_confidence:
                if self.debug_mode:
                    self.logger.debug(
                        f"Confidence below threshold: {entity.confidence} < "
                        f"{min_confidence}"
                    )
                return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating entity {entity}: {str(e)}")
            return False

    def _validate_text(self, text: str, rules: Dict) -> bool:
        """
        Validate basic text properties.
        
        Args:
            text: Text to validate
            rules: Validation rules dictionary
            
        Returns:
            bool: Whether text passes basic validation
        """
        if not text or not text.strip():
            if self.debug_mode:
                self.logger.debug("Empty text")
            return False

        # Check length requirements
        min_chars = rules.get('min_chars', 2)
        max_length = rules.get('max_length', 1000)
        
        if len(text) < min_chars:
            if self.debug_mode:
                self.logger.debug(f"Text too short: {len(text)} < {min_chars}")
            return False
            
        if len(text) > max_length:
            if self.debug_mode:
                self.logger.debug(f"Text too long: {len(text)} > {max_length}")
            return False

        # Check required prefix/suffix if specified
        if 'required_prefix' in rules:
            prefix = rules['required_prefix']
            if not text.startswith(prefix):
                if self.debug_mode:
                    self.logger.debug(f"Missing required prefix: {prefix}")
                return False

        if 'required_suffix' in rules:
            suffix = rules['required_suffix']
            if not text.endswith(suffix):
                if self.debug_mode:
                    self.logger.debug(f"Missing required suffix: {suffix}")
                return False

        return True

    def _validate_pattern_match(self, entity: Entity, rules: Dict) -> bool:
        """
        Validate entity against configured patterns.
        
        Args:
            entity: Entity to validate
            rules: Validation rules dictionary
            
        Returns:
            bool: Whether entity matches required patterns
        """
        if not self.patterns.get(entity.entity_type):
            if self.debug_mode:
                self.logger.debug(f"No patterns defined for {entity.entity_type}")
            return True  # Skip pattern validation if no patterns defined

        # Check if pattern match is required
        if not rules.get('require_pattern_match', True):
            return True

        # Try to match against any pattern
        matched = False
        for pattern in self.patterns[entity.entity_type]:
            pattern_regex = pattern['regex'] if isinstance(pattern, dict) else pattern
            try:
                if re.search(pattern_regex, entity.text, re.IGNORECASE):
                    matched = True
                    # Apply confidence boost if specified
                    if isinstance(pattern, dict) and 'confidence' in pattern:
                        entity.confidence = max(entity.confidence, pattern['confidence'])
                    break
            except re.error:
                if self.logger:
                    self.logger.warning(f"Invalid pattern: {pattern_regex}")
                continue

        if not matched and self.debug_mode:
            self.logger.debug(f"No pattern match for: {entity.text}")
            
        return matched

    def _validate_type_specific(self, entity: Entity, config: Dict, text: str) -> bool:
        """
        Apply entity type-specific validation rules.
        
        Args:
            entity: Entity to validate
            config: Entity configuration dictionary
            text: Full text context
            
        Returns:
            bool: Whether entity passes type-specific validation
        """
        entity_type = entity.entity_type
        type_rules = config.get('type_specific_rules', {})
        
        if not type_rules:
            return True  # No type-specific rules defined
            
        try:
            # Numeric validation
            if 'numeric_range' in type_rules:
                range_config = type_rules['numeric_range']
                numbers = re.findall(r'\d+\.?\d*', entity.text)
                if not numbers:
                    return False
                    
                value = float(numbers[0])
                min_val = range_config.get('min', float('-inf'))
                max_val = range_config.get('max', float('inf'))
                
                if not (min_val <= value <= max_val):
                    if self.debug_mode:
                        self.logger.debug(
                            f"Value {value} outside range [{min_val}, {max_val}]"
                        )
                    return False

            # Context validation
            if 'required_context' in type_rules:
                context_config = type_rules['required_context']
                context_start = max(0, entity.start - context_config.get('window', 50))
                context_end = min(len(text), entity.end + context_config.get('window', 50))
                context = text[context_start:context_end].lower()
                
                required_terms = context_config.get('terms', [])
                found_terms = sum(1 for term in required_terms if term.lower() in context)
                min_terms = context_config.get('min_terms', 1)
                
                if found_terms < min_terms:
                    if self.debug_mode:
                        self.logger.debug(
                            f"Insufficient context terms: {found_terms} < {min_terms}"
                        )
                    return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(
                    f"Error in type-specific validation for {entity_type}: {str(e)}"
                )
            return False