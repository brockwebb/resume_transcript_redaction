# redactor/detectors/validation_pattern_matcher.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################

import re
from typing import Dict, List, Optional, Any, Set
from .base_detector import BaseDetector, Entity

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
        super().__init__(config_loader, logger)
        self.patterns = {}
        self._cached_configs = {}
        self.validation_params = {}
        self._load_patterns()
        
        if self.debug_mode:
            self.logger.debug("ValidationPatternMatcher initialized")
            self.logger.debug(f"Loaded patterns for types: {list(self.patterns.keys())}")

    def _get_config(self, config_name: str) -> Dict:
        """
        Get config with caching to avoid repeated loads.
        
        Args:
            config_name: Name of config to load
            
        Returns:
            Dict containing configuration
        """
        if config_name not in self._cached_configs:
            self._cached_configs[config_name] = self.config_loader.get_config(config_name)
        return self._cached_configs[config_name]

    def _validate_config(self) -> bool:
        """
        Validate required configuration is present.
        
        Returns:
            bool indicating if config is valid
        """
        try:
            # Load validation params
            validation_config = self._get_config("validation_params")
            if not validation_config:
                self.logger.error("Missing validation_params configuration")
                return False

            # Verify required entity types have patterns
            required_types = {'DATE_TIME', 'EMAIL_ADDRESS', 'PHONE_NUMBER', 'GPA'}
            for entity_type in required_types:
                entity_config = validation_config.get(entity_type.lower().replace('_', ''))
                if not entity_config or 'patterns' not in entity_config:
                    self.logger.error(f"Missing patterns for {entity_type}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False

###############################################################################
    # SECTION 3: PATTERN LOADING AND MANAGEMENT
    ###############################################################################

    def _load_patterns(self) -> None:
        """
        Load patterns from validation_params.json.
        Maps entity types to their corresponding validation patterns.
        """
        try:
            validation_config = self._get_config("validation_params")
            self.validation_params = validation_config

            # Map entity types to their pattern configurations
            pattern_mappings = {
                'DATE_TIME': ('date', 'patterns'),
                'EMAIL_ADDRESS': ('email_address', 'patterns'),
                'PHONE_NUMBER': ('phone', 'patterns'),
                'GPA': ('gpa', 'common_formats'),
                'INTERNET_REFERENCE': ('internet_reference', 'patterns'),
                'LOCATION': ('location', 'format_patterns'),
                'EDUCATIONAL_INSTITUTION': ('educational', 'format_patterns')
            }

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

        except Exception as e:
            self.logger.error(f"Error loading patterns: {str(e)}")
            raise

    def _get_entity_config(self, entity_type: str) -> Dict:
        """
        Get entity-specific configuration including patterns and rules.
        
        Args:
            entity_type: Type of entity to get config for
            
        Returns:
            Dict containing entity configuration
        """
        config_key = entity_type.lower().replace('_', '')
        return self.validation_params.get(config_key, {})

###############################################################################
    # SECTION 4: ENTITY DETECTION
    ###############################################################################

    def detect_entities(self, text: str) -> List[Entity]:
        """
        Detect entities using validation patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected entities
        """
        if not text:
            return []

        detected_entities = []

        try:
            # Process each entity type
            for entity_type, patterns in self.patterns.items():
                # Get entity-specific validation config
                entity_config = self._get_entity_config(entity_type)
                
                if self.debug_mode:
                    self.logger.debug(f"Processing {entity_type} with {len(patterns)} patterns")

                # Find matches for all patterns
                matches = self._find_pattern_matches(text, patterns, entity_type)

                # Validate and filter matches
                for match in matches:
                    if self._validate_match(match, entity_type, entity_config):
                        detected_entities.append(match)

            return detected_entities

        except Exception as e:
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
            List of matched entities
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
                    continue  # Skip invalid pattern formats

                if not regex:
                    continue

                try:
                    # Apply early filtering for certain entity types
                    if entity_type == 'DATE_TIME':
                        text = self._preprocess_date_text(text)
                    elif entity_type == 'LOCATION':
                        text = self._preprocess_location_text(text)

                    # Find all matches
                    for match in re.finditer(regex, text, re.IGNORECASE):
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
                    self.logger.warning(
                        f"Invalid regex pattern for {entity_type} - {pattern_name}: {e}"
                    )
                    continue

            return matches

        except Exception as e:
            self.logger.error(f"Error finding pattern matches: {str(e)}")
            return []


###############################################################################
    # SECTION 5: PREPROCESSING AND FILTERING
    ###############################################################################

    def _preprocess_date_text(self, text: str) -> str:
        """
        Preprocess text for date detection.
        Removes known false positive contexts.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove known false positive contexts
        false_pos_words = {'hours:', 'minutes:', 'seconds:', 'grade:', 'score:'}
        lines = []
        for line in text.split('\n'):
            if not any(fp in line.lower() for fp in false_pos_words):
                lines.append(line)
                
        if self.debug_mode and len(lines) < len(text.split('\n')):
            self.logger.debug("Removed lines containing false positive date contexts")
            
        return '\n'.join(lines)

    def _preprocess_location_text(self, text: str) -> str:
        """
        Preprocess text for location detection.
        Removes common false positive patterns.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Remove common false positive patterns
        false_pos_patterns = [
            r'(?i)organizer,\s+\w+',  # "Organizer, AI" etc.
            r'(?i)lead\s+\w+,\s+\w+'  # "Lead Something, Something"
        ]
        processed = text
        for pattern in false_pos_patterns:
            processed = re.sub(pattern, '', processed)
            
        if self.debug_mode and processed != text:
            self.logger.debug("Removed false positive location patterns")
            
        return processed

    def _is_false_positive(self, text: str, entity_type: str) -> bool:
        """
        Check if match is likely a false positive.
        
        Args:
            text: Matched text to check
            entity_type: Type of entity detected
            
        Returns:
            bool indicating if match is likely false positive
        """
        text_lower = text.lower()
        
        if entity_type == 'LOCATION':
            false_pos_terms = {'organizer', 'lead', 'head of'}
            if any(term in text_lower for term in false_pos_terms):
                if self.debug_mode:
                    self.logger.debug(f"Found false positive location term in: {text}")
                return True
            
        elif entity_type == 'DATE_TIME':
            false_pos_terms = {'hour', 'minute', 'second', 'grade', 'score'}
            if any(term in text_lower for term in false_pos_terms):
                if self.debug_mode:
                    self.logger.debug(f"Found false positive date term in: {text}")
                return True
            
        return False

###############################################################################
    # SECTION 6: VALIDATION LOGIC
    ###############################################################################

    def _validate_match(self, entity: Entity, entity_type: str, config: Dict) -> bool:
        """
        Validate a pattern match using entity-specific rules.
        
        Args:
            entity: Entity to validate
            entity_type: Type of entity
            config: Entity-specific validation config
            
        Returns:
            bool indicating if match is valid
        """
        try:
            # Get validation rules
            validation_rules = config.get('validation_rules', {})

            # Basic validation
            if not entity.text.strip():
                return False

            text_length = len(entity.text)
            min_chars = validation_rules.get('min_chars', 2)
            max_length = validation_rules.get('max_length', 1000)

            if text_length < min_chars or text_length > max_length:
                if self.debug_mode:
                    self.logger.debug(
                        f"Length validation failed for {entity_type}: "
                        f"{text_length} chars (min={min_chars}, max={max_length})"
                    )
                return False

            # Entity-specific validation
            if entity_type == 'DATE_TIME':
                return self._validate_date(entity, config)
            elif entity_type == 'GPA':
                return self._validate_gpa(entity, config)
            elif entity_type == 'PHONE_NUMBER':
                return self._validate_phone(entity, config)
            elif entity_type == 'EMAIL_ADDRESS':
                return self._validate_email(entity, config)
            elif entity_type == 'LOCATION':
                return self._validate_location(entity, config)

            return True

        except Exception as e:
            self.logger.error(f"Error validating match: {str(e)}")
            return False

    def _validate_date(self, entity: Entity, config: Dict) -> bool:
        """
        Validate date matches using date-specific rules.
        
        Args:
            entity: Date entity to validate
            config: Date validation configuration
            
        Returns:
            bool indicating if date is valid
        """
        text_lower = entity.text.lower()
        
        # Check for false positive contexts
        false_positives = config.get('false_positive_contexts', [])
        if any(fp in text_lower for fp in false_positives):
            if self.debug_mode:
                self.logger.debug(f"Date appears in false positive context: {text_lower}")
            return False

        # Apply confidence adjustments
        adjustments = config.get('confidence_adjustments', {})
        
        # Boost for year markers
        year_markers = config.get('year_markers', [])
        if any(marker in text_lower for marker in year_markers):
            boost = adjustments.get('has_marker', 0.2)
            entity.confidence += boost
            if self.debug_mode:
                self.logger.debug(f"Applied date marker boost: +{boost}")

        return entity.confidence >= self.confidence_threshold

    def _validate_gpa(self, entity: Entity, config: Dict) -> bool:
        """
        Validate GPA matches using GPA-specific rules.
        
        Args:
            entity: GPA entity to validate
            config: GPA validation configuration
            
        Returns:
            bool indicating if GPA is valid
        """
        text_lower = entity.text.lower()
        value_match = re.search(r'(\d+\.?\d*)', text_lower)
        
        if not value_match:
            return False

        try:
            value = float(value_match.group(1))
            min_value = config.get('validation_rules', {}).get('min_value', 0.0)
            max_value = config.get('validation_rules', {}).get('max_value', 4.0)

            if not (min_value <= value <= max_value):
                if self.debug_mode:
                    self.logger.debug(f"GPA value {value} outside valid range [{min_value}, {max_value}]")
                return False

            # Apply confidence adjustments
            boosts = config.get('confidence_boosts', {})
            
            # Check for GPA indicators
            if any(ind in text_lower for ind in config.get('indicators', [])):
                boost = boosts.get('has_indicator', 0.2)
                entity.confidence += boost
                if self.debug_mode:
                    self.logger.debug(f"Applied GPA indicator boost: +{boost}")

            return entity.confidence >= self.confidence_threshold

        except ValueError:
            return False

    def _validate_phone(self, entity: Entity, config: Dict) -> bool:
        """
        Validate phone number matches using phone-specific rules.
        
        Args:
            entity: Phone entity to validate
            config: Phone validation configuration
            
        Returns:
            bool indicating if phone number is valid
        """
        # Extract digits
        digits = re.sub(r'\D', '', entity.text)
        
        # Check digit count
        min_digits = config.get('validation_rules', {}).get('min_digits', 10)
        max_digits = config.get('validation_rules', {}).get('max_digits', 15)
        
        if not (min_digits <= len(digits) <= max_digits):
            if self.debug_mode:
                self.logger.debug(f"Invalid digit count: {len(digits)}")
            return False

        # Apply confidence adjustments
        boosts = config.get('confidence_boosts', {})
        
        # Standard format boost
        if re.match(r'^\(\d{3}\)\s*\d{3}-\d{4}$', entity.text):
            boost = boosts.get('standard_format', 0.1)
            entity.confidence += boost
            if self.debug_mode:
                self.logger.debug(f"Applied standard format boost: +{boost}")

        return entity.confidence >= self.confidence_threshold

    def _validate_email(self, entity: Entity, config: Dict) -> bool:
        """
        Validate email address matches using email-specific rules.
        
        Args:
            entity: Email entity to validate
            config: Email validation configuration
            
        Returns:
            bool indicating if email is valid
        """
        text = entity.text.strip()
        
        # Basic email validation
        if not re.match(r'^[^@]+@[^@]+\.[^@]+$', text):
            return False

        # Length validation
        max_length = config.get('validation_rules', {}).get('max_length', 254)
        if len(text) > max_length:
            return False

        # Apply confidence adjustments
        boosts = config.get('confidence_boosts', {})
        
        # Known domain boost
        domain = text.split('@')[1].lower()
        if domain in config.get('known_domains', []):
            boost = boosts.get('known_domain', 0.1)
            entity.confidence += boost
            if self.debug_mode:
                self.logger.debug(f"Applied known domain boost: +{boost}")

        return entity.confidence >= self.confidence_threshold

    def _validate_location(self, entity: Entity, config: Dict) -> bool:
        """
        Validate location matches using location-specific rules.
        
        Args:
            entity: Location entity to validate
            config: Location validation configuration
            
        Returns:
            bool indicating if location is valid
        """
        validation_rules = config.get('validation_rules', {})
        format_patterns = config.get('format_patterns', {})
        confidence_boosts = config.get('confidence_boosts', {})
        
        words = entity.text.split()
        
        # Basic validation
        if len(entity.text) < validation_rules.get('min_chars', 2):
            return False
            
        # Check format patterns
        for pattern_name, pattern_info in format_patterns.items():
            if re.match(pattern_info.get('regex', ''), entity.text):
                boost = confidence_boosts.get('known_format', 0.1)
                entity.confidence += boost
                if self.debug_mode:
                    self.logger.debug(f"Applied {pattern_name} format boost: +{boost}")
                return True
                
        # Single word validation
        if len(words) == 1 and not validation_rules.get('allow_single_word', True):
            return False
            
        return entity.confidence >= self.confidence_threshold

