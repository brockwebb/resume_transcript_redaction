# redactor/detectors/custom_recognizers.py

from typing import List, Dict, Any, Optional
from presidio_analyzer import Pattern, PatternRecognizer
from presidio_analyzer.recognizer_result import RecognizerResult
import logging
import re

class ConfigDrivenAddressRecognizer(PatternRecognizer):
    """Address recognizer using patterns from config."""
    
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomAddressRecognizer",
    ):
        """Initialize recognizer with patterns from config."""
        self.logger = logger
        
        try:
            # Load patterns from detection_patterns.yaml
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing detection patterns configuration")
            
            # Convert config patterns to Presidio Pattern objects
            patterns = []
            address_patterns = patterns_config.get("address_patterns", [])
            for pattern in address_patterns:
                if pattern["entity_type"] == "ADDRESS":
                    if self.logger:
                        self.logger.debug(f"Loading address pattern: {pattern['name']}")
                    patterns.append(
                        Pattern(
                            name=pattern["name"],
                            regex=pattern["regex"],
                            score=pattern["score"]
                        )
                    )
            
            # Add context words if available
            context = patterns_config.get("context_words", {}).get("address", [])
            
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} address patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
            
            super().__init__(
                supported_entity="ADDRESS",  # Use standardized type
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_result(self, pattern_text: str) -> bool:
        """Validate address beyond regex match.
        
        Args:
            pattern_text: The text that matched the pattern
                
        Returns:
            bool: Whether the match is valid
        """
        if not pattern_text or not pattern_text.strip():
            return False

        # Basic validation steps
        try:
            parts = pattern_text.strip().split()
            
            # Check for minimal address components
            if len(parts) < 2:  # Need at least number and street
                if self.logger:
                    self.logger.debug(f"Address too short: {pattern_text}")
                return False

            # First part should be a number (building number)
            if not parts[0].replace(',', '').isdigit():
                if self.logger:
                    self.logger.debug(f"Missing building number: {pattern_text}")
                return False

            # Look for street type indicators
            street_types = {
                'street', 'st', 'avenue', 'ave', 'boulevard', 'blvd', 
                'lane', 'ln', 'road', 'rd', 'drive', 'dr', 'court', 
                'ct', 'circle', 'cir', 'way', 'parkway', 'pkwy'
            }
            
            has_street_type = False
            for part in parts[1:]:  # Skip building number
                clean_part = part.lower().strip('.,')
                if clean_part in street_types:
                    has_street_type = True
                    break

            if not has_street_type:
                if self.logger:
                    self.logger.debug(f"No street type found: {pattern_text}")
                return False

            # Validate unit/apartment if present
            unit_indicators = {'apt', 'apartment', 'unit', 'suite', 'ste', '#'}
            for part in parts:
                clean_part = part.lower().strip('.,#')
                if clean_part in unit_indicators:
                    if not self._validate_unit_format(parts, part):
                        return False

            return True

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in address validation: {str(e)}")
            return False

    def _validate_unit_format(self, address_parts: List[str], unit_indicator: str) -> bool:
        """Validate unit/apartment number format.
        
        Args:
            address_parts: Split address parts
            unit_indicator: The part containing unit/apt indicator
            
        Returns:
            bool: Whether unit format is valid
        """
        try:
            # Find position of unit indicator
            idx = address_parts.index(unit_indicator)
            
            # Check if there's a number after the indicator
            if idx + 1 >= len(address_parts):
                if self.logger:
                    self.logger.debug("Missing unit number after indicator")
                return False
            
            unit_num = address_parts[idx + 1].strip('.,#')
            
            # Allow digits or alphanumeric (e.g., "100A")
            if not (unit_num.isdigit() or 
                   (unit_num.isalnum() and len(unit_num) <= 4)):
                if self.logger:
                    self.logger.debug(f"Invalid unit number: {unit_num}")
                return False
                
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error validating unit format: {str(e)}")
            return False

    def invalidate_result(self, pattern_text: str) -> bool:
        """Check if result should be invalidated.
        
        Args:
            pattern_text: The text that matched the pattern
            
        Returns:
            bool: Whether the match should be invalidated
        """
        # Use same validation logic for consistency
        return not self.validate_result(pattern_text)
        

class ConfigDrivenLocationRecognizer(PatternRecognizer):
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomLocationRecognizer",
    ):
        self.logger = logger
        self.max_component_distance = 30  # characters between components
        
        try:
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing core patterns configuration")
            
            patterns = []
            location_patterns = patterns_config.get("location_patterns", [])
            for pattern in location_patterns:
                if self.logger:
                    self.logger.debug(f"Loading location pattern: {pattern['name']}")
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
            
            context = patterns_config.get("context_words", {}).get("location", [])
            
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} location patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
            
            super().__init__(
                supported_entity="LOCATION",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def analyze(self, text: str, entities: List[str] = None, nlp_artifacts=None):
        """Override analyze to combine adjacent location components."""
        results = super().analyze(text, entities, nlp_artifacts)
        
        if not results:
            return results
            
        # Group results by proximity
        sorted_results = sorted(results, key=lambda x: x.start)
        grouped_results = self._group_adjacent_components(sorted_results, text)
        
        # Combine each group into a single result
        combined_results = []
        for group in grouped_results:
            if len(group) == 1:
                combined_results.append(group[0])
            else:
                combined = self._combine_location_components(group, text)
                if combined:
                    combined_results.append(combined)
        
        return combined_results

    def _group_adjacent_components(self, results, text) -> List[List[RecognizerResult]]:
        groups = []
        current_group = []
        
        for i, result in enumerate(results):
            if not current_group:
                current_group.append(result)
                continue
                
            # Reduce max distance between components
            last_result = current_group[-1]
            distance = result.start - last_result.end
            
            # Only combine if they're truly adjacent (3-5 chars max)
            if distance <= 5:  # Reduced from 30
                between_text = text[last_result.end:result.start].strip()
                # Only combine with comma or space separators
                if re.match(r'^[\s,]*$', between_text):
                    current_group.append(result)
                    continue
                        
            if current_group:
                groups.append(current_group)
            current_group = [result]
                
        if current_group:
            groups.append(current_group)
                
        return groups

    def _combine_location_components(self, components: List[RecognizerResult], text: str) -> Optional[RecognizerResult]:
        """Combine multiple location components into one."""
        if not components:
            return None
            
        # Get the full text span
        start = min(c.start for c in components)
        end = max(c.end for c in components)
        full_text = text[start:end]
        
        # Calculate combined score
        combined_score = sum(c.score for c in components) / len(components)
        
        # Create new result
        return RecognizerResult(
            entity_type="LOCATION",
            start=start,
            end=end,
            score=combined_score,
            analysis_explanation=None,
            recognition_metadata={
                "combined_from": [c.recognition_metadata for c in components]
            }
        )

    def validate_result(self, pattern_text: str) -> bool:
       if not pattern_text or not pattern_text.strip():
           return False
           
       # Reject multiline text
       if '\n' in pattern_text:
           return False
           
       try:
           # Split into city and state parts
           parts = pattern_text.strip().split(',')
           if len(parts) != 2:
               return False
               
           city, state_part = parts[0].strip(), parts[1].strip()
           
           # City validation
           if not city[0].isalpha() or len(city) < 2:
               return False
               
           # Handle unicode characters in city name
           if not all(c.isalpha() or c.isspace() or c in 'À-ÿ-' for c in city):
               return False
               
           # Extract state and optional zip
           state_zip_match = re.match(r'^([A-Z]{2})(?:\s+(\d{5}(?:-\d{4})?))?\s*$', state_part)
           if not state_zip_match:
               return False
               
           return True
               
       except Exception as e:
           if self.logger:
               self.logger.debug(f"Location validation error: {str(e)}")
           return False
        
  
    def _validate_zip(self, zip_code: str) -> bool:
        # Add ZIP code validation logic
        return bool(re.match(r'^\d{5}(?:-\d{4})?$', zip_code))


class ConfigDrivenPhoneRecognizer(PatternRecognizer):
    """Phone number recognizer that uses patterns from config."""
    
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomPhoneRecognizer",
    ):
        """Initialize recognizer with patterns from config."""
        self.logger = logger
    
        # Load patterns from config
        try:
            if self.logger:
                self.logger.debug("Attempting to load core_patterns config")
                if hasattr(config_loader, 'loaded_paths'):
                    self.logger.debug(f"Loaded config paths: {config_loader.loaded_paths}")
                    
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                if self.logger:
                    self.logger.error("core_patterns config not found")
                    self.logger.debug(f"Available configs: {list(config_loader.configs.keys())}")
                raise ValueError("Missing core patterns configuration")
    
            # Convert config patterns to Presidio Pattern objects
            patterns = []
            for pattern in patterns_config.get("phone_patterns", []):
                if self.logger:
                    self.logger.debug(f"Loading phone pattern: {pattern['name']}")
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
    
            # Get context words from config
            context = patterns_config.get("context_words", {}).get("phone", [])
    
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} phone patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
    
            super().__init__(
                supported_entity="PHONE_NUMBER",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
    
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_result(self, pattern_text: str) -> bool:
        """Additional validation beyond regex match.
        
        Args:
            pattern_text: The text that matched the pattern
                
        Returns:
            bool: Whether the match is valid
        """
        # Remove all non-digit characters for counting and validation
        digits = ''.join(c for c in pattern_text if c.isdigit())
        
        # Log the normalized number for debugging
        if self.logger:
            self.logger.debug(f"Validating phone number: {pattern_text}")
            self.logger.debug(f"Extracted digits: {digits}")
        
        # Basic validation - check digit count ranges
        digit_count = len(digits)
        
        # Handle different formats:
        # - US domestic: 10 digits
        # - US with country code: 11 digits
        # - International: 10-15 digits
        # - With extension: up to 15 digits
        if digit_count < 10:
            if self.logger:
                self.logger.debug(f"Invalid: insufficient digits ({digit_count})")
            return False
            
        if digit_count > 15:
            if self.logger:
                self.logger.debug(f"Invalid: too many digits ({digit_count})")
            return False
    
        # Check for basic invalid patterns
        return not self._is_invalid_pattern(digits, pattern_text)
    
    def _is_invalid_pattern(self, digits: str, original_text: str) -> bool:
        """Check for obviously invalid number patterns.
        
        Args:
            digits: String containing only the digits from the phone number
            original_text: Original text including formatting
            
        Returns:
            bool: True if pattern is invalid, False if valid
        """
        # Always allow standard US area code "555" in test numbers
        if "555" in digits and len(digits) >= 10:
            return False
            
        # Handle international format specially
        if original_text.startswith('+'):
            # Extract country code (1-3 digits)
            for i in range(1, 4):
                country_code = digits[:i]
                # Valid country codes shouldn't start with 0
                if country_code.startswith('0'):
                    if self.logger:
                        self.logger.debug(f"Invalid country code: {country_code}")
                    return True
            return False  # Valid international number
            
        # For US numbers (10 or 11 digits), check area code
        if len(digits) in (10, 11):
            area_code_start = 1 if len(digits) == 11 else 0
            area_code = digits[area_code_start:area_code_start + 3]
            
            # Area code can't start with 0 or 1 (except "1" for country code)
            if area_code.startswith(('0', '1')) and not (len(digits) == 11 and digits.startswith('1')):
                if self.logger:
                    self.logger.debug(f"Invalid US area code: {area_code}")
                return True
        
        # Check for obviously fake patterns
        invalid_patterns = [
            "0000000",
            "1111111",
            "9999999",
            "1234567",
            "7654321"
        ]
        
        for pattern in invalid_patterns:
            if pattern in digits:
                if self.logger:
                    self.logger.debug(f"Found invalid pattern: {pattern}")
                return True
                
        # Check for excessive repetition
        most_common = max(set(digits), key=digits.count)
        repetition_ratio = digits.count(most_common) / len(digits)
        if repetition_ratio > 0.7 and most_common != '5':  # Allow more 5s for test numbers
            if self.logger:
                self.logger.debug(f"Excessive digit repetition: {repetition_ratio:.2f}")
            return True
                
        return False  # Pattern is valid
    
    def invalidate_result(self, pattern_text: str) -> bool:
        """Check if result should be invalidated.
        
        Args:
            pattern_text: The text that matched the pattern
            
        Returns:
            bool: Whether the match should be invalidated
        """
        # Use the same validation logic for consistency
        digits = ''.join(c for c in pattern_text if c.isdigit())
        return self._is_invalid_pattern(digits, pattern_text)

class ConfigDrivenEducationalInstitutionRecognizer(PatternRecognizer):
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomEducationalInstitutionRecognizer",
    ):
        self.logger = logger
        
        try:
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing core patterns configuration")
            
            patterns = []
            institution_patterns = patterns_config.get("educational_institution_patterns", [])
            for pattern in institution_patterns:
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
            
            context = patterns_config.get("context_words", {}).get("educational", [])
            
            super().__init__(
                supported_entity="EDUCATIONAL_INSTITUTION",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_result(self, pattern_text):
        """Validate and clean educational institution matches."""
        # Skip empty or invalid text
        if not pattern_text or len(pattern_text.strip()) == 0:
            return False
            
        # Split on newlines to handle multi-line matches
        lines = pattern_text.split('\n')
        
        # Find the line containing university/college/institute
        edu_keywords = ['University', 'College', 'Institute', 'School']
        university_line = ''
        for line in lines:
            if any(keyword in line for keyword in edu_keywords):
                university_line = line.strip()
                break
                
        if not university_line:
            return False
            
        # Must be properly capitalized
        words = university_line.split()
        if len(words) < 2:  # Require at least 2 words
            return False
            
        # Check capitalization of significant words
        for word in words:
            # Skip short words and common lowercase words
            if len(word) <= 3 or word.lower() in {'of', 'the', 'and', 'in', 'at', 'by', 'for'}:
                continue
            if not word[0].isupper():
                return False
                
        # Check for common edu keywords
        if not any(keyword in university_line for keyword in edu_keywords):
            return False
            
        # Filter out degree names without institution
        degree_indicators = {'bachelor', 'master', 'phd', 'doctorate', 'bs', 'ba', 'ms', 'ma'}
        if any(indicator in university_line.lower() for indicator in degree_indicators):
            if not any(keyword in university_line for keyword in edu_keywords):
                return False
                
        return True

class ConfigDrivenGpaRecognizer(PatternRecognizer):
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomGpaRecognizer",
    ):
        self.logger = logger
        
        try:
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing core patterns configuration")
            
            patterns = []
            gpa_patterns = patterns_config.get("gpa_patterns", [])
            for pattern in gpa_patterns:
                if self.logger:
                    self.logger.debug(f"Loading GPA pattern: {pattern['name']}")
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
            
            context = patterns_config.get("context_words", {}).get("gpa", [])
            
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} GPA patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
            
            super().__init__(
                supported_entity="GPA",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_result(self, pattern_text: str) -> bool:
        """Validate detected GPA value.
        
        Args:
            pattern_text: The text that matched the pattern
            
        Returns:
            bool: Whether the GPA value is valid
        """
        try:
            # Extract just the numeric GPA value using regex
            gpa_match = re.search(r'([0-3](?:\.[0-9]{1,2})?|4(?:\.0{1,2})?)', pattern_text)
            if not gpa_match:
                if self.logger:
                    self.logger.debug(f"No valid GPA value found in: {pattern_text}")
                return False
                
            value = float(gpa_match.group(1))
            
            # Validate range (0-4)
            if not (0 <= value <= 4):
                if self.logger:
                    self.logger.debug(f"GPA value {value} outside valid range")
                return False
                
            if self.logger:
                self.logger.debug(f"Valid GPA found: {value}")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"GPA validation error: {str(e)}")
            return False


from typing import Optional
import logging
from presidio_analyzer import Pattern, PatternRecognizer
class ConfigDrivenSocialMediaRecognizer(PatternRecognizer):
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomSocialMediaRecognizer",
    ):
        self.logger = logger
        
        try:
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing core patterns configuration")
            
            patterns = []
            social_patterns = patterns_config.get("social_media_patterns", [])
            for pattern in social_patterns:
                if self.logger:
                    self.logger.debug(f"Loading social media pattern: {pattern['name']}")
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
            
            context = patterns_config.get("context_words", {}).get("social_media", [])
            
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} social media patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
            
            super().__init__(
                supported_entity="SOCIAL_MEDIA",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)

    def validate_result(self, pattern_text: str) -> bool:
        if not pattern_text or not pattern_text.strip():
            return False
            
        try:
            # Twitter handle validation
            if pattern_text.startswith('@'):
                handle = pattern_text[1:]
                valid = len(handle) >= 1 and len(handle) <= 19 and all(c.isalnum() or c == '_' for c in handle)
                if self.logger:
                    self.logger.debug(f"Twitter handle validation {'passed' if valid else 'failed'}: {pattern_text}")
                return valid
                    
            # URL validations
            if 'github.com/' in pattern_text.lower() or 'linkedin.com/' in pattern_text.lower():
                parts = pattern_text.split('/')
                if len(parts) < 4:
                    return False
                username = parts[-1].strip('/')
                return bool(username and len(username) >= 2)
                    
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.debug(f"Social media validation error: {str(e)}")
            return False


class ConfigDrivenDemographicRecognizer(PatternRecognizer):
    def __init__(self, config_loader, logger):
        # Load patterns from config
        patterns = self._load_patterns(config_loader)
        
        super().__init__(
            supported_entity="DEMOGRAPHIC",
            patterns=patterns,
            context=["diversity", "inclusion", "women", "underrepresented", "minority"]
        )

    def _load_patterns(self, config_loader):
        try:
            core_patterns = config_loader.get_config("core_patterns")
            return core_patterns.get("demographic_patterns", [])
        except Exception as e:
            self.logger.error(f"Error loading demographic patterns: {str(e)}")
            return []

    def validate_result(self, pattern_text):
        """Validate demographic organization matches."""
        clean_text = re.sub(r'\([^)]*\)', '', pattern_text).strip()
        
        # Must have proper capitalization
        words = clean_text.split()
        return any(w[0].isupper() for w in words if len(w) > 3)



class ConfigDrivenSsnRecognizer(PatternRecognizer):
    """SSN recognizer that uses patterns from config."""
    
    def __init__(
        self,
        config_loader: Any,
        logger: Optional[logging.Logger] = None,
        supported_language: str = "en",
        name: str = "CustomSsnRecognizer",
    ):
        """Initialize recognizer with patterns from config."""
        self.logger = logger
        
        try:
            patterns_config = config_loader.get_config("core_patterns")
            if not patterns_config:
                raise ValueError("Missing core patterns configuration")
            
            # Convert config patterns to Presidio Pattern objects
            patterns = []
            for pattern in patterns_config.get("ssn_patterns", []):
                if self.logger:
                    self.logger.debug(f"Loading SSN pattern: {pattern['name']}")
                patterns.append(
                    Pattern(
                        name=pattern["name"],
                        regex=pattern["regex"],
                        score=pattern["score"]
                    )
                )
            
            # Get context words
            context = patterns_config.get("context_words", {}).get("ssn", [])
            
            if self.logger:
                self.logger.debug(f"Loaded {len(patterns)} SSN patterns")
                self.logger.debug(f"Loaded {len(context)} context words")
            
            super().__init__(
                supported_entity="US_SSN",
                patterns=patterns,
                supported_language=supported_language,
                context=context,
                name=name,
            )
            
        except Exception as e:
            error_msg = f"Error initializing {name}: {str(e)}"
            if self.logger:
                self.logger.error(error_msg)
            raise ValueError(error_msg)
    
    def validate_result(self, pattern_text: str) -> bool:
        """Validate SSN format.
        
        Args:
            pattern_text: The text that matched the pattern
            
        Returns:
            bool: Whether the match is valid
        """
        # Remove any non-digit characters
        digits = ''.join(c for c in pattern_text if c.isdigit())
        
        if self.logger:
            self.logger.debug(f"Validating SSN: {pattern_text}")
            
        # Must be exactly 9 digits
        if len(digits) != 9:
            if self.logger:
                self.logger.debug(f"Invalid SSN length: {len(digits)} digits")
            return False
            
        # Check for invalid area numbers (first 3 digits)
        area = digits[:3]
        if area in ('000', '666') or area.startswith('9'):
            if self.logger:
                self.logger.debug(f"Invalid SSN area number: {area}")
            return False
            
        # Check for invalid group number (middle 2 digits)
        group = digits[3:5]
        if group == '00':
            if self.logger:
                self.logger.debug("Invalid SSN group number")
            return False
            
        # Check for invalid serial number (last 4 digits)
        serial = digits[5:]
        if serial == '0000':
            if self.logger:
                self.logger.debug("Invalid SSN serial number")
            return False
            
        return True