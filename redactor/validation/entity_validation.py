# redactor/validation/entity_validation.py

import logging
import re
import unicodedata
from typing import Dict, List, Optional, Tuple
import json

from evaluation.models import Entity, SensitivityLevel

def validate_educational_institution(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate an entity of type 'EDUCATIONAL_INSTITUTION' using configuration rules.

    Core Validation Logic:
    1. Check for known acronyms (e.g., 'MIT')
    2. Enforce minimum word count
    3. Prevent generic terms
    4. Require core institution terms
    5. Apply confidence adjustments

    Args:
        entity: The Entity object to validate
        text: Context text (optional)
        rules: Validation configuration rules
        logger: Optional logger for debug information

    Returns:
        Validation result dictionary with validation status and confidence adjustments
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    def remove_accents(s: str) -> str:
        """
        Normalize and remove diacritical marks.
        Helps match core terms like 'ecole' or 'polytechnique'.
        """
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )

    # Extract configuration parameters
    core_terms = set(rules.get("core_terms", []))
    known_acronyms = set(rules.get("known_acronyms", []))
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    generic_terms = set(rules.get("generic_terms", []))
    min_words = rules.get("min_words", 2)

    # Prepare text for analysis
    text_str = entity.text.strip()
    text_lower = text_str.lower()
    text_noaccents = remove_accents(text_lower)
    words = text_str.split()
    reasons = []

    # Known Acronym Check (Skip single-word penalty)
    if text_str.upper() in known_acronyms:
        boost = confidence_boosts.get("known_acronym", 0.4)
        reasons.append(f"Boost for known acronym '{text_str.upper()}': {boost}")
        return {
            "is_valid": True,
            "confidence_adjustment": boost,
            "reasons": reasons
        }

    # Single-word Penalty
    if len(words) < min_words:
        penalty = confidence_penalties.get("single_word", -0.3)
        reasons.append(f"Penalty for single word: {penalty}")
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    # Generic Terms Check
    for gterm in generic_terms:
        gterm_noaccents = remove_accents(gterm.lower())
        if gterm_noaccents in text_noaccents:
            penalty = confidence_penalties.get("generic_term", -1.0)
            reasons.append(f"Penalty for generic term '{gterm}': {penalty}")
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }

    # Core Terms Check
    extended_core_terms = set(core_terms) | {"ecole", "polytechnique"}
    if not any(term in text_noaccents for term in extended_core_terms):
        penalty = confidence_penalties.get("no_core_terms", -1.0)
        reasons.append(f"Penalty for missing core terms: {penalty}")
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    # Boost for Core Term
    boost = confidence_boosts.get("full_name", 0.2)
    reasons.append(f"Core term found => Boost: {boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": boost,
        "reasons": reasons
    }


# redactor/validation/entity_validation.py

###############################################################################
# SECTION: PERSON ENTITY VALIDATION
###############################################################################

def validate_person(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate person entities using configuration rules.
    
    Core Validation Logic:
    1. Enforce minimum length requirements
    2. Check word count
    3. Handle capitalization rules
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    title_markers = set(rules.get("title_markers", []))
    
    # Prepare text for analysis
    text_lower = text.lower()
    name_text = entity.text
    name_lower = name_text.lower()
    name_parts = name_text.split()
    reasons = []
    confidence_adjustment = 0.0

    ###############################################################################
    # SUBSECTION: LENGTH VALIDATION
    ###############################################################################
    # Check minimum character length
    if len(name_text) < validation_rules.get("min_chars", 2):
        reasons.append(f"Name too short: {len(name_text)} chars")
        penalty = confidence_penalties.get("single_word", -0.3)
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    # Check minimum word count
    if len(name_parts) < validation_rules.get("min_words", 1):
        reasons.append("Insufficient word count")
        penalty = confidence_penalties.get("single_word", -0.3)
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: TITLE AND CONTEXT CHECKS
    ###############################################################################
    # Title check
    has_title = any(marker in name_lower for marker in title_markers)
    if has_title:
        boost = confidence_boosts.get("has_title", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Title marker found: +{boost}")

    ###############################################################################
    # SUBSECTION: CAPITALIZATION RULES
    ###############################################################################
    # Handle ALL CAPS names
    if name_text.isupper():
        if validation_rules.get("allow_all_caps", True):
            if len(name_parts) >= 2 or has_title:
                boost = confidence_boosts.get("multiple_words", 0.1)
                confidence_adjustment += boost
                reasons.append(f"Valid ALL CAPS name: +{boost}")
            else:
                reasons.append("Single word ALL CAPS without title")
                return {
                    "is_valid": False,
                    "confidence_adjustment": -0.3,
                    "reasons": reasons
                }
    else:
        # Regular capitalization check
        if not all(part[0].isupper() for part in name_parts):
            reasons.append("Name parts must be properly capitalized")
            penalty = confidence_penalties.get("lowercase", -0.2)
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }
        
        boost = confidence_boosts.get("proper_case", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Proper capitalization: +{boost}")

    ###############################################################################
    # SUBSECTION: MULTIPLE WORD BOOST
    ###############################################################################
    # Multiple word boost
    if len(name_parts) >= 2:
        boost = confidence_boosts.get("multiple_words", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Multiple words: +{boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }

# redactor/validation/entity_validation.py

###############################################################################
# SECTION: LOCATION ENTITY VALIDATION
###############################################################################

def validate_location(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate location entities using configuration rules.
    
    Core Validation Logic:
    1. Enforce minimum length requirements
    2. Check capitalization
    3. Validate against format patterns
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    us_states = rules.get("us_states", {})
    format_patterns = rules.get("format_patterns", {})
    
    ###############################################################################
    # SUBSECTION: BASIC VALIDATION
    ###############################################################################
    text_str = entity.text.strip()
    text_lower = text_str.lower()
    words = text_str.split()
    reasons = []
    confidence_adjustment = 0.0

    # Length check
    if len(text_str) < validation_rules.get("min_chars", 2):
        reasons.append("Location too short")
        penalty = confidence_penalties.get("single_word", -0.2)
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: FORMAT PATTERN CHECKING
    ###############################################################################
    def _check_location_formats(entity_text: str, patterns: Dict) -> Tuple[List[str], float, List[str]]:
        """
        Check location formats using regex patterns.
        
        Returns:
            Tuple of (matched_formats, total_boost, reasons)
        """
        matched_formats = []
        total_boost = 0.0
        format_reasons = []

        # US City-State Format
        us_pattern = patterns.get("us_city_state", {}).get("regex", "")
        if us_pattern and re.match(us_pattern, entity_text):
            matched_formats.append("us_city_state")
            boost = confidence_boosts.get("known_format", 0.1) + confidence_boosts.get("has_state", 0.2)
            total_boost += boost
            format_reasons.append(f"Matches US city-state format: +{boost}")
            
            # Additional boost for ZIP code
            if re.search(r'\d{5}(?:-\d{4})?$', entity_text):
                zip_boost = confidence_boosts.get("has_zip", 0.1)
                total_boost += zip_boost
                format_reasons.append(f"Includes ZIP code: +{zip_boost}")
                matched_formats.append("zip_code")

        # International Format
        intl_pattern = patterns.get("international", {}).get("regex", "")
        if intl_pattern and re.match(intl_pattern, entity_text):
            matched_formats.append("international")
            boost = confidence_boosts.get("international", 0.1)
            total_boost += boost
            format_reasons.append(f"Matches international format: +{boost}")

        # Metro Area Format
        metro_pattern = patterns.get("metro_area", {}).get("regex", "")
        if metro_pattern and re.match(metro_pattern, entity_text):
            matched_formats.append("metro_area")
            boost = confidence_boosts.get("metro_area", 0.2)
            total_boost += boost
            format_reasons.append(f"Matches metro area format: +{boost}")

        return matched_formats, total_boost, format_reasons

    ###############################################################################
    # SUBSECTION: CAPITALIZATION AND FORMATTING
    ###############################################################################
    # Capitalization check
    if validation_rules.get("require_capitalization", True):
        if not text_str[0].isupper():
            reasons.append("Location not properly capitalized")
            penalty = confidence_penalties.get("lowercase", -0.3)
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }

    # Run format pattern check
    matched_formats, format_boost, format_reasons = _check_location_formats(text_str, format_patterns)
    confidence_adjustment += format_boost
    reasons.extend(format_reasons)

    ###############################################################################
    # SUBSECTION: SINGLE WORD AND MULTI-WORD HANDLING
    ###############################################################################
    # Handle single-word locations
    if len(words) == 1:
        if not validation_rules.get("allow_single_word", True):
            reasons.append("Single word locations not allowed")
            return {
                "is_valid": False,
                "confidence_adjustment": -0.3,
                "reasons": reasons
            }
        
        # Check for US state
        state_list = [s.lower() for s in 
                      us_states.get("full", []) + 
                      us_states.get("abbrev", [])]
        if text_lower in state_list:
            boost = confidence_boosts.get("has_state", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Contains US state: +{boost}")
        elif text_str.isupper() and validation_rules.get("allow_all_caps", True):
            boost = confidence_boosts.get("proper_case", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Valid ALL CAPS location: +{boost}")
        else:
            penalty = confidence_penalties.get("single_word", -0.2)
            confidence_adjustment += penalty
            reasons.append(f"Single word penalty: {penalty}")

    # Boost for multi-word locations if no format was matched
    if not matched_formats and len(words) > 1:
        boost = confidence_boosts.get("multi_word", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Multi-word location: +{boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }

# redactor/validation/entity_validation.py

###############################################################################
# SECTION: ADDRESS ENTITY VALIDATION
###############################################################################

def validate_address(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate address entities using configuration rules.
    
    Core Validation Logic:
    1. Enforce minimum length requirements
    2. Require street number
    3. Check for street type
    4. Validate unit numbers and directionals
    5. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    
    ###############################################################################
    # SUBSECTION: LOAD CONFIGURATION LISTS
    ###############################################################################
    # Load type lists from config
    street_types = set(
        t.lower() for t in 
        rules.get("street_types", {}).get("full", []) +
        rules.get("street_types", {}).get("abbrev", [])
    )
    unit_types = set(t.lower() for t in rules.get("unit_types", []))
    directionals = set(d.lower() for d in rules.get("directionals", []))
    
    ###############################################################################
    # SUBSECTION: BASIC VALIDATION
    ###############################################################################
    text_str = entity.text.strip()
    text_lower = text_str.lower()
    reasons = []
    confidence_adjustment = 0.0

    # Minimum length check
    if len(text_str) < validation_rules.get("min_chars", 5):
        reasons.append("Address too short")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.5,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: NUMERICAL REQUIREMENTS
    ###############################################################################
    # Check for required number
    if validation_rules.get("require_number", True):
        if not any(c.isdigit() for c in text_str):
            penalty = confidence_penalties.get("no_number", -0.5)
            reasons.append("Missing street number")
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }

    ###############################################################################
    # SUBSECTION: STREET TYPE VALIDATION
    ###############################################################################
    # Check for street type using word boundaries
    has_street_type = False
    for st in street_types:
        if re.search(rf'\b{st}\b(?:[,\s]|$)', text_lower):
            has_street_type = True
            break

    if validation_rules.get("require_street_type", True) and not has_street_type:
        penalty = confidence_penalties.get("no_street_type", -0.3)
        reasons.append("Missing street type")
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }
    else:
        boost = confidence_boosts.get("street_match", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Valid street type: +{boost}")

    ###############################################################################
    # SUBSECTION: UNIT AND DIRECTIONAL CHECKS
    ###############################################################################
    # Check for unit numbers
    has_unit = False
    for unit in unit_types:
        if re.search(rf'\b{unit}\b', text_lower):
            has_unit = True
            boost = confidence_boosts.get("has_unit", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Has unit/suite: +{boost}")
            break

    # Check for directionals
    has_directional = False
    for dir in directionals:
        if re.search(rf'\b{dir}\b', text_lower):
            has_directional = True
            boost = confidence_boosts.get("directional", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Has directional: +{boost}")
            break

    ###############################################################################
    # SUBSECTION: COMPREHENSIVE ADDRESS STRUCTURE
    ###############################################################################
    # Additional boost for complete address structure
    if len(reasons) >= 2:  # Has at least two valid components
        boost = confidence_boosts.get("full_address", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Complete address structure: +{boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }

###############################################################################
# SECTION: EMAIL ADDRESS ENTITY VALIDATION
###############################################################################

def validate_email(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate email address using configuration rules.
    
    Core Validation Logic:
    1. Enforce length requirements
    2. Validate format
    3. Check domain
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    known_domains = set(rules.get("known_domains", []))
    patterns = rules.get("patterns", {})
    
    ###############################################################################
    # SUBSECTION: BASIC LENGTH VALIDATION
    ###############################################################################
    email = entity.text.strip()
    reasons = []
    confidence_adjustment = 0.0

    # Length validation
    if len(email) < validation_rules.get("min_chars", 5):
        reasons.append("Email too short")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }

    if len(email) > validation_rules.get("max_length", 254):
        reasons.append("Email too long")
        penalty = confidence_penalties.get("excessive_length", -0.2)
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: EMAIL FORMAT VALIDATION
    ###############################################################################
    # Check for @ symbol
    if '@' not in email:
        reasons.append("Missing @ symbol")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }

    # Split into local and domain parts
    local_part, domain = email.rsplit('@', 1)
    
    # Local part validation
    if not local_part or local_part[-1] == '.':
        reasons.append("Invalid local part format")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }

    if '..' in local_part:
        reasons.append("Consecutive dots in local part")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: FORMAT AND PATTERN CHECKS
    ###############################################################################
    # Basic format check
    basic_pattern = patterns.get("basic", "")
    if not re.match(basic_pattern, email):
        reasons.append("Invalid email format")
        penalty = confidence_penalties.get("invalid_format", -0.3)
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: DOMAIN VALIDATION
    ###############################################################################
    # Known domain boost
    if any(domain.lower().endswith(d) for d in known_domains):
        boost = confidence_boosts.get("known_domain", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Known domain: +{boost}")

    # Strict pattern check
    if re.match(patterns.get("strict", ""), email):
        boost = confidence_boosts.get("proper_format", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Strict format match: +{boost}")

    ###############################################################################
    # SUBSECTION: CONTEXT VALIDATION
    ###############################################################################
    # Context boost (if context is provided)
    context_words = {"email", "contact", "mail", "address", "@"}
    if any(word in text.lower() for word in context_words):
        boost = confidence_boosts.get("has_context", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Email context: +{boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }



###############################################################################
# SECTION: PHONE NUMBER ENTITY VALIDATION
###############################################################################

def validate_phone(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate phone number using configuration rules.
    
    Core Validation Logic:
    1. Enforce digit count requirements
    2. Validate format patterns
    3. Check context
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    context_words = set(rules.get("context_words", []))
    patterns = rules.get("patterns", {})
    
    ###############################################################################
    # SUBSECTION: DIGIT EXTRACTION AND VALIDATION
    ###############################################################################
    phone_text = entity.text.strip()
    digits = ''.join(c for c in phone_text if c.isdigit())
    reasons = []
    confidence_adjustment = 0.0

    # Digit count validation
    if len(digits) < validation_rules.get("min_digits", 10):
        return {
            "is_valid": False,
            "confidence_adjustment": -0.5,
            "reasons": ["Too few digits"]
        }
    
    if len(digits) > validation_rules.get("max_digits", 15):
        return {
            "is_valid": False,
            "confidence_adjustment": -0.5,
            "reasons": ["Too many digits"]
        }

    ###############################################################################
    # SUBSECTION: EXTENSION HANDLING
    ###############################################################################
    # Check for extension and strip it
    if patterns.get("extension") and re.search(patterns["extension"], phone_text, re.IGNORECASE):
        boost = confidence_boosts.get("has_extension", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Extension present: +{boost}")
        phone_text = re.split(patterns["extension"], phone_text, flags=re.IGNORECASE)[0].strip()

    ###############################################################################
    # SUBSECTION: CONTEXT VALIDATION
    ###############################################################################
    # Check context words
    has_context = any(word in text.lower() for word in context_words)
    if has_context:
        boost = confidence_boosts.get("has_context", 0.1)
        confidence_adjustment += boost
        reasons.append(f"Phone context: +{boost}")

    ###############################################################################
    # SUBSECTION: FORMAT PATTERN MATCHING
    ###############################################################################
    format_matched = False
    format_patterns = [
        ("parentheses", patterns.get("parentheses")),
        ("standard", patterns.get("standard")),
        ("area_code", patterns.get("area_code")),
        ("full_intl", patterns.get("full_intl"))
    ]

    for pattern_name, pattern in format_patterns:
        if pattern and re.search(pattern, phone_text):
            boost = confidence_boosts.get("format_match", 0.2)
            confidence_adjustment += boost
            reasons.append(f"{pattern_name.replace('_', ' ').title()} format: +{boost}")
            format_matched = True
            break

    ###############################################################################
    # SUBSECTION: VALIDATION RESULT
    ###############################################################################
    # Final validity determination
    is_valid = bool(format_matched or has_context)
    
    return {
        "is_valid": is_valid,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }

###############################################################################
# SECTION: GPA ENTITY VALIDATION
###############################################################################

def validate_gpa(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate GPA entities using numeric range and format checks.
    
    Core Validation Logic:
    1. Extract numeric value
    2. Validate range
    3. Check scale format
    4. Verify context indicators
    5. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    indicators = set(rules.get("indicators", []))
    scale_indicators = set(rules.get("scale_indicators", []))
    
    ###############################################################################
    # SUBSECTION: NUMERIC VALUE EXTRACTION
    ###############################################################################
    gpa_text = entity.text.strip()
    text_lower = text.lower()
    reasons = []
    confidence_adjustment = 0.0

    # Extract numeric value
    numeric_match = re.search(r'(-?\d+\.?\d*)', gpa_text)
    if not numeric_match:
        reasons.append("No valid GPA value found")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.5,
            "reasons": reasons
        }

    try:
        gpa_value = float(numeric_match.group(1))
    except ValueError:
        reasons.append("Could not parse GPA value")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.5,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: RANGE VALIDATION
    ###############################################################################
    # Validate value range
    min_value = validation_rules.get("min_value", 0.0)
    max_value = validation_rules.get("max_value", 4.0)

    if gpa_value < min_value or gpa_value > max_value:
        penalty = confidence_penalties.get("invalid_value", -0.5)
        reasons.append(f"GPA value {gpa_value} outside valid range [{min_value}, {max_value}]")
        return {
            "is_valid": False,
            "confidence_adjustment": penalty,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: SCALE FORMAT VALIDATION
    ###############################################################################
    # Check for scale format
    has_scale = '/' in gpa_text
    if has_scale:
        try:
            scale_part = gpa_text.split('/')[1].strip()
            scale_value = float(scale_part)
            
            if scale_value != max_value:
                penalty = confidence_penalties.get("invalid_value", -0.5)
                reasons.append(f"Invalid scale value: {scale_value} (expected {max_value})")
                return {
                    "is_valid": False,
                    "confidence_adjustment": penalty,
                    "reasons": reasons
                }
            
            boost = confidence_boosts.get("has_scale", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Valid scale indicator: +{boost}")
        except ValueError:
            reasons.append("Invalid scale format")
            return {
                "is_valid": False,
                "confidence_adjustment": -0.5,
                "reasons": reasons
            }

    ###############################################################################
    # SUBSECTION: CONTEXT AND INDICATOR VALIDATION
    ###############################################################################
    # Check for GPA indicators
    has_indicator = any(
        indicator.lower() in text_lower or 
        indicator.lower() in gpa_text.lower() 
        for indicator in indicators
    )
    
    # Require indicator if not in scale format
    if not has_indicator and not has_scale:
        if validation_rules.get("require_indicator", True):
            penalty = confidence_penalties.get("no_indicator", -0.3)
            reasons.append("Missing GPA indicator")
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }

    # Add confidence boosts
    if has_indicator:
        boost = confidence_boosts.get("has_indicator", 0.3)
        confidence_adjustment += boost
        reasons.append(f"Found GPA indicator: +{boost}")

    ###############################################################################
    # SUBSECTION: DECIMAL PLACE VALIDATION
    ###############################################################################
    # Check decimal places
    if '.' in str(gpa_value):
        decimal_places = len(str(gpa_value).split('.')[-1])
        if decimal_places > validation_rules.get("decimal_places", 2):
            penalty = confidence_penalties.get("suspicious_format", -0.2)
            reasons.append(f"Too many decimal places: {decimal_places}")
            return {
                "is_valid": False,
                "confidence_adjustment": penalty,
                "reasons": reasons
            }

    ###############################################################################
    # SUBSECTION: RANGE-BASED BOOST
    ###############################################################################
    # Add boost for proper value range
    if 2.0 <= gpa_value <= max_value:
        boost = confidence_boosts.get("proper_value_range", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Valid GPA range: +{boost}")

    return {
        "is_valid": True,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }


###############################################################################
# SECTION: INTERNET REFERENCE ENTITY VALIDATION
###############################################################################

def validate_internet_reference(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate internet references (URLs, social handles) using configuration rules.
    
    Core Validation Logic:
    1. Enforce length requirements
    2. Validate URL or social handle formats
    3. Check known platforms
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    validation_types = rules.get("validation_types", {})
    known_platforms = rules.get("known_platforms", {})
    
    ###############################################################################
    # SUBSECTION: BASIC VALIDATION
    ###############################################################################
    ref_text = entity.text.strip()
    text_lower = text.lower()
    reasons = []
    confidence_adjustment = 0.0

    # Length validation
    if len(ref_text) < validation_rules.get("min_chars", 4):
        reasons.append(f"Reference too short (< {validation_rules.get('min_chars', 4)} chars)")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }
        
    if len(ref_text) > validation_rules.get("max_length", 512):
        reasons.append(f"Reference too long (> {validation_rules.get('max_length', 512)} chars)")
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: URL VALIDATION
    ###############################################################################
    url_config = validation_types.get("url", {})
    url_patterns = url_config.get("patterns", {})
    
    url_match_results = {}
    for name, pattern in url_patterns.items():
        match = re.match(pattern, ref_text, re.IGNORECASE)
        url_match_results[name] = bool(match)
    
    is_url = any(url_match_results.values())

    ###############################################################################
    # SUBSECTION: SOCIAL HANDLE VALIDATION
    ###############################################################################
    handle_config = validation_types.get("social_handle", {})
    handle_patterns = handle_config.get("patterns", {})
    max_handle_length = handle_config.get("max_length", 30)
    
    handle_match_results = {}
    for name, pattern in handle_patterns.items():
        match = re.match(pattern, ref_text, re.IGNORECASE)
        handle_match_results[name] = bool(match)
    
    is_handle = (any(handle_match_results.values()) 
                 and len(ref_text) <= max_handle_length)

    ###############################################################################
    # SUBSECTION: PLATFORM RECOGNITION
    ###############################################################################
    social_platforms = set(known_platforms.get("social", []))
    common_domains = set(known_platforms.get("common_domains", []))

    ###############################################################################
    # SUBSECTION: URL SPECIFIC VALIDATION
    ###############################################################################
    if is_url:
        # Check for known social platforms
        if any(platform in ref_text.lower() for platform in social_platforms):
            boost = confidence_boosts.get("known_platform", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Known social platform URL: +{boost}")
        
        # Check for common domains
        elif any(domain in ref_text.lower() for domain in common_domains):
            boost = confidence_boosts.get("known_platform", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Known domain: +{boost}")
        
        # Check for protocol
        allowed_schemes = url_config.get("allowed_schemes", [])
        if any(scheme + "://" in ref_text.lower() for scheme in allowed_schemes):
            boost = confidence_boosts.get("has_protocol", 0.1)
            confidence_adjustment += boost
            reasons.append(f"Valid URL protocol: +{boost}")
        
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }
    
    ###############################################################################
    # SUBSECTION: SOCIAL HANDLE VALIDATION
    ###############################################################################
    if is_handle:
        # Look for social context in surrounding text
        platform_names = [p.split('.')[0] for p in social_platforms]
        has_social_context = any(platform in text.lower() for platform in platform_names)
        
        if has_social_context:
            boost = confidence_boosts.get("known_platform", 0.2)
            confidence_adjustment += boost
            reasons.append(f"Social media context: +{boost}")
            
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }
    
    ###############################################################################
    # SUBSECTION: INVALID FORMAT HANDLING
    ###############################################################################
    # If it starts with @ but doesn't match handle pattern
    if ref_text.startswith('@'):
        return {
            "is_valid": False,
            "confidence_adjustment": -0.3,
            "reasons": ["Invalid social media handle format"]
        }
        
    return {
        "is_valid": False,
        "confidence_adjustment": -0.3,
        "reasons": ["Not a valid URL or social media handle"]
    }

###############################################################################
# SECTION: PROTECTED CLASS ENTITY VALIDATION
###############################################################################

def validate_protected_class(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate protected class entities using linguistic and contextual analysis.
    
    Core Validation Logic:
    1. Check pronoun patterns
    2. Identify identity terms
    3. Validate contextual references
    4. Apply confidence boosts/penalties
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    categories = rules.get("categories", {})
    
    ###############################################################################
    # SUBSECTION: INITIAL VALIDATION PARAMETERS
    ###############################################################################
    term_match_score = validation_rules.get("term_match_score", 0.3)
    context_match_score = validation_rules.get("context_match_score", 0.3)
    
    entity_text = entity.text.strip()
    text_lower = text.lower()
    reasons = []
    confidence_adjustment = 0.0

    ###############################################################################
    # SUBSECTION: PRONOUN PATTERN CHECKING
    ###############################################################################
    def check_pronoun_patterns(text: str, patterns: Dict) -> Optional[Dict]:
        """Check pronoun patterns in text."""
        for pattern_name, pattern in patterns.items():
            try:
                if re.search(pattern, text, re.IGNORECASE):
                    return {
                        "is_valid": True,
                        "confidence_adjustment": term_match_score,
                        "reasons": [f"Pronoun pattern match ({pattern_name})"]
                    }
            except re.error:
                debug_logger.warning(f"Invalid regex pattern: {pattern}")
        return None

    pronoun_result = check_pronoun_patterns(
        entity_text, 
        categories.get("pronoun_patterns", {})
    )
    if pronoun_result:
        return pronoun_result

    ###############################################################################
    # SUBSECTION: IDENTITY TERM IDENTIFICATION
    ###############################################################################
    def find_identity_term(text: str, categories: Dict) -> Optional[str]:
        """Find a matching identity term in text."""
        all_terms = (
            categories.get("gender_patterns", {}).get("gender_terms", []) +
            categories.get("religious_patterns", {}).get("religious_terms", []) +
            categories.get("orientation", []) +
            categories.get("race_ethnicity", [])
        )
        text_lower = text.lower()
        for term in all_terms:
            if term.lower() in text_lower:
                return term
        return None

    identity_term = find_identity_term(entity_text, categories)

    ###############################################################################
    # SUBSECTION: CONTEXT CHECKING
    ###############################################################################
    def get_identity_contexts(categories: Dict) -> List[str]:
        """Get merged context phrases from identity and organization keys."""
        contexts = []
        if "gender_patterns" in categories:
            contexts.extend(
                categories["gender_patterns"].get("identity_context", []) +
                categories["gender_patterns"].get("org_context", [])
            )
        if "religious_patterns" in categories:
            contexts.extend(
                categories["religious_patterns"].get("identity_context", []) +
                categories["religious_patterns"].get("org_context", []) +
                categories["religious_patterns"].get("leadership_roles", [])
            )
        return contexts

    def check_identity_context(
        text: str, 
        identity_term: str, 
        contexts: List[str], 
        boosts: Dict
    ) -> Tuple[bool, float]:
        """Check for valid identity or organizational context."""
        text_lower = text.lower()
        words = text_lower.split()
        
        # Direct organizational keyword check
        org_keywords = ["organization", "church", "network", "association", "society"]
        for org_kw in org_keywords:
            if org_kw in text_lower:
                boost = boosts.get("org_context_match", 0.3)
                return True, boost

        # Context phrase checking
        for ctx in contexts:
            ctx_lower = ctx.lower()
            # Full context match
            if ctx_lower in text_lower:
                if any(org_kw in ctx_lower for org_kw in org_keywords):
                    boost = boosts.get("org_context_match", 0.3)
                else:
                    boost = boosts.get("gender_context_match", 0.3)
                return True, boost
            
            # Prefix match
            for w in words:
                if len(ctx_lower) >= 4 and len(w) >= 4 and w[:4] == ctx_lower[:4]:
                    if any(org_kw in ctx_lower for org_kw in org_keywords):
                        boost = boosts.get("org_context_match", 0.3)
                    else:
                        boost = boosts.get("gender_context_match", 0.3)
                    return True, boost
        
        return False, 0.0

    ###############################################################################
    # SUBSECTION: VALIDATION RESULT DETERMINATION
    ###############################################################################
    if identity_term:
        identity_contexts = get_identity_contexts(categories)
        context_found, context_boost = check_identity_context(
            text, 
            identity_term, 
            identity_contexts, 
            confidence_boosts
        )
        
        if context_found:
            confidence_adjustment += context_boost
            reasons.append(f"Identity context found: +{context_boost}")
            return {
                "is_valid": True,
                "confidence_adjustment": confidence_adjustment,
                "reasons": reasons
            }

    ###############################################################################
    # SUBSECTION: FALLBACK AND PENALTY
    ###############################################################################
    penalty = confidence_penalties.get("no_context", -0.4)
    confidence_adjustment += penalty
    reasons.append(f"No protected class context: {penalty}")
    
    return {
        "is_valid": False,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }

###############################################################################
# SECTION: PHI (PROTECTED HEALTH INFORMATION) ENTITY VALIDATION
###############################################################################

def validate_phi(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate Protected Health Information entities using comprehensive rules.
    
    Core Validation Logic:
    1. Check medical condition patterns
    2. Validate context and terminology
    3. Apply confidence boosts/penalties
    4. Handle sensitivity levels
    
    Args:
        entity: The Entity object to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
    
    Returns:
        Validation result dictionary
    """
    # Debug logging setup
    debug_logger = logger or logging.getLogger(__name__)
    
    # Extract validation configurations
    validation_rules = rules.get("validation_rules", {})
    confidence_boosts = rules.get("confidence_boosts", {})
    confidence_penalties = rules.get("confidence_penalties", {})
    categories = rules.get("categories", {})
    patterns = rules.get("patterns", {})
    
    ###############################################################################
    # SUBSECTION: INITIAL VALIDATION PARAMETERS
    ###############################################################################
    phi_text = entity.text.strip()
    text_lower = text.lower()
    reasons = []
    confidence_adjustment = 0.0

    # Always valid conditions check
    always_valid_conditions = validation_rules.get("always_valid_conditions", [])
    if any(condition.lower() in phi_text.lower() for condition in always_valid_conditions):
        boost = confidence_boosts.get("explicit_condition_disclosure", 0.3)
        confidence_adjustment += boost
        reasons.append(f"Explicitly listed condition: +{boost}")
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: MEDICAL CONDITION PATTERN MATCHING
    ###############################################################################
    def check_medical_patterns(text: str, conditions: Dict) -> Optional[Dict]:
        """Check for specific medical condition patterns."""
        all_conditions = {
            condition.lower() 
            for category in conditions.values() 
            for condition in category
        }
        
        for condition in all_conditions:
            if condition in text.lower():
                return {
                    "is_valid": True,
                    "boost": 0.4,
                    "reason": f"Medical condition match ({condition})"
                }
        return None

    condition_result = check_medical_patterns(
        phi_text, 
        categories.get("medical_conditions", {})
    )
    if condition_result:
        boost = condition_result["boost"]
        confidence_adjustment += boost
        reasons.append(condition_result["reason"])
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: CONTEXT WORD VALIDATION
    ###############################################################################
    context_requirements = rules.get("generic_term_context_requirements", {})
    context_words = context_requirements.get("required_context_words", [])
    min_required = context_requirements.get("minimum_context_words", 1)
    
    # Split context on delimiters to catch terms in different formats
    context_tokens = re.split(r'[:\s,]+', text_lower)
    matching_contexts = [word for word in context_words if word in context_tokens]
    
    if len(matching_contexts) >= min_required:
        boost = confidence_boosts.get("has_medical_context", 0.4)
        confidence_adjustment += boost
        reasons.append(f"Medical context terms found: {', '.join(matching_contexts[:2])}: +{boost}")
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: SPECIFIC PATTERN CHECKS
    ###############################################################################
    specific_patterns = patterns.get("specific_conditions", {}).get("regex", [])
    for pattern in specific_patterns:
        if re.search(pattern, text_lower, flags=re.IGNORECASE):
            boost = confidence_boosts.get("condition_specific_match", 0.4)
            confidence_adjustment += boost
            reasons.append(f"Matched specific condition pattern: +{boost}")
            return {
                "is_valid": True,
                "confidence_adjustment": confidence_adjustment,
                "reasons": reasons
            }

    ###############################################################################
    # SUBSECTION: TREATMENT AND PROCEDURE CHECKS
    ###############################################################################
    treatments = []
    for treatment_type in categories.get("treatments", {}).values():
        treatments.extend(treatment_type)
    
    # Check if text contains treatment and context has condition terms
    is_treatment = any(t.lower() in phi_text.lower() for t in treatments)
    has_condition = any(
        word in text_lower 
        for word in context_words 
        if word in ['condition', 'diagnosis', 'diagnosed']
    )
    
    if is_treatment and has_condition:
        boost = confidence_boosts.get("treatment_context_match", 0.2)
        confidence_adjustment += boost
        reasons.append(f"Treatment with condition context: +{boost}")
        return {
            "is_valid": True,
            "confidence_adjustment": confidence_adjustment,
            "reasons": reasons
        }

    ###############################################################################
    # SUBSECTION: FALLBACK AND PENALTY
    ###############################################################################
    penalty = confidence_penalties.get("no_context", -0.7)
    confidence_adjustment += penalty
    reasons.append(f"No medical context: {penalty}")
    
    return {
        "is_valid": False,
        "confidence_adjustment": confidence_adjustment,
        "reasons": reasons
    }
###############################################################################
# SECTION N: DATE VALIDATION
###############################################################################

def validate_date(
    entity: Entity, 
    text: str, 
    rules: Dict, 
    logger: Optional[logging.Logger] = None,
    source: Optional[str] = None
) -> Dict:
    """
    Multi-strategy date/time validation supporting ensemble, Presidio, and SpaCy approaches.
    
    Args:
        entity: Entity to validate
        text: Context text
        rules: Validation configuration rules
        logger: Optional logger for debug information
        source: Optional detector source (ensemble, spacy, presidio)
    
    Returns:
        Validation result dictionary
    """
    # Default to presidio if no source specified
    if source is None:
        source = "presidio"
    
    def _validate_presidio_date(entity, text, rules):
        """
        Presidio-specific date validation focusing on regex patterns.
        
        Args:
            entity: Entity to validate
            text: Context text
            rules: Validation configuration rules
        
        Returns:
            Dictionary with validation results
        """
        patterns = rules.get("patterns", {})
        confidence_adjustments = rules.get("confidence_adjustments", {})
        year_markers = set(rules.get("year_markers", []))
        text_lower = text.lower()
        total_confidence = 0.0
        reasons = []
    
        # Add year marker boost first (always check context)
        if any(marker in text_lower for marker in year_markers):
            boost = confidence_adjustments.get("has_marker", 0.2)
            total_confidence += boost
            reasons.append(f"Year marker context: +{boost}")
    
        # Pattern priority order
        pattern_priority = ["education_year", "year_range", "year_month", "month_year", "single_year"]
        
        # Detailed pattern matching
        for pattern_name in pattern_priority:
            pattern_info = patterns.get(pattern_name, {})
            pattern = pattern_info.get("regex", "")
            
            try:
                # Handle confidence values
                if isinstance(pattern_info.get("confidence"), dict):
                    conf = pattern_info.get("confidence", {}).get("confidence", 0.7)
                else:
                    conf = pattern_info.get("confidence", 0.7)
    
                # Regex matching
                if pattern and re.match(pattern, entity.text, re.IGNORECASE):
                    # Adjust confidence for specific patterns
                    if pattern_name == "education_year":
                        conf = 1.0
                    elif pattern_name == "year_range":
                        conf = 0.85
                    
                    total_confidence += conf
                    reasons.append(f"Matched {pattern_name} pattern: +{conf}")
                    
                    return {
                        "is_valid": True,
                        "confidence_adjustment": total_confidence,
                        "reasons": reasons
                    }
            except Exception as e:
                logging.warning(f"Error processing {pattern_name}: {str(e)}")
    
        # Fallback return
        return {
            "is_valid": bool(reasons),
            "confidence_adjustment": total_confidence,
            "reasons": reasons
        }

    def _validate_spacy_date(entity, text, rules):
        """
        SpaCy-specific date validation focusing on linguistic patterns.
        
        Args:
            entity: Entity to validate
            text: Context text
            rules: Validation configuration rules
        
        Returns:
            Dictionary with validation results
        """
        confidence_adjustments = rules.get("confidence_adjustments", {})
        year_markers = set(rules.get("year_markers", []))
        text_lower = text.lower()
        total_confidence = 0.0
        reasons = []
    
        # Check for year markers
        if any(marker in text_lower for marker in year_markers):
            boost = confidence_adjustments.get("has_marker", 0.2)
            total_confidence += boost
            reasons.append(f"Year marker context: +{boost}")
    
        # Check education year pattern first
        education_pattern = rules.get("patterns", {}).get("education_year", {})
        if education_pattern:
            pattern = education_pattern.get("regex", "")
            if pattern and re.match(pattern, entity.text, re.IGNORECASE):
                conf = education_pattern.get("confidence", 1.0)
                total_confidence += conf
                reasons.append(f"Education year pattern: +{conf}")
                return {
                    "is_valid": True,
                    "confidence_adjustment": total_confidence,
                    "reasons": reasons
                }
    
        # Fallback return
        return {
            "is_valid": bool(reasons),
            "confidence_adjustment": total_confidence,
            "reasons": reasons
        }

    # Ensemble configuration retrieval
    def _get_ensemble_config(rules):
        """
        Retrieve ensemble configuration for DATE_TIME validation.
        
        Args:
            rules: Validation configuration rules
        
        Returns:
            Dictionary with ensemble configuration for DATE_TIME
        """
        # Default weights if not explicitly specified
        default_config = {
            "presidio_weight": 0.65,
            "spacy_weight": 0.35,
            "minimum_combined": 0.7
        }
        
        # Try to get ensemble configuration from rules
        ensemble_config = rules.get("ensemble_required", {}).get("confidence_thresholds", {}).get("DATE_TIME", {})
        
        # Merge default config with any specified configuration
        return {**default_config, **ensemble_config}



    def _get_ensemble_config(rules):
        """
        Retrieve ensemble configuration for DATE_TIME validation.
        
        Args:
            rules: Validation configuration rules
        
        Returns:
            Dictionary with ensemble configuration for DATE_TIME
        """
        # Default weights if not explicitly specified
        default_config = {
            "presidio_weight": 0.65,
            "spacy_weight": 0.35,
            "minimum_combined": 0.7
        }
        
        # Try to get ensemble configuration from rules
        ensemble_config = rules.get("ensemble_required", {}).get("confidence_thresholds", {}).get("DATE_TIME", {})
        
        # Merge default config with any specified configuration
        return {**default_config, **ensemble_config}

    # Main validation logic
    if source == "ensemble":
        ensemble_config = _get_ensemble_config(rules)
        presidio_weight = ensemble_config.get("presidio_weight", 0.5)
        spacy_weight = ensemble_config.get("spacy_weight", 0.5)
        
        return _validate_ensemble_date(entity, text, rules, presidio_weight, spacy_weight)
    
    elif source == "spacy":
        return _validate_spacy_date(entity, text, rules)
    
    else:  # Default to Presidio
        return _validate_presidio_date(entity, text, rules)




###############END OF ENTITIES###################

# Module-level method mapping (fallback if config fails)
_DEFAULT_METHOD_MAP = {
    "ADDRESS": validate_address,
    "DATE_TIME": validate_date,
    "EDUCATIONAL_INSTITUTION": validate_educational_institution,
    "EMAIL_ADDRESS": validate_email,
    "GPA": validate_gpa,
    "INTERNET_REFERENCE": validate_internet_reference,
    "URL": validate_internet_reference,
    "IP_ADDRESS": validate_internet_reference,
    "LOCATION": validate_location,
    "PERSON": validate_person,
    "PHI": validate_phi,
    "PHONE_NUMBER": validate_phone,
    "PROTECTED_CLASS": validate_protected_class
}


###############################################################################
# End of entity_validation.py
###############################################################################