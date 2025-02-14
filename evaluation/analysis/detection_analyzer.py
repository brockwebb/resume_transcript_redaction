# /evaluation/analysis/detection_analyzer.py

###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from evaluation.models import Entity
import logging

###############################################################################
# SECTION 2: DATA MODELS
###############################################################################
@dataclass
class DetectionAnalysis:
    """Contains comprehensive analysis of entity detection."""
    entity: Entity
    detector_info: Dict  # Raw detector data
    validation_info: Dict  # Validation steps/adjustments
    context_info: Dict  # Surrounding text analysis
    confidence_trace: List[Dict]  # Track confidence changes

###############################################################################
# SECTION 3: DETECTION ANALYZER CLASS
###############################################################################
class DetectionAnalyzer:
    """
    Analyzes entity detection process and results.
    Provides detailed insight into detection decisions and validation steps.
    """

###############################################################################
# SECTION 4: CLASS INITIALIZATION AND CONFIGURATION
###############################################################################
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize detection analyzer with configuration.
        
        Args:
            config_loader: Configuration loader instance
            logger: Optional logger for debug information
        """
        self.logger = logger
        self.config_loader = config_loader
        self.debug_mode = False
        
        # Set debug mode based on logger configuration
        if logger:
            if hasattr(logger, 'level'):
                self.debug_mode = logger.level == 10
            elif hasattr(logger, 'getEffectiveLevel'):
                self.debug_mode = logger.getEffectiveLevel() == 10
                
        # Cache needed configurations
        self._cached_configs = {}
        self._cache_configs()
        
        if self.debug_mode:
            self.logger.debug("Initialized DetectionAnalyzer")
            
    def _cache_configs(self) -> None:
        """Cache commonly used configurations."""
        try:
            config_names = ["validation_params", "entity_routing"]
            for name in config_names:
                config = self.config_loader.get_config(name)
                if config:
                    self._cached_configs[name] = config
                elif self.debug_mode:
                    self.logger.debug(f"No config found for {name}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error caching configs: {str(e)}")

###############################################################################
# SECTION 5: ANALYSIS METHODS
###############################################################################
    def analyze_detection(self, 
                        entity: Entity, 
                        original_text: str,
                        truth_set: Optional[List[Entity]] = None) -> DetectionAnalysis:
        """
        Perform comprehensive analysis of entity detection.
        
        Args:
            entity: Detected entity to analyze
            original_text: Full text where entity was detected
            truth_set: Optional ground truth entities for comparison
            
        Returns:
            DetectionAnalysis containing detailed analysis results
        """
        try:
            # Get detector-specific information
            detector_info = self._analyze_detector_output(entity, original_text)
            
            # Get validation process details
            validation_info = self._analyze_validation_process(entity)
            
            # Analyze surrounding context
            context_info = self._analyze_context(entity, original_text)
            
            # Build confidence adjustment trace
            confidence_trace = self._build_confidence_trace(entity)
            
            return DetectionAnalysis(
                entity=entity,
                detector_info=detector_info,
                validation_info=validation_info,
                context_info=context_info,
                confidence_trace=confidence_trace
            )
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error in detection analysis: {str(e)}")
            raise

###############################################################################
# SECTION 6: DETECTOR-SPECIFIC ANALYSIS
###############################################################################
    def _analyze_detector_output(self, 
                               entity: Entity, 
                               text: str) -> Dict:
        """
        Analyze raw detector output based on detection source.
        
        Args:
            entity: Entity to analyze
            text: Original text where entity was detected
            
        Returns:
            Dict containing detector-specific analysis
        """
        detector_source = entity.detector_source or "unknown"
        
        if detector_source == "spacy":
            return self._analyze_spacy_detection(entity, text)
        elif detector_source == "presidio":
            return self._analyze_presidio_detection(entity, text)
        elif detector_source == "ensemble":
            return self._analyze_ensemble_detection(entity, text)
        else:
            return {
                "source": detector_source,
                "confidence": entity.confidence,
                "error": "Unknown detector source"
            }

    def _analyze_spacy_detection(self, entity: Entity, text: str) -> Dict:
        """
        Analyze spaCy-specific detection features.
        
        Args:
            entity: Entity detected by spaCy
            text: Original text
            
        Returns:
            Dict with spaCy-specific analysis
        """
        # Get entity thresholds from routing config
        routing_config = self._cached_configs.get("entity_routing", {})
        spacy_config = routing_config.get("routing", {}).get("spacy_primary", {})
        threshold = spacy_config.get("thresholds", {}).get(entity.entity_type)
        
        # Extract context window
        start = max(0, entity.start_char - 50)
        end = min(len(text), entity.end_char + 50)
        context = text[start:end]
        
        return {
            "source": "spacy",
            "initial_confidence": entity.original_confidence,
            "final_confidence": entity.confidence,
            "threshold": threshold,
            "at_threshold": abs(entity.confidence - threshold) < 0.01 if threshold else False,
            "context_window": context,
            "linguistic_features": {
                "all_caps": entity.text.isupper(),
                "title_case": entity.text.istitle(),
                "contains_digits": any(c.isdigit() for c in entity.text),
                "word_count": len(entity.text.split())
            }
        }

    def _analyze_presidio_detection(self, entity: Entity, text: str) -> Dict:
        """
        Analyze Presidio-specific detection features.
        
        Args:
            entity: Entity detected by Presidio
            text: Original text
            
        Returns:
            Dict with Presidio-specific analysis
        """
        # Get entity thresholds from routing config
        routing_config = self._cached_configs.get("entity_routing", {})
        presidio_config = routing_config.get("routing", {}).get("presidio_primary", {})
        threshold = presidio_config.get("thresholds", {}).get(entity.entity_type)
        
        return {
            "source": "presidio",
            "initial_confidence": entity.original_confidence,
            "final_confidence": entity.confidence,
            "threshold": threshold,
            "at_threshold": abs(entity.confidence - threshold) < 0.01 if threshold else False,
            "pattern_features": {
                "length": len(entity.text),
                "contains_spaces": " " in entity.text,
                "contains_punctuation": any(not c.isalnum() for c in entity.text)
            }
        }

    def _analyze_ensemble_detection(self, entity: Entity, text: str) -> Dict:
        """
        Analyze ensemble detection features.
        
        Args:
            entity: Entity detected by ensemble
            text: Original text
            
        Returns:
            Dict with ensemble-specific analysis
        """
        # Get ensemble thresholds from routing config
        routing_config = self._cached_configs.get("entity_routing", {})
        ensemble_config = routing_config.get("routing", {}).get("ensemble_required", {})
        thresholds = ensemble_config.get("confidence_thresholds", {}).get(entity.entity_type, {})
        
        return {
            "source": "ensemble",
            "initial_confidence": entity.original_confidence,
            "final_confidence": entity.confidence,
            "threshold": thresholds.get("minimum_combined"),
            "weights": {
                "presidio": thresholds.get("presidio_weight"),
                "spacy": thresholds.get("spacy_weight")
            },
            "combined_features": {
                "length": len(entity.text),
                "word_count": len(entity.text.split()),
                "contains_spaces": " " in entity.text
            }
        }

###############################################################################
# SECTION 7: VALIDATION ANALYSIS
###############################################################################
   def _analyze_validation_process(self, entity: Entity) -> Dict:
       """
       Analyze validation process and rule applications.
       
       Args:
           entity: Entity to analyze validation for
           
       Returns:
           Dict containing validation analysis
       """
       validation_params = self._cached_configs.get("validation_params", {})
       entity_rules = validation_params.get(entity.entity_type, {})
       
       # Track validation rule applications
       validation_trace = []
       original_confidence = entity.original_confidence or entity.confidence
       current_confidence = original_confidence
       
       # Analyze confidence boosts
       applied_boosts = []
       boost_rules = entity_rules.get("confidence_boosts", {})
       for rule_name, boost_value in boost_rules.items():
           if self._check_boost_applied(entity, rule_name):
               applied_boosts.append({
                   "rule": rule_name,
                   "value": boost_value,
                   "triggered": True
               })
               current_confidence += boost_value
               validation_trace.append({
                   "stage": "boost",
                   "rule": rule_name,
                   "adjustment": boost_value,
                   "confidence_after": min(1.0, max(0.0, current_confidence))
               })

       # Analyze confidence penalties
       applied_penalties = []
       penalty_rules = entity_rules.get("confidence_penalties", {})
       for rule_name, penalty_value in penalty_rules.items():
           if self._check_penalty_applied(entity, rule_name):
               applied_penalties.append({
                   "rule": rule_name,
                   "value": penalty_value,
                   "triggered": True
               })
               current_confidence += penalty_value  # Penalties are negative values
               validation_trace.append({
                   "stage": "penalty",
                   "rule": rule_name,
                   "adjustment": penalty_value,
                   "confidence_after": min(1.0, max(0.0, current_confidence))
               })

       return {
           "validation_rules": entity_rules,
           "applied_boosts": applied_boosts,
           "applied_penalties": applied_penalties,
           "validation_trace": validation_trace,
           "confidence_impact": {
               "original": original_confidence,
               "final": entity.confidence,
               "total_adjustment": entity.confidence - original_confidence
           }
       }

   def _check_boost_applied(self, entity: Entity, rule_name: str) -> bool:
       """
       Check if a specific confidence boost was applied.
       
       Args:
           entity: Entity to check
           rule_name: Name of boost rule to check
           
       Returns:
           bool indicating if boost was applied
       """
       if not entity.validation_rules:
           return False
           
       # Look for boost marker in validation rules
       boost_marker = f"confidence_adj_+"
       matching_rules = [
           rule for rule in entity.validation_rules
           if rule.startswith(boost_marker) and rule_name.lower() in rule.lower()
       ]
       
       return len(matching_rules) > 0

   def _check_penalty_applied(self, entity: Entity, rule_name: str) -> bool:
       """
       Check if a specific confidence penalty was applied.
       
       Args:
           entity: Entity to check
           rule_name: Name of penalty rule to check
           
       Returns:
           bool indicating if penalty was applied
       """
       if not entity.validation_rules:
           return False
           
       # Look for penalty marker in validation rules
       penalty_marker = f"confidence_adj_-"
       matching_rules = [
           rule for rule in entity.validation_rules
           if rule.startswith(penalty_marker) and rule_name.lower() in rule.lower()
       ]
       
       return len(matching_rules) > 0

   def _build_confidence_trace(self, entity: Entity) -> List[Dict]:
       """
       Build complete trace of confidence adjustments.
       
       Args:
           entity: Entity to analyze confidence changes
           
       Returns:
           List of confidence adjustment steps
       """
       trace = [{
           "stage": "initial",
           "confidence": entity.original_confidence or entity.confidence,
           "adjustment": 0.0,
           "reason": "Initial detection confidence"
       }]
       
       if entity.validation_rules:
           current_confidence = entity.original_confidence or entity.confidence
           
           for rule in entity.validation_rules:
               if rule.startswith("confidence_adj"):
                   try:
                       # Extract adjustment value
                       adj_str = rule.split("_")[-1]
                       adjustment = float(adj_str)
                       
                       current_confidence = min(1.0, max(0.0, current_confidence + adjustment))
                       
                       trace.append({
                           "stage": "validation",
                           "confidence": current_confidence,
                           "adjustment": adjustment,
                           "reason": rule
                       })
                   except ValueError:
                       continue
       
       # Add final confidence if different from last trace entry
       if trace[-1]["confidence"] != entity.confidence:
           trace.append({
               "stage": "final",
               "confidence": entity.confidence,
               "adjustment": entity.confidence - trace[-1]["confidence"],
               "reason": "Final validation adjustment"
           })
           
       return trace

###############################################################################
# SECTION 8: CONTEXT ANALYSIS
###############################################################################
   def _analyze_context(self, entity: Entity, text: str) -> Dict:
       """
       Analyze surrounding context of detected entity.
       
       Args:
           entity: Entity to analyze context for
           text: Original text
           
       Returns:
           Dict containing context analysis
       """
       # Get text window around entity
       window_size = 50  # Characters before/after
       start = max(0, entity.start_char - window_size)
       end = min(len(text), entity.end_char + window_size)
       
       preceding_text = text[start:entity.start_char].strip()
       following_text = text[entity.end_char:end].strip()
       
       # Break context into words
       preceding_words = preceding_text.split()[-5:]  # Last 5 words
       following_words = following_text.split()[:5]   # First 5 words
       
       # Get entity type specific patterns
       patterns = self._get_context_patterns(entity.entity_type)
       
       context_analysis = {
           "text_windows": {
               "preceding": preceding_text,
               "following": following_text,
               "full_context": text[start:end]
           },
           "word_context": {
               "preceding_words": preceding_words,
               "following_words": following_words
           },
           "patterns": {
               "matches": self._analyze_context_patterns(
                   text[start:end], 
                   patterns
               )
           },
           "indicators": self._analyze_context_indicators(
               text[start:end],
               entity
           )
       }

       return context_analysis

   def _get_context_patterns(self, entity_type: str) -> Dict:
       """
       Get relevant context patterns for entity type.
       
       Args:
           entity_type: Type of entity to get patterns for
           
       Returns:
           Dict of context patterns
       """
       validation_params = self._cached_configs.get("validation_params", {})
       entity_config = validation_params.get(entity_type, {})
       
       patterns = {
           "positive_indicators": set(),
           "negative_indicators": set(),
           "software_terms": {
               "library", "framework", "tool", "package", "module",
               "version", "release", "documentation", "docs"
           }
       }
       
       # Add entity-specific patterns
       if entity_type == "PERSON":
           patterns["positive_indicators"].update(
               entity_config.get("title_markers", [])
           )
           patterns["negative_indicators"].update(
               patterns["software_terms"]
           )
           
       elif entity_type == "EDUCATIONAL_INSTITUTION":
           patterns["positive_indicators"].update(
               entity_config.get("core_terms", [])
           )
           patterns["negative_indicators"].update(
               entity_config.get("generic_terms", [])
           )
           
       return patterns

   def _analyze_context_patterns(self, context: str, patterns: Dict) -> Dict:
       """
       Analyze context for pattern matches.
       
       Args:
           context: Text context to analyze
           patterns: Patterns to check for
           
       Returns:
           Dict containing pattern matches
       """
       context_lower = context.lower()
       
       matches = {
           "positive": [],
           "negative": [],
           "software_related": False
       }
       
       # Check positive indicators
       for indicator in patterns["positive_indicators"]:
           if indicator.lower() in context_lower:
               matches["positive"].append(indicator)
               
       # Check negative indicators
       for indicator in patterns["negative_indicators"]:
           if indicator.lower() in context_lower:
               matches["negative"].append(indicator)
               
       # Check software terms
       matches["software_related"] = any(
           term.lower() in context_lower 
           for term in patterns["software_terms"]
       )
       
       return matches

   def _analyze_context_indicators(self, context: str, entity: Entity) -> Dict:
       """
       Analyze context for entity-specific indicators.
       
       Args:
           context: Text context to analyze
           entity: Entity for context analysis
           
       Returns:
           Dict containing indicator analysis
       """
       indicators = {
           "likely_false_positive": False,
           "confidence_impact": 0.0,
           "reasons": []
       }
       
       # Software/technology context check for PERSON entities
       if entity.entity_type == "PERSON":
           software_indicators = {
               "using", "import", "install", "version", "release",
               "documentation", "package", "library", "framework"
           }
           
           tech_words = [word for word in context.lower().split() 
                        if word in software_indicators]
           
           if tech_words:
               indicators["likely_false_positive"] = True
               indicators["confidence_impact"] -= 0.2
               indicators["reasons"].append(
                   f"Software/technology context words: {', '.join(tech_words)}"
               )
               
       # Add more entity-specific context analysis as needed
       
       return indicators

###############################################################################
# SECTION 9: REPORT GENERATION
###############################################################################
   def generate_detection_report(self, 
                               analysis: DetectionAnalysis,
                               format: str = "text") -> str:
       """
       Generate formatted report from detection analysis.
       
       Args:
           analysis: DetectionAnalysis to generate report for
           format: Output format ('text' or 'json')
           
       Returns:
           Formatted report string
       """
       if format == "json":
           return self._generate_json_report(analysis)
       
       # Default to text report
       return self._generate_text_report(analysis)

   def _generate_text_report(self, analysis: DetectionAnalysis) -> str:
       """
       Generate human-readable text report.
       
       Args:
           analysis: DetectionAnalysis to report on
           
       Returns:
           Formatted text report
       """
       report = []
       entity = analysis.entity
       
       # Basic entity information
       report.append("Entity Detection Analysis Report")
       report.append("=" * 40)
       report.append(f"\nEntity: '{entity.text}'")
       report.append(f"Type: {entity.entity_type}")
       report.append(f"Position: chars {entity.start_char}-{entity.end_char}")
       report.append(f"Detector: {entity.detector_source}")
       
       # Detector-specific information
       report.append("\nDetection Details:")
       report.append("-" * 20)
       detector_info = analysis.detector_info
       report.append(f"Initial Confidence: {detector_info.get('initial_confidence', 'N/A'):.2f}")
       report.append(f"Final Confidence: {detector_info.get('final_confidence', 'N/A'):.2f}")
       report.append(f"Threshold: {detector_info.get('threshold', 'N/A')}")
       
       # Validation information
       report.append("\nValidation Process:")
       report.append("-" * 20)
       validation = analysis.validation_info
       
       if validation.get("applied_boosts"):
           report.append("\nApplied Boosts:")
           for boost in validation["applied_boosts"]:
               report.append(f"  • {boost['rule']}: +{boost['value']}")
               
       if validation.get("applied_penalties"):
           report.append("\nApplied Penalties:")
           for penalty in validation["applied_penalties"]:
               report.append(f"  • {penalty['rule']}: {penalty['value']}")
       
       # Confidence trace
       report.append("\nConfidence Adjustments:")
       report.append("-" * 20)
       for step in validation.get("validation_trace", []):
           report.append(
               f"  • {step['stage']}: {step.get('rule', 'N/A')} "
               f"-> {step.get('confidence_after', 'N/A'):.2f}"
           )
       
       # Context analysis
       report.append("\nContext Analysis:")
       report.append("-" * 20)
       context = analysis.context_info
       
       # Show surrounding text
       windows = context["text_windows"]
       report.append("\nSurrounding Text:")
       report.append(f"  Before: '...{windows['preceding']}'")
       report.append(f"  After: '{windows['following']}...'")
       
       # Show pattern matches
       patterns = context["patterns"]["matches"]
       if patterns.get("positive"):
           report.append("\nPositive Indicators Found:")
           for indicator in patterns["positive"]:
               report.append(f"  • {indicator}")
               
       if patterns.get("negative"):
           report.append("\nNegative Indicators Found:")
           for indicator in patterns["negative"]:
               report.append(f"  • {indicator}")
       
       # Show specific indicators
       indicators = context["indicators"]
       if indicators.get("likely_false_positive"):
           report.append("\nPotential False Positive:")
           for reason in indicators["reasons"]:
               report.append(f"  • {reason}")
       
       return "\n".join(report)

   def _generate_json_report(self, analysis: DetectionAnalysis) -> str:
       """
       Generate JSON format report.
       
       Args:
           analysis: DetectionAnalysis to report on
           
       Returns:
           JSON formatted report string
       """
       import json
       from dataclasses import asdict
       
       # Convert analysis to dict and clean up for JSON
       report_dict = {
           "entity": {
               "text": analysis.entity.text,
               "type": analysis.entity.entity_type,
               "position": {
                   "start": analysis.entity.start_char,
                   "end": analysis.entity.end_char
               },
               "detector": analysis.entity.detector_source,
               "confidence": analysis.entity.confidence
           },
           "detection_analysis": analysis.detector_info,
           "validation_analysis": analysis.validation_info,
           "context_analysis": analysis.context_info
       }
       
       return json.dumps(report_dict, indent=2)

###############################################################################
# SECTION 10: BATCH ANALYSIS
###############################################################################
   def analyze_batch(self, 
                    entities: List[Entity],
                    text: str,
                    truth_set: Optional[List[Entity]] = None) -> List[DetectionAnalysis]:
       """
       Analyze multiple entities from same text.
       
       Args:
           entities: List of entities to analyze
           text: Original text
           truth_set: Optional ground truth entities
           
       Returns:
           List of DetectionAnalysis results
       """
       analyses = []
       for entity in entities:
           analysis = self.analyze_detection(
               entity=entity,
               original_text=text,
               truth_set=truth_set
           )
           analyses.append(analysis)
           
       return analyses

   def generate_batch_summary(self, 
                            analyses: List[DetectionAnalysis],
                            format: str = "text") -> str:
       """
       Generate summary report for batch analysis.
       
       Args:
           analyses: List of analyses to summarize
           format: Output format ('text' or 'json')
           
       Returns:
           Formatted summary report
       """
       if format == "json":
           return self._generate_json_batch_summary(analyses)
           
       return self._generate_text_batch_summary(analyses)

###############################################################################
# SECTION 11: BATCH SUMMARY GENERATION
###############################################################################
   def _generate_text_batch_summary(self, 
                                  analyses: List[DetectionAnalysis]) -> str:
       """
       Generate text summary of batch analysis results.
       
       Args:
           analyses: List of analyses to summarize
           
       Returns:
           Text summary report
       """
       summary = []
       
       # Overall statistics
       summary.append("Batch Analysis Summary")
       summary.append("=" * 40)
       
       total_entities = len(analyses)
       by_type = {}
       by_detector = {}
       confidence_ranges = {
           "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
           "0.6-0.8": 0, "0.8-1.0": 0
       }
       
       likely_false_positives = []
       validation_stats = {
           "total_adjustments": 0,
           "positive_adjustments": 0,
           "negative_adjustments": 0
       }
       
       # Collect statistics
       for analysis in analyses:
           entity = analysis.entity
           
           # Count by type
           by_type[entity.entity_type] = by_type.get(entity.entity_type, 0) + 1
           
           # Count by detector
           by_detector[entity.detector_source] = by_detector.get(entity.detector_source, 0) + 1
           
           # Track confidence ranges
           conf = entity.confidence
           for range_str in confidence_ranges:
               low, high = map(float, range_str.split("-"))
               if low <= conf < high:
                   confidence_ranges[range_str] += 1
           
           # Track validation adjustments
           if analysis.validation_info:
               adj = analysis.validation_info.get("confidence_impact", {}).get("total_adjustment", 0)
               if adj != 0:
                   validation_stats["total_adjustments"] += 1
                   if adj > 0:
                       validation_stats["positive_adjustments"] += 1
                   else:
                       validation_stats["negative_adjustments"] += 1
           
           # Track potential false positives
           if analysis.context_info["indicators"].get("likely_false_positive"):
               likely_false_positives.append({
                   "text": entity.text,
                   "type": entity.entity_type,
                   "reasons": analysis.context_info["indicators"]["reasons"]
               })
       
       # Generate summary sections
       summary.append(f"\nTotal Entities Analyzed: {total_entities}")
       
       summary.append("\nBy Entity Type:")
       for entity_type, count in by_type.items():
           summary.append(f"  • {entity_type}: {count}")
           
       summary.append("\nBy Detector:")
       for detector, count in by_detector.items():
           summary.append(f"  • {detector}: {count}")
           
       summary.append("\nConfidence Distribution:")
       for range_str, count in confidence_ranges.items():
           if count > 0:
               pct = (count / total_entities) * 100
               summary.append(f"  • {range_str}: {count} ({pct:.1f}%)")
               
       summary.append("\nValidation Statistics:")
       summary.append(f"  • Total Adjustments: {validation_stats['total_adjustments']}")
       summary.append(f"  • Positive Adjustments: {validation_stats['positive_adjustments']}")
       summary.append(f"  • Negative Adjustments: {validation_stats['negative_adjustments']}")
       
       if likely_false_positives:
           summary.append("\nPotential False Positives:")
           for fp in likely_false_positives:
               summary.append(f"\n  Entity: '{fp['text']}' ({fp['type']})")
               for reason in fp["reasons"]:
                   summary.append(f"    • {reason}")
       
       return "\n".join(summary)

   def _generate_json_batch_summary(self, 
                                  analyses: List[DetectionAnalysis]) -> str:
       """
       Generate JSON summary of batch analysis results.
       
       Args:
           analyses: List of analyses to summarize
           
       Returns:
           JSON formatted summary
       """
       import json
       
       summary = {
           "total_entities": len(analyses),
           "by_type": {},
           "by_detector": {},
           "confidence_distribution": {
               "0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.6": 0,
               "0.6-0.8": 0, "0.8-1.0": 0
           },
           "validation_statistics": {
               "total_adjustments": 0,
               "positive_adjustments": 0,
               "negative_adjustments": 0
           },
           "potential_false_positives": []
       }
       
       # Collect statistics
       for analysis in analyses:
           entity = analysis.entity
           
           # Count by type
           summary["by_type"][entity.entity_type] = \
               summary["by_type"].get(entity.entity_type, 0) + 1
           
           # Count by detector
           summary["by_detector"][entity.detector_source] = \
               summary["by_detector"].get(entity.detector_source, 0) + 1
           
           # Track confidence ranges
           conf = entity.confidence
           for range_str in summary["confidence_distribution"]:
               low, high = map(float, range_str.split("-"))
               if low <= conf < high:
                   summary["confidence_distribution"][range_str] += 1
           
           # Track validation adjustments
           if analysis.validation_info:
               adj = analysis.validation_info.get("confidence_impact", {}).get("total_adjustment", 0)
               if adj != 0:
                   summary["validation_statistics"]["total_adjustments"] += 1
                   if adj > 0:
                       summary["validation_statistics"]["positive_adjustments"] += 1
                   else:
                       summary["validation_statistics"]["negative_adjustments"] += 1
           
           # Track potential false positives
           if analysis.context_info["indicators"].get("likely_false_positive"):
               summary["potential_false_positives"].append({
                   "text": entity.text,
                   "type": entity.entity_type,
                   "confidence": entity.confidence,
                   "reasons": analysis.context_info["indicators"]["reasons"]
               })
       
       return json.dumps(summary, indent=2)

###############################################################################
# END OF FILE
###############################################################################