###############################################################################
# SECTION 1: IMPORTS AND TYPE DEFINITIONS
###############################################################################
from pathlib import Path
import sys
import argparse
import logging  
from dataclasses import dataclass, replace
from typing import List, Dict, Set, Any, Optional
import yaml
from tabulate import tabulate
from datetime import datetime  

from .test_runner import TestRunner
from redactor.detectors.ensemble_coordinator import EnsembleCoordinator
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from evaluation.models import Entity, SensitivityLevel
from evaluation.metrics.entity_metrics import EntityMetrics
from evaluation.analysis.detection_analyzer import DetectionAnalyzer


# Type aliases for clarity
AnalysisResult = Dict[str, Any]
EntityList = List[Entity]
ValidationResult = Dict[str, Any]

###############################################################################
# SECTION 2: CONSTANTS AND CONFIGURATIONS
###############################################################################

PACKAGE_ROOT = Path(__file__).parent
PROJECT_ROOT = PACKAGE_ROOT.parent

# Valid entity types for validation
VALID_ENTITIES = [
    "EDUCATIONAL_INSTITUTION", "PERSON", "ADDRESS", "EMAIL_ADDRESS", 
    "PHONE_NUMBER", "INTERNET_REFERENCE", "LOCATION", "GPA",
    "DATE_TIME", "PROTECTED_CLASS", "PHI"
]

# Analysis pattern definitions with descriptions
ANALYSIS_PATTERNS = {
    "validation-rules": "Analyze validation rule effectiveness",
    "confidence-impact": "Study confidence adjustment patterns",
    "context-patterns": "Identify contextual patterns",
    "false-positives": "Detect potential false positives"
}

###############################################################################
# SECTION 3: EVALUATOR CLASS DEFINITION
###############################################################################

class StageOneEvaluator:
    """First stage evaluator for entity detection and analysis."""
    
    def __init__(self) -> None:
        """Initialize evaluation components and configurations."""
        # Set debug mode based on logger
        self.debug_mode = False
        
        # Validate environment
        self.test_suite_dir = self._validate_test_suite()
        
        # Initialize logging
        self.logger = RedactionLogger(
            name="evaluation",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True
        )
        
        # Set debug mode based on logger configuration
        if self.logger:
            if hasattr(self.logger, 'level'):
                self.debug_mode = self.logger.level == 10
            elif hasattr(self.logger, 'getEffectiveLevel'):
                self.debug_mode = self.logger.getEffectiveLevel() == 10
        
        # Load configurations
        self.config_loader = ConfigLoader()
        self.config_loader.load_all_configs()
        
        # Initialize core components
        self._init_components()

    def _validate_test_suite(self) -> Path:
        """Validate and return test suite directory."""
        test_dir = Path("data/test_suite")
        if not test_dir.exists():
            raise FileNotFoundError(
                f"Test suite directory not found: {test_dir}\n"
                "Make sure you're running from the project root"
            )
        return test_dir

    def _init_components(self) -> None:
        """Initialize core system components."""
        try:
            # Initialize detection components
            self.ensemble = EnsembleCoordinator(
                config_loader=self.config_loader,
                logger=self.logger
            )
            
            # Initialize metrics and analysis
            self.entity_metrics = EntityMetrics(config_loader=self.config_loader)
            self.detection_analyzer = DetectionAnalyzer(
                config_loader=self.config_loader,
                logger=self.logger
            )
            
            if self.logger.debug_mode:
                self.logger.debug("Core components initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {str(e)}")
            raise

###############################################################################
# SECTION 4: CORE ANALYSIS METHODS
###############################################################################

    def analyze_detection(self, 
                        test_id: str, 
                        target_entities: Optional[List[str]] = None) -> AnalysisResult:
        """
        Analyze entity detection with enhanced analysis capabilities.
        
        Args:
            test_id: Test case identifier
            target_entities: Optional list of entity types to analyze
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Load test content
            text = self._load_test_document(test_id)
            ground_truth = self._load_ground_truth(test_id)
            
            if self.logger.debug_mode:
                self.logger.debug(f"Loaded test document: {test_id}")
                self.logger.debug(f"Ground truth entities: {len(ground_truth)}")
            

            # First get detected entities
            target_entity_type = target_entities[0] if target_entities else None
            detected_entities = self._process_detected_entities(
                self.ensemble.detect_entities(text),
                target_entity_type
            )
            
            if self.debug_mode:
                self.logger.debug(f"Detected entities: {len(detected_entities)}")
                for entity in detected_entities:
                    self.logger.debug(
                        f"Entity: {entity.text}, Type: {entity.entity_type}, "
                        f"Confidence: {entity.confidence}, Source: {entity.detector_source}"
                    )

            
            # Validate and adjust entities based on analysis
            validated_entities = []
            for entity in detected_entities:
                analysis_result = self.detection_analyzer.analyze_detection(
                    entity=entity,
                    original_text=text,
                    truth_set=ground_truth
                )
                
                # Get confidence adjustment from validation_info
                confidence_adjustment = analysis_result.validation_info.get('confidence_adjustment', 0.0)
                
                # Create new entity with adjusted confidence using replace()
                if confidence_adjustment != 0:
                    adjusted_confidence = max(0.0, min(1.0, 
                        entity.confidence + confidence_adjustment))
                    entity = replace(
                        entity,
                        confidence=adjusted_confidence,
                        validation_rules=list(entity.validation_rules or []) + 
                            [f"confidence_adj_{confidence_adjustment}"]
                    )
                validated_entities.append(entity)
            
            # Analyze the validated entities
            analysis_results = self._analyze_entities(
                validated_entities,
                text,
                ground_truth
            )
            
            # Generate complete results
            return {
                "detection_analysis": {
                    "entities": validated_entities,
                    "analysis_results": analysis_results,
                    "statistics": self._analyze_detections(ground_truth, validated_entities),
                    "detector_performance": self._analyze_detector_performance(validated_entities)
                },
                "validation_analysis": self._analyze_validation_results(analysis_results),
                "context_analysis": self._analyze_context_patterns(analysis_results),
                "ground_truth": ground_truth,
                "original_text": text
            }
        
        except Exception as e:
            self.logger.error(f"Error in analyze_detection: {str(e)}")
            raise

    def _process_detected_entities(self, 
                                 entities: List[Entity],
                                 target_entity_type: Optional[str] = None) -> List[Entity]:
        """
        Process and filter detected entities.
        
        Args:
            entities: List of detected entities
            target_entity_type: Optional entity type to filter for
            
        Returns:
            List of processed entities
        """
        processed = []
        if self.debug_mode:
            self.logger.debug(f"Processing {len(entities)} detected entities")
            
        for entity in entities:
            if self.debug_mode:
                self.logger.debug(f"Processing entity: {entity.text}")
                self.logger.debug(f"Original attributes: {vars(entity)}")
                
            confidence = getattr(entity, 'confidence', 0.0)
            
            # Skip if entity doesn't match target type
            if target_entity_type and entity.entity_type != target_entity_type:
                continue
                
            # Create base entity
            base_entity = Entity(
                text=entity.text,
                entity_type=entity.entity_type,
                start_char=getattr(entity, 'start_char', 0),
                end_char=getattr(entity, 'end_char', 
                               getattr(entity, 'start_char', 0) + len(entity.text)),
                confidence=confidence,
                page=None,
                sensitivity=None,
                detector_source=getattr(entity, 'detector_source', 'unknown'),
                validation_rules=[],
                original_confidence=confidence
            )
            
            # Update with any additional attributes via replace
            updated_entity = replace(
                base_entity,
                validation_rules=getattr(entity, 'validation_rules', []),
                original_confidence=getattr(entity, 'original_confidence', confidence)
            )
            
            processed.append(updated_entity)
                    
        return processed
        
###############################################################################
# SECTION 5: DOCUMENT LOADING AND PREPROCESSING
###############################################################################

    def _load_test_document(self, test_id: str) -> str:
        """
        Load and extract text from test document.
        
        Args:
            test_id: Test case identifier
            
        Returns:
            Extracted document text
            
        Raises:
            FileNotFoundError: If document not found
            RuntimeError: If text extraction fails
        """
        pdf_path = self.test_suite_dir / "originals" / f"{test_id}.pdf"
        if not pdf_path.exists():
            raise FileNotFoundError(f"Test document not found: {pdf_path}")
        
        try:
            import fitz  # PyMuPDF
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            
            if not text.strip():
                raise RuntimeError(f"No text extracted from {pdf_path}")
                
            return text
            
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) required for PDF processing")
        except Exception as e:
            self.logger.error(f"Error reading PDF {pdf_path}: {str(e)}")
            raise RuntimeError(f"Failed to process PDF: {str(e)}")

    def _load_ground_truth(self, test_id: str) -> EntityList:
        """
        Load ground truth annotations from YAML.
        
        Args:
            test_id: Test case identifier
            
        Returns:
            List of ground truth entities
            
        Raises:
            FileNotFoundError: If annotation file not found
            ValueError: If invalid annotation format
        """
        annotation_path = self.test_suite_dir / "annotations" / f"{test_id}.yaml"
        if not annotation_path.exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
            
        try:
            with open(annotation_path) as f:
                annotations = yaml.safe_load(f)
                
            if not annotations or "entities" not in annotations:
                raise ValueError(f"Invalid annotation format in {annotation_path}")
                
            ground_truth = []
            for ann in annotations.get("entities", []):
                entity = Entity(
                    text=ann["text"],
                    entity_type=ann["type"],
                    start_char=ann.get("location", {}).get("start_char", 0),
                    end_char=ann.get("location", {}).get("end_char", len(ann["text"])),
                    confidence=1.0,  # Ground truth always has full confidence
                    sensitivity=SensitivityLevel[ann.get("sensitivity", "LOW").upper()],
                    detector_source='ground_truth'
                )
                ground_truth.append(entity)
                
            return ground_truth
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {annotation_path}: {str(e)}")
        except Exception as e:
            self.logger.error(f"Error loading annotations from {annotation_path}: {str(e)}")
            raise

###############################################################################
# SECTION 6: ANALYSIS HELPER METHODS
###############################################################################

    def _analyze_entities(self, 
                         entities: EntityList,
                         text: str,
                         ground_truth: EntityList) -> List[AnalysisResult]:
        """
        Analyze each detected entity in detail.
        
        Args:
            entities: List of detected entities
            text: Original document text
            ground_truth: Ground truth entities
            
        Returns:
            List of detailed analysis results
        """
        analysis_results = []
        
        for entity in entities:
            try:
                analysis = self.detection_analyzer.analyze_detection(
                    entity=entity,
                    original_text=text,
                    truth_set=ground_truth
                )
                analysis_results.append(analysis)
                
            except Exception as e:
                self.logger.warning(
                    f"Error analyzing entity '{entity.text}': {str(e)}"
                )
                continue
                
        return analysis_results

    def _analyze_validation_results(self, 
                                  analysis_results: List[AnalysisResult]) -> ValidationResult:
        """
        Analyze validation performance and rule effectiveness.
        
        Args:
            analysis_results: List of detection analyses
            
        Returns:
            Validation analysis summary
        """
        validation_stats = {
            "rules": {},
            "confidence_impacts": {
                "increases": 0,
                "decreases": 0,
                "no_change": 0
            },
            "by_entity_type": {}
        }
        
        for analysis in analysis_results:
            self._process_validation_analysis(analysis, validation_stats)
            
        return validation_stats
        

    def _analyze_detections(self,
                       truth: List[Entity],
                       detected: List[Entity]) -> Dict:
        """
        Analyze detection performance by entity type and detection source.
        
        Args:
            truth: Ground truth entities
            detected: Detected entities
            
        Returns:
            Dict containing detection statistics
        """
        if self.logger.debug_mode:
            self.logger.debug("Starting detection analysis")
            self.logger.debug(f"Ground truth entities: {len(truth)}")
            self.logger.debug(f"Detected entities: {len(detected)}")
            for d in detected:
                # Use the final adjusted confidence, not original
                adjusted_confidence = getattr(d, "confidence", 0.0)
                original_confidence = getattr(d, "original_confidence", adjusted_confidence)
                self.logger.debug(
                    f"Detected entity confidence: {adjusted_confidence} "
                    f"original: {original_confidence} "
                    f"type: {type(adjusted_confidence)}"
                )

        analysis = {
            "by_type": {},
            "by_detector": {},
            "confidence_distribution": {},
            "validation_effectiveness": {},
            "configuration_coverage": self._analyze_config_coverage()
        }
        
        # Analyze by entity type
        truth_by_type = {}
        for t in truth:
            if t.entity_type not in truth_by_type:
                truth_by_type[t.entity_type] = {"total": 0, "detected": 0}
            truth_by_type[t.entity_type]["total"] += 1
        
        for d in detected:
            if d.entity_type in truth_by_type:
                truth_by_type[d.entity_type]["detected"] += 1
            
            # Handle None values safely and use adjusted confidence
            original_conf = d.original_confidence if d.original_confidence is not None else d.confidence
            current_conf = d.confidence if d.confidence is not None else 0.0
            
            conf_delta = current_conf - original_conf
            
            # Update detector stats 
            detector = d.detector_source or "unknown"
            if detector not in analysis["by_detector"]:
                analysis["by_detector"][detector] = {
                    "total": 0,
                    "confidence_sum": 0.0,
                    "validation_improvements": 0,
                    "validation_reductions": 0
                }
            
            det_stats = analysis["by_detector"][detector]
            det_stats["total"] += 1
            det_stats["confidence_sum"] += current_conf  # Use adjusted confidence
            
            if conf_delta > 0:
                det_stats["validation_improvements"] += 1
            elif conf_delta < 0:
                det_stats["validation_reductions"] += 1
                
        # Calculate detection rates
        for entity_type, counts in truth_by_type.items():
            detection_rate = counts["detected"] / counts["total"] if counts["total"] > 0 else 0
            analysis["by_type"][entity_type] = {
                "total": counts["total"],
                "detected": counts["detected"],
                "detection_rate": detection_rate
            }
        
        return analysis
    
    def _analyze_config_coverage(self) -> Dict:
        """
        Analyze how well configuration covers detected entities.
        
        Returns:
            Dict containing configuration coverage analysis
        """
        routing_config = self.config_loader.get_config("entity_routing")
        validation_config = self.config_loader.get_config("validation_params")
        
        coverage = {
            "routing": {
                "total_entity_types": 0,
                "configured_types": 0,
                "routing_rules": {},
                "threshold_coverage": {}
            },
            "validation": {
                "total_rules": 0,
                "rules_by_type": {},
                "missing_validations": []
            }
        }
        
        # Analyze routing coverage
        for section in ["presidio_primary", "spacy_primary", "ensemble_required"]:
            if section in routing_config["routing"]:
                entities = routing_config["routing"][section]["entities"]
                coverage["routing"]["total_entity_types"] += len(entities)
                coverage["routing"]["routing_rules"][section] = len(entities)
        
        return coverage

    def _process_validation_analysis(self,
                                   analysis: AnalysisResult,
                                   stats: Dict) -> None:
        """
        Process validation analysis for a single entity.
        
        Args:
            analysis: Single entity analysis result
            stats: Statistics dictionary to update
        """
        entity = analysis.entity
        entity_type = entity.entity_type
        
        # Initialize entity type stats if needed
        if entity_type not in stats["by_entity_type"]:
            stats["by_entity_type"][entity_type] = {
                "total": 0,
                "rules_triggered": {},
                "confidence_impacts": {
                    "increases": 0,
                    "decreases": 0,
                    "no_change": 0
                }
            }
        
        type_stats = stats["by_entity_type"][entity_type]
        type_stats["total"] += 1
        
        # Process validation rules
        self._update_rule_stats(
            analysis.validation_info.get("validation_rules", []),
            stats["rules"],
            type_stats["rules_triggered"],
            entity_type
        )
        
        # Process confidence impacts
        self._update_confidence_stats(
            analysis.validation_info.get("confidence_impact", {}),
            stats["confidence_impacts"],
            type_stats["confidence_impacts"]
        )

    def _update_rule_stats(self,
                          rules: List[str],
                          global_stats: Dict,
                          type_stats: Dict,
                          entity_type: str) -> None:
        """
        Update rule statistics.
        
        Args:
            rules: List of applied rules
            global_stats: Global statistics to update
            type_stats: Entity type statistics to update
            entity_type: Current entity type
        """
        for rule in rules:
            if rule not in global_stats:
                global_stats[rule] = {
                    "total_triggers": 0,
                    "entity_types": set(),
                    "confidence_impacts": []
                }
            if rule not in type_stats:
                type_stats[rule] = 0
                
            global_stats[rule]["total_triggers"] += 1
            global_stats[rule]["entity_types"].add(entity_type)
            type_stats[rule] += 1

    def _update_confidence_stats(self,
                               impact: Dict,
                               global_stats: Dict,
                               type_stats: Dict) -> None:
        """
        Update confidence impact statistics.
        
        Args:
            impact: Confidence impact information
            global_stats: Global statistics to update
            type_stats: Entity type statistics to update
        """
        adjustment = impact.get("total_adjustment", 0)
        
        if adjustment > 0:
            global_stats["increases"] += 1
            type_stats["increases"] += 1
        elif adjustment < 0:
            global_stats["decreases"] += 1
            type_stats["decreases"] += 1
        else:
            global_stats["no_change"] += 1
            type_stats["no_change"] += 1

    def _update_pattern_stats(self,
                            pattern_type: str,
                            match_info: Dict,
                            global_stats: Dict,
                            type_stats: Dict,
                            entity_type: str,
                            entity_text: str) -> None:
        """Update pattern statistics at global and type level."""
        # Initialize pattern stats if needed
        if pattern_type not in global_stats:
            global_stats[pattern_type] = {
                "total": 0,
                "entity_types": set(),
                "examples": []
            }
        if pattern_type not in type_stats:
            type_stats[pattern_type] = {
                "count": 0,
                "examples": []
            }
        
        # Update global stats
        global_stats[pattern_type]["total"] += 1
        global_stats[pattern_type]["entity_types"].add(entity_type)
        if len(global_stats[pattern_type]["examples"]) < 5:
            global_stats[pattern_type]["examples"].append(entity_text)
            
        # Update type stats
        type_stats[pattern_type]["count"] += 1
        if len(type_stats[pattern_type]["examples"]) < 3:
            type_stats[pattern_type]["examples"].append(entity_text)

    def _update_indicator_stats(self,
                              indicators: Dict,
                              global_stats: Dict,
                              type_stats: Dict,
                              entity_type: str) -> None:
        """Update indicator statistics at global and type level."""
        for indicator, details in indicators.items():
            # Global stats
            if indicator not in global_stats:
                global_stats[indicator] = {
                    "total": 0,
                    "entity_types": set()
                }
            global_stats[indicator]["total"] += 1
            global_stats[indicator]["entity_types"].add(entity_type)
            
            # Type stats
            if indicator not in type_stats:
                type_stats[indicator] = 0
            type_stats[indicator] += 1

    def _update_detector_stats(self,
                             stats: Dict,
                             entity: Entity) -> None:
        """Update detector performance statistics."""
        stats["total_detections"] += 1
        stats["confidence_sum"] += entity.confidence
        
        # Track by entity type
        if entity.entity_type not in stats["by_entity_type"]:
            stats["by_entity_type"][entity.entity_type] = {
                "count": 0,
                "confidence_sum": 0,
                "examples": []
            }
        
        type_stats = stats["by_entity_type"][entity.entity_type]
        type_stats["count"] += 1
        type_stats["confidence_sum"] += entity.confidence
        
        if len(type_stats["examples"]) < 5:
            type_stats["examples"].append(entity.text)
        
        # Track confidence ranges
        conf = entity.confidence
        for range_str in stats["confidence_ranges"]:
            low, high = map(float, range_str.split("-"))
            if low <= conf < high:
                stats["confidence_ranges"][range_str] += 1
        
        # Track validation impact
        original_conf = getattr(entity, 'original_confidence', entity.confidence)
        if entity.confidence > original_conf:
            stats["validation_impacts"]["positive"] += 1
        elif entity.confidence < original_conf:
            stats["validation_impacts"]["negative"] += 1
        else:
            stats["validation_impacts"]["neutral"] += 1

    def _finalize_detector_stats(self, detector_stats: Dict) -> None:
        """Calculate final statistics for detector performance."""
        for detector, stats in detector_stats.items():
            if stats["total_detections"] > 0:
                stats["average_confidence"] = stats["confidence_sum"] / stats["total_detections"]
                
                for entity_type, type_stats in stats["by_entity_type"].items():
                    if type_stats["count"] > 0:
                        type_stats["average_confidence"] = type_stats["confidence_sum"] / type_stats["count"]
                    del type_stats["confidence_sum"]
                    
            del stats["confidence_sum"]

###############################################################################
# SECTION 7: CONTEXT AND PATTERN ANALYSIS
###############################################################################

    def _analyze_context_patterns(self, 
                                analysis_results: List[AnalysisResult]) -> Dict:
        """
        Analyze contextual patterns in detections.
        
        Args:
            analysis_results: List of detection analyses
            
        Returns:
            Context pattern analysis dictionary
        """
        context_stats = {
            "patterns": {},
            "by_entity_type": {},
            "common_indicators": {}
        }
        
        for analysis in analysis_results:
            entity = analysis.entity
            entity_type = entity.entity_type
            context_info = analysis.context_info
            
            # Initialize entity type stats
            if entity_type not in context_stats["by_entity_type"]:
                context_stats["by_entity_type"][entity_type] = {
                    "total": 0,
                    "patterns": {},
                    "indicators": {}
                }
                
            self._update_context_stats(
                context_info,
                context_stats,
                entity_type,
                entity.text
            )
            
        return context_stats

    def _update_context_stats(self,
                            context_info: Dict,
                            stats: Dict,
                            entity_type: str,
                            entity_text: str) -> None:
        """
        Update context pattern statistics.
        
        Args:
            context_info: Context analysis information
            stats: Statistics dictionary to update
            entity_type: Current entity type
            entity_text: Entity text for examples
        """
        type_stats = stats["by_entity_type"][entity_type]
        type_stats["total"] += 1
        
        # Process pattern matches
        matches = context_info.get("patterns", {}).get("matches", {})
        for pattern_type, match_info in matches.items():
            self._update_pattern_stats(
                pattern_type,
                match_info,
                stats["patterns"],
                type_stats["patterns"],
                entity_type,
                entity_text
            )
            
        # Process indicators
        indicators = context_info.get("indicators", {})
        self._update_indicator_stats(
            indicators,
            stats["common_indicators"],
            type_stats["indicators"],
            entity_type
        )

    def _analyze_detector_performance(self, detected: EntityList) -> Dict:
        """
        Analyze detector performance metrics.
        
        Args:
            detected: List of detected entities
            
        Returns:
            Detector performance analysis
        """
        detector_stats = {}
        
        for entity in detected:
            detector = getattr(entity, 'detector_source', 'unknown')
            if detector not in detector_stats:
                detector_stats[detector] = self._initialize_detector_stats()
                
            self._update_detector_stats(
                detector_stats[detector],
                entity
            )
            
        self._finalize_detector_stats(detector_stats)
        return detector_stats

    def _initialize_detector_stats(self) -> Dict:
        """Initialize detector statistics structure."""
        return {
            "total_detections": 0,
            "confidence_sum": 0,
            "by_entity_type": {},
            "confidence_ranges": {
                "0.0-0.2": 0,
                "0.2-0.4": 0,
                "0.4-0.6": 0,
                "0.6-0.8": 0,
                "0.8-1.0": 0
            },
            "validation_impacts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            }
        }

###############################################################################
# SECTION 8: DISPLAY METHODS
###############################################################################

    def display_analysis(self, 
                        analysis: Dict,
                        target_entities: Optional[List[str]] = None) -> None:
        """
        Display comprehensive analysis results.
        
        Args:
            analysis: Analysis results dictionary
            target_entities: Optional list of entity types to focus on
        """
        # Filter entities if target specified
        entities = analysis["detection_analysis"]["entities"]
        if target_entities:
            entities = [e for e in entities if e.entity_type in target_entities]
            
        self._display_overview(entities)
        self._display_detector_results(analysis["detection_analysis"]["detector_performance"])
        self._display_validation_summary(analysis["validation_analysis"])
        self._display_context_summary(analysis["context_analysis"])

    def display_focused_analysis(self,
                               analysis: Dict,
                               pattern: str,
                               entity_type: Optional[str] = None) -> None:
        """
        Display focused analysis for a specific pattern.
        
        Args:
            analysis: Analysis results
            pattern: Analysis pattern to focus on
            entity_type: Optional entity type to filter by
        """
        analysis_displays = {
            "validation-rules": self._display_validation_focused,
            "confidence-impact": self._display_confidence_focused,
            "context-patterns": self._display_context_focused,
            "false-positives": self._display_false_positive_focused
        }
        
        display_func = analysis_displays.get(pattern)
        if display_func:
            display_func(analysis, entity_type)
        else:
            print(f"Unknown analysis pattern: {pattern}")

    def _display_overview(self, entities: EntityList) -> None:
        """Display overview of detected entities."""
        # Entity type breakdown
        by_type = {}
        for entity in entities:
            if entity.entity_type not in by_type:
                by_type[entity.entity_type] = 0
            by_type[entity.entity_type] += 1
        
        print("\nEntity Detection Overview")
        print("=======================")
        print(f"Total Entities: {len(entities)}")
        
        print("\nBy Entity Type:")
        for entity_type, count in by_type.items():
            print(f"  {entity_type}: {count}")

    def _display_detector_results(self, detector_stats: Dict) -> None:
        """Display detector performance results."""
        print("\nDetector Performance")
        print("===================")
        
        for detector, stats in detector_stats.items():
            print(f"\n{detector.upper()}:")
            print(f"  Total Detections: {stats['total_detections']}")
            print(f"  Average Confidence: {stats.get('average_confidence', 0):.2f}")
            
            if stats.get("confidence_ranges"):
                print("\n  Confidence Distribution:")
                for range_str, count in stats["confidence_ranges"].items():
                    if count > 0:
                        pct = (count / stats['total_detections']) * 100
                        print(f"    {range_str}: {count} ({pct:.1f}%)")
            
            if stats.get("by_entity_type"):
                print("\n  By Entity Type:")
                for entity_type, type_stats in stats["by_entity_type"].items():
                    print(f"    {entity_type}:")
                    print(f"      Count: {type_stats['count']}")
                    print(f"      Average Confidence: {type_stats.get('average_confidence', 0):.2f}")

    def _display_validation_summary(self, validation_stats: Dict) -> None:
        """Display validation analysis summary."""
        print("\nValidation Analysis")
        print("==================")
        
        # Rule effectiveness
        if validation_stats.get("rules"):
            print("\nRule Effectiveness:")
            for rule, stats in validation_stats["rules"].items():
                print(f"\n  {rule}:")
                print(f"    Triggers: {stats['total_triggers']}")
                print(f"    Entity Types: {', '.join(stats['entity_types'])}")
        
        # Confidence impacts
        impacts = validation_stats.get("confidence_impacts", {})
        if impacts:
            print("\nConfidence Impacts:")
            total = sum(impacts.values())
            for impact_type, count in impacts.items():
                pct = (count / total) * 100 if total > 0 else 0
                print(f"  {impact_type.title()}: {count} ({pct:.1f}%)")

    def _display_context_summary(self, context_stats: Dict) -> None:
        """Display context analysis summary."""
        print("\nContext Analysis")
        print("===============")
        
        # Pattern matches
        if context_stats.get("patterns"):
            print("\nPattern Matches:")
            for pattern, stats in context_stats["patterns"].items():
                print(f"\n  {pattern}:")
                print(f"    Total Occurrences: {stats['total']}")
                print(f"    Found in: {', '.join(stats['entity_types'])}")
                if stats.get("examples"):
                    print("    Examples:")
                    for ex in stats["examples"][:3]:
                        print(f"      - {ex}")
        
        # Common indicators
        if context_stats.get("common_indicators"):
            print("\nCommon Indicators:")
            for indicator, stats in context_stats["common_indicators"].items():
                print(f"\n  {indicator}:")
                print(f"    Occurrences: {stats['total']}")
                print(f"    Entity Types: {', '.join(stats['entity_types'])}")

    def _display_validation_focused(self,
                                  analysis: Dict,
                                  entity_type: Optional[str] = None) -> None:
        """Display focused validation rule analysis."""
        validation_stats = analysis["validation_analysis"]
        
        print("\nValidation Rule Analysis")
        print("======================")
        
        if entity_type:
            type_stats = validation_stats["by_entity_type"].get(entity_type, {})
            if type_stats:
                print(f"\nAnalysis for {entity_type}")
                print(f"Total Entities: {type_stats['total']}")
                
                # Rule effectiveness
                if type_stats.get("rules_triggered"):
                    print("\nRule Effectiveness:")
                    for rule, count in sorted(
                        type_stats["rules_triggered"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    ):
                        pct = (count / type_stats['total']) * 100
                        print(f"\n  {rule}:")
                        print(f"    Triggers: {count} ({pct:.1f}%)")
        else:
            # Global analysis
            self._display_validation_summary(validation_stats)

    def _display_confidence_focused(self,
                                  analysis: Dict,
                                  entity_type: Optional[str] = None) -> None:
        """Display focused confidence analysis."""
        detector_stats = analysis["detection_analysis"]["detector_performance"]
        
        print("\nConfidence Analysis")
        print("==================")
        
        if entity_type:
            print(f"\nAnalysis for {entity_type}")
            for detector, stats in detector_stats.items():
                type_stats = stats["by_entity_type"].get(entity_type)
                if type_stats:
                    print(f"\n{detector.upper()}:")
                    print(f"  Count: {type_stats['count']}")
                    print(f"  Average Confidence: {type_stats.get('average_confidence', 0):.2f}")
                    if type_stats.get("examples"):
                        print("  Examples:")
                        for ex in type_stats["examples"]:
                            print(f"    - {ex}")
        else:
            self._display_detector_results(detector_stats)

    def _display_context_focused(self,
                               analysis: Dict,
                               entity_type: Optional[str] = None) -> None:
        """Display focused context pattern analysis."""
        context_stats = analysis["context_analysis"]
        
        print("\nContext Pattern Analysis")
        print("=======================")
        
        if entity_type:
            type_stats = context_stats["by_entity_type"].get(entity_type)
            if type_stats:
                print(f"\nAnalysis for {entity_type}")
                print(f"Total Entities: {type_stats['total']}")
                
                if type_stats.get("patterns"):
                    print("\nPatterns Found:")
                    for pattern, stats in type_stats["patterns"].items():
                        print(f"\n  {pattern}:")
                        print(f"    Count: {stats['count']}")
                        if stats.get("examples"):
                            print("    Examples:")
                            for ex in stats["examples"]:
                                print(f"      - {ex}")
        else:
            self._display_context_summary(context_stats)

    def _display_false_positive_focused(self,
                                      analysis: Dict,
                                      entity_type: Optional[str] = None) -> None:
        """Display focused false positive analysis."""
        detected = analysis["detection_analysis"]["entities"]
        if entity_type:
            detected = [e for e in detected if e.entity_type == entity_type]
            
        print("\nFalse Positive Analysis")
        print("=====================")
        
        # Identify potential false positives
        false_positives = []
        for entity in detected:
            indicators = []
            
            # Check confidence
            if entity.confidence <= 0.6:
                indicators.append("Low confidence score")
                
            # Check validation impacts
            analysis_result = next(
                (r for r in analysis["detection_analysis"]["analysis_results"]
                 if r.entity.text == entity.text),
                None
            )
            
            if analysis_result:
                # Check context indicators
                context_info = analysis_result.context_info.get("indicators", {})
                if context_info.get("likely_false_positive"):
                    indicators.extend(context_info.get("reasons", []))
                
                # Check confidence adjustments
                impact = analysis_result.validation_info.get("confidence_impact", {})
                if impact.get("total_adjustment", 0) < -0.2:
                    indicators.append("Significant negative validation adjustment")
            
            if indicators:
                false_positives.append({
                    "entity": entity,
                    "indicators": indicators,
                    "context": analysis_result.context_info if analysis_result else {}
                })
        
        if false_positives:
            print(f"\nFound {len(false_positives)} potential false positives:")
            for fp in false_positives:
                print(f"\nEntity: {fp['entity'].text}")
                print(f"Type: {fp['entity'].entity_type}")
                print(f"Confidence: {fp['entity'].confidence:.2f}")
                print("Indicators:")
                for indicator in fp["indicators"]:
                    print(f"  - {indicator}")
        else:
            print("\nNo clear false positives detected")

###############################################################################
# SECTION 9: COMMAND LINE INTERFACE
###############################################################################

def main():
    """Main entry point for evaluation tool."""
    try:
        args = parse_arguments()
        configure_logging(args.quiet)
        
        evaluator = StageOneEvaluator()
        handle_command(evaluator, args)
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Redaction system evaluation tool",
        usage="python -m evaluation.evaluate [args]"
    )
    
    parser.add_argument(
        "command",
        choices=["detect", "analyze", "batch", "compare"],
        help="Evaluation command to run"
    )
    
    parser.add_argument(
        "--test",
        nargs='+',
        required=True,
        help="Test ID(s) to evaluate"
    )
    
    parser.add_argument(
        "--entity",
        nargs='+',
        choices=VALID_ENTITIES,
        help="Target specific entity types"
    )
    
    parser.add_argument(
        "--pattern",
        choices=list(ANALYSIS_PATTERNS.keys()),
        help="Specific analysis pattern to run"
    )
    
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep analysis including all patterns"
    )
    
    parser.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format"
    )
    
    parser.add_argument(
        "--output",
        help="Output file path"
    )
    
    parser.add_argument(
        "--compare-with",
        help="Previous test run ID to compare against"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    return parser.parse_args()

def handle_command(evaluator: StageOneEvaluator, args: argparse.Namespace) -> None:
    """
    Handle evaluation command execution.
    
    Args:
        evaluator: Initialized evaluator instance
        args: Parsed command line arguments
    """
    if args.command == "analyze":
        results = evaluator.analyze_detection(args.test[0], args.entity)
        
        if args.pattern:
            evaluator.display_focused_analysis(
                results,
                args.pattern,
                args.entity[0] if args.entity else None
            )
        elif args.deep:
            for pattern in ANALYSIS_PATTERNS:
                print(f"\n{ANALYSIS_PATTERNS[pattern]}:")
                print("=" * len(ANALYSIS_PATTERNS[pattern]))
                evaluator.display_focused_analysis(
                    results,
                    pattern,
                    args.entity[0] if args.entity else None
                )
        else:
            evaluator.display_analysis(results, args.entity)
            
        if args.output:
            save_results(results, args.output, args.format)
            
    elif args.command == "detect":
        results = evaluator.analyze_detection(args.test[0], args.entity)
        evaluator.display_analysis(results, args.entity)

def configure_logging(quiet: bool) -> None:
    """Configure logging based on verbosity level."""
    log_level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=log_level)
    
    # Reduce noise from other loggers
    logging.getLogger('config_loader').setLevel(logging.WARNING)
    logging.getLogger('presidio-analyzer').setLevel(logging.WARNING)
    logging.getLogger('evaluation').setLevel(log_level)

def save_results(results: Dict, 
                output_path: str, 
                format: str = "json") -> None:
    """
    Save analysis results to file.
    
    Args:
        results: Analysis results to save
        output_path: Output file path
        format: Output format
    """
    try:
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
        elif format == "csv":
            # TODO: Implement CSV output
            pass
            
    except Exception as e:
        print(f"Error saving results: {str(e)}", file=sys.stderr)

if __name__ == "__main__":
    main()

