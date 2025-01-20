# redactor/detectors/pattern_matcher.py

from typing import List, Dict, Optional, Any
import spacy
from presidio_analyzer import AnalyzerEngine, PatternRecognizer, Pattern
import time
from app.utils import config_loader
import os
import yaml

class PatternMatcher:
    """
    Coordinates entity detection between spaCy and Presidio.
    Handles loading of patterns from configuration and provides unified detection interface.
    """
    def __init__(self, 
                 detection_config: Dict[str, Any],
                 confidence_threshold: float = 0.7,
                 logger: Optional[Any] = None):
        """
        Initialize the pattern matcher with configuration.
        
        Args:
            detection_config: Dict containing pattern definitions
            confidence_threshold: Minimum confidence score for detections
            logger: Optional logger instance for detailed logging
        """
        self.logger = logger
        self.confidence_threshold = confidence_threshold
        
        # Initialize NLP components
        self._init_spacy()
        self._init_presidio(detection_config)
        
        if self.logger:
            self.logger.info("PatternMatcher initialized successfully")

    def _init_spacy(self):
        """Initialize spaCy with required model."""
        start_time = time.time()
        try:
            self.nlp = spacy.load("en_core_web_lg")
            if self.logger:
                self.logger.debug(
                    f"spaCy model loaded in {time.time() - start_time:.2f}s"
                )
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to load spaCy model: {str(e)}")
            raise
    
    def _init_presidio(self, detection_config: Dict[str, Any]):
        """Initialize Presidio analyzer with patterns from config."""
        try:
            if not isinstance(detection_config, dict):
                raise ValueError("Invalid detection_config: must be a dictionary of categories")
    
            self.analyzer = AnalyzerEngine()
    
            # Log the start of pattern registration
            self.logger.info("Starting pattern registration into Presidio...")
    
            patterns_loaded = 0
            for category, patterns in detection_config.items():
                if not isinstance(patterns, list):
                    self.logger.warning(f"Skipping category '{category}' - not a list")
                    continue
    
                for pattern in patterns:
                    # Validate pattern structure
                    if not all(key in pattern for key in ["name", "entity_type", "regex"]):
                        self.logger.error(f"Invalid pattern in category '{category}': {pattern}")
                        continue
    
                    try:
                        # Log pattern registration details
                        self.logger.debug(
                            f"Registering pattern from category '{category}': {pattern['name']} "
                            f"for entity type: {pattern['entity_type']} "
                            f"regex: {pattern['regex'][:50]}..."
                        )
    
                        # Register the pattern
                        recognizer = PatternRecognizer(
                            supported_entity=pattern["entity_type"],
                            patterns=[Pattern(
                                name=pattern["name"],
                                regex=pattern["regex"],
                                score=pattern.get("score", 0.7)
                            )]
                        )
                        self.analyzer.registry.add_recognizer(recognizer)
                        patterns_loaded += 1
                        self.logger.debug(f"Successfully registered pattern: {pattern['name']}")
    
                    except Exception as e:
                        self.logger.error(f"Error registering pattern '{pattern['name']}' in category '{category}': {str(e)}")
                        continue
    
            # Log final count of registered patterns
            if patterns_loaded == 0:
                self.logger.error("No patterns were registered into Presidio. Check configuration and logic.")
            else:
                self.logger.info(f"Loaded {patterns_loaded} patterns across all categories into Presidio")
    
        except Exception as e:
            self.logger.error(f"Failed to initialize Presidio: {str(e)}")
            raise



    def detect_entities(self, text: str) -> List[Dict]:
        """
        Detect entities using both spaCy and Presidio.
        
        Args:
            text: Text to analyze
            
        Returns:
            List of detected entities with their metadata
        """
        if not text.strip():
            return []

        entities = []
        
        try:
            if self.logger:
                self.logger.debug(f"Starting entity detection on text: {text[:100]}...")
            
            # SpaCy detection
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in {"PERSON", "ORG", "GPE", "DATE", "TIME", 
                                "MONEY", "PERCENT", "PHONE", "EMAIL"}:
                    if self.logger:
                        self.logger.debug(
                            f"SpaCy found: {ent.label_} - '{ent.text}'"
                        )
                    entities.append({
                        "text": ent.text,
                        "entity_type": ent.label_,
                        "confidence": 0.8,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "source": "spacy"
                    })

            # Presidio detection
            if self.logger:
                self.logger.debug("Starting Presidio detection")
            presidio_results = self.analyzer.analyze(
                text=text,
                language="en"
            )
            
            for result in presidio_results:
                if result.score >= self.confidence_threshold:
                    detected_text = text[result.start:result.end]
                    if self.logger:
                        self.logger.debug(
                            f"Presidio found: {result.entity_type} - '{detected_text}' "
                            f"(score: {result.score:.2f})"
                        )
                    entities.append({
                        "text": detected_text,
                        "entity_type": result.entity_type,
                        "confidence": result.score,
                        "start": result.start,
                        "end": result.end,
                        "source": "presidio"
                    })

            if self.logger:
                self.logger.debug(
                    f"Detection complete. Found {len(entities)} entities "
                    f"({len([e for e in entities if e['source'] == 'spacy'])} from spaCy, "
                    f"{len([e for e in entities if e['source'] == 'presidio'])} from Presidio)"
                )

        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during entity detection: {str(e)}")
            raise

        return self._merge_overlapping_entities(entities)

    def _merge_overlapping_entities(self, entities: List[Dict]) -> List[Dict]:
        """
        Merge overlapping entity detections, preferring higher confidence scores.
        
        Args:
            entities: List of detected entities
            
        Returns:
            List of merged entities with overlaps resolved
        """
        if not entities:
            return []

        # Sort by start position and confidence
        sorted_entities = sorted(
            entities,
            key=lambda x: (x["start"], -x["confidence"])
        )
        
        merged = []
        current = sorted_entities[0]
        
        for next_entity in sorted_entities[1:]:
            if current["end"] >= next_entity["start"]:
                # Overlapping entities - keep the one with higher confidence
                if next_entity["confidence"] > current["confidence"]:
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
                
        merged.append(current)
        
        return merged