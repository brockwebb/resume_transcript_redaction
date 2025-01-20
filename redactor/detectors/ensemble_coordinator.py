# redactor/detectors/ensemble_coordinator.py

from typing import List, Dict, Optional, Any, Set, Tuple
from .base_detector import BaseDetector, Entity
from .presidio_detector import PresidioDetector
from .spacy_detector import SpacyDetector

class EnsembleCoordinator:
    """Coordinates multiple detectors and combines their results."""
    
    def __init__(self, 
                 config_loader: Any,
                 logger: Optional[Any] = None):
        """Initialize coordinator with detectors."""
        self.logger = logger
        self.config_loader = config_loader
        
        # Initialize routing rules and thresholds
        self.presidio_entities: Set[str] = set()
        self.spacy_entities: Set[str] = set()
        self.ensemble_entities: Set[str] = set()
        self.ensemble_weights: Dict[str, float] = {}
        self.min_combined_confidence: float = 0.8
        self.min_individual_confidence: float = 0.6
        
        # Load configuration
        if not self._load_config():
            raise ValueError("Failed to load ensemble configuration")
        
        # Initialize detectors
        self.presidio_detector = PresidioDetector(config_loader, logger)
        self.spacy_detector = SpacyDetector(config_loader, logger)
        
    def _load_config(self) -> bool:
        """Load and validate ensemble configuration."""
        try:
            routing_config = self.config_loader.get_config("entity_routing")
            if not routing_config:
                self.logger.error("Missing entity routing configuration")
                return False
                
            routing = routing_config.get("routing", {})
            
            # Load entity type assignments
            self.presidio_entities = set(
                routing.get("presidio_primary", {}).get("entities", [])
            )
            self.spacy_entities = set(
                routing.get("spacy_primary", {}).get("entities", [])
            )
            self.ensemble_entities = set(
                routing.get("ensemble_required", {}).get("entities", [])
            )
            
            # Load ensemble configuration
            ensemble_config = routing.get("ensemble_required", {}).get("confidence_thresholds", {})
            self.ensemble_weights = {
                "presidio": ensemble_config.get("presidio_weight", 0.4),
                "spacy": ensemble_config.get("spacy_weight", 0.6)
            }
            
            self.min_combined_confidence = ensemble_config.get("minimum_combined", 0.8)
            self.min_individual_confidence = ensemble_config.get("minimum_individual", 0.6)
            
            # Validate configuration
            if not self._validate_config():
                return False
                
            self.logger.debug(
                f"Loaded ensemble config:\n"
                f"Presidio entities: {self.presidio_entities}\n"
                f"spaCy entities: {self.spacy_entities}\n"
                f"Ensemble entities: {self.ensemble_entities}\n"
                f"Weights: {self.ensemble_weights}"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading ensemble config: {str(e)}")
            return False
            
    def _validate_config(self) -> bool:
        """Validate ensemble configuration."""
        # Check for overlapping entity assignments
        presidio_overlap = self.presidio_entities & self.spacy_entities
        if presidio_overlap:
            self.logger.error(f"Entities assigned to both Presidio and spaCy: {presidio_overlap}")
            return False
            
        # Validate weights
        weight_sum = sum(self.ensemble_weights.values())
        if not (0.99 <= weight_sum <= 1.01):  # Allow for floating point imprecision
            self.logger.error(f"Ensemble weights must sum to 1.0, got {weight_sum}")
            return False
            
        return True

    def detect_entities(self, text: str) -> List[Entity]:
        """Detect entities using appropriate detectors based on entity type."""
        if not text:
            return []
            
        try:
            self.logger.debug(f"Starting ensemble detection on text: {text[:100]}...")
            
            # Get detections from both systems
            presidio_entities = self.presidio_detector.detect_entities(text)
            spacy_entities = self.spacy_detector.detect_entities(text)
            
            self.logger.debug(
                f"Initial detections - Presidio: {len(presidio_entities)}, "
                f"spaCy: {len(spacy_entities)}"
            )
            
            # Process entities by type
            final_entities = []
            
            # Add Presidio primary entities
            presidio_primary = [
                e for e in presidio_entities 
                if e.entity_type in self.presidio_entities
            ]
            final_entities.extend(presidio_primary)
            
            # Add spaCy primary entities
            spacy_primary = [
                e for e in spacy_entities 
                if e.entity_type in self.spacy_entities
            ]
            final_entities.extend(spacy_primary)
            
            # Process ensemble entities
            ensemble_results = self._process_ensemble_entities(
                presidio_entities, 
                spacy_entities
            )
            final_entities.extend(ensemble_results)
            
            # Resolve overlaps
            final_entities = self._resolve_overlaps(final_entities)
            
            self.logger.debug(
                f"Final detection count: {len(final_entities)} "
                f"(Presidio primary: {len(presidio_primary)}, "
                f"spaCy primary: {len(spacy_primary)}, "
                f"Ensemble: {len(ensemble_results)})"
            )
            
            return final_entities
            
        except Exception as e:
            self.logger.error(f"Error in ensemble detection: {str(e)}")
            return []

    def _process_ensemble_entities(
        self,
        presidio_entities: List[Entity],
        spacy_entities: List[Entity]
    ) -> List[Entity]:
        """Process entities requiring ensemble approach."""
        try:
            ensemble_results = []
            
            # Get ensemble candidates
            presidio_candidates = {
                self._get_span_key(e): e 
                for e in presidio_entities 
                if e.entity_type in self.ensemble_entities
            }
            spacy_candidates = {
                self._get_span_key(e): e 
                for e in spacy_entities 
                if e.entity_type in self.ensemble_entities
            }
            
            # Process overlapping detections
            processed_spans = set()
            for span_key, presidio_ent in presidio_candidates.items():
                # Look for matching spaCy entities
                if span_key in spacy_candidates:
                    spacy_ent = spacy_candidates[span_key]
                    combined = self._combine_entities(presidio_ent, spacy_ent)
                    if combined:
                        ensemble_results.append(combined)
                        processed_spans.add(span_key)
            
            # Add high-confidence individual detections
            self._add_individual_detections(
                ensemble_results,
                presidio_candidates,
                processed_spans,
                "presidio"
            )
            self._add_individual_detections(
                ensemble_results,
                spacy_candidates,
                processed_spans,
                "spacy"
            )
            
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error processing ensemble entities: {str(e)}")
            return []

    def _add_individual_detections(
        self,
        results: List[Entity],
        candidates: Dict[Tuple[int, int], Entity],
        processed_spans: Set[Tuple[int, int]],
        source: str
    ) -> None:
        """Add high-confidence individual detections."""
        for span_key, entity in candidates.items():
            if span_key not in processed_spans:
                if entity.confidence >= self.min_individual_confidence:
                    results.append(entity)
                    processed_spans.add(span_key)

    def _combine_entities(
        self,
        presidio_ent: Entity,
        spacy_ent: Entity
    ) -> Optional[Entity]:
        """Combine two entity detections into one."""
        try:
            # Calculate combined confidence
            combined_confidence = (
                presidio_ent.confidence * self.ensemble_weights["presidio"] +
                spacy_ent.confidence * self.ensemble_weights["spacy"]
            )
            
            if combined_confidence < self.min_combined_confidence:
                return None
                
            # Use the entity with larger span
            base_ent = (
                presidio_ent 
                if (presidio_ent.end - presidio_ent.start) > (spacy_ent.end - spacy_ent.start)
                else spacy_ent
            )
            
            return Entity(
                text=base_ent.text,
                entity_type=base_ent.entity_type,
                confidence=combined_confidence,
                start=base_ent.start,
                end=base_ent.end,
                source="ensemble",
                metadata={
                    "presidio_confidence": presidio_ent.confidence,
                    "spacy_confidence": spacy_ent.confidence,
                    "presidio_metadata": presidio_ent.metadata,
                    "spacy_metadata": spacy_ent.metadata
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error combining entities: {str(e)}")
            return None

    def _resolve_overlaps(self, entities: List[Entity]) -> List[Entity]:
        """Resolve overlapping entities, preferring higher confidence."""
        if not entities:
            return []
            
        try:
            # Sort by start position and confidence
            sorted_entities = sorted(
                entities,
                key=lambda x: (x.start, -x.confidence)
            )
            
            resolved = []
            current = sorted_entities[0]
            
            for next_ent in sorted_entities[1:]:
                if current.end > next_ent.start:
                    # Overlapping entities - keep the one with higher confidence
                    if next_ent.confidence > current.confidence:
                        current = next_ent
                else:
                    resolved.append(current)
                    current = next_ent
                    
            resolved.append(current)
            
            return resolved
            
        except Exception as e:
            self.logger.error(f"Error resolving overlaps: {str(e)}")
            return []

    @staticmethod
    def _get_span_key(entity: Entity) -> Tuple[int, int]:
        """Get a unique key for an entity span."""
        return (entity.start, entity.end)