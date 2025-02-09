# tools/pipeline_debug.py

###############################################################################
# SECTION 1: IMPORTS AND SETUP
###############################################################################
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

###############################################################################
# SECTION 2: DETECTOR-SPECIFIC DEBUG FUNCTIONS
###############################################################################
def initialize_recognizers(analyzer):
    """Initialize required recognizers if missing"""
    from presidio_analyzer.predefined_recognizers import (
        EmailRecognizer,
        PhoneRecognizer,
        IpRecognizer,
        CreditCardRecognizer,
        DateRecognizer
    )
    
    registry = analyzer.registry
    required_recognizers = {
        "EmailRecognizer": EmailRecognizer,
        "PhoneRecognizer": PhoneRecognizer,
        "IpRecognizer": IpRecognizer,
        "CreditCardRecognizer": CreditCardRecognizer,
        "DateRecognizer": DateRecognizer
    }
    
    # Get current recognizer class names
    current_recognizers = [r.__class__.__name__ for r in registry.recognizers]
    
    # Add any missing recognizers
    for name, recognizer_class in required_recognizers.items():
        if name not in current_recognizers:
            print(f"Adding missing {name}")
            registry.add_recognizer(recognizer_class())

def debug_presidio_stage(detector, text: str, entity_type: str):
    """Detailed debug of Presidio detection stage"""
    print("\n1. Presidio Detection Stage")
    print("-" * 30)
    
    # First check recognizer initialization
    print("Checking Presidio recognizers...")
    raw_analyzer = detector.analyzer
    registry = raw_analyzer.registry
    
    # List available recognizers
    available_recognizers = registry.recognizers
    print(f"\nAvailable recognizers ({len(available_recognizers)}):")
    for recognizer in available_recognizers:
        print(f"- {recognizer.__class__.__name__}")
        print(f"  Supported entities: {recognizer.supported_entities}")
    
    # Initialize missing recognizers
    initialize_recognizers(raw_analyzer)
    
    # Show text snippet around known email position
    start_pos = max(0, 104-20)  # 20 chars before known position
    end_pos = min(len(text), 134+20)  # 20 chars after
    print(f"\nText context (pos 104-134):")
    print(f"...{text[start_pos:end_pos]}...")
    
    try:
        # Raw Presidio detection
        raw_results = raw_analyzer.analyze(
            text=text,
            language="en",
            entities=[entity_type],
            allow_list=None
        )
        print(f"\nRaw Presidio results:")
        print(f"Found {len(raw_results)} entities")
        for result in raw_results:
            print(f"- Text: {text[result.start:result.end]}")
            print(f"  Score: {result.score}")
            print(f"  Position: {result.start}-{result.end}")
    except Exception as e:
        print(f"\nError in raw Presidio detection: {str(e)}")
    
    # Detector processing
    print("\nChecking detector processing...")
    try:
        results = detector.detect_entities(text)
        entities = [e for e in results if e.entity_type == entity_type]
        
        print(f"Processed Presidio results:")
        print(f"Found {len(entities)} entities")
        for entity in entities:
            print(f"- Text: {entity.text}")
            print(f"  Confidence: {entity.confidence}")
            print(f"  Position: {entity.start}-{entity.end}")
            if hasattr(entity, 'metadata'):
                print(f"  Metadata: {entity.metadata}")
    except Exception as e:
        print(f"Error in detector processing: {str(e)}")

def debug_spacy_stage(detector, text: str, entity_type: str):
    """Detailed debug of spaCy detection stage"""
    print("\n2. spaCy Detection Stage")
    print("-" * 30)
    
    try:
        # Get raw spaCy detections
        doc = detector.nlp(text)
        print(f"\nRaw spaCy entities:")
        for ent in doc.ents:
            print(f"- Text: {ent.text}")
            print(f"  Label: {ent.label_}")
            print(f"  Position: {ent.start_char}-{ent.end_char}")
        
        # Get processed results
        results = detector.detect_entities(text)
        entities = [e for e in results if e.entity_type == entity_type]
        
        print(f"\nProcessed spaCy results:")
        print(f"Found {len(entities)} entities")
        for entity in entities:
            print(f"- Text: {entity.text}")
            print(f"  Confidence: {getattr(entity, 'confidence', 'N/A')}")
            print(f"  Position: {entity.start}-{entity.end}")
    except Exception as e:
        print(f"Error in spaCy processing: {str(e)}")

###############################################################################
# SECTION 3: CONFIGURATION DEBUG FUNCTIONS
###############################################################################
def debug_routing_config(config_loader: Any, entity_type: str):
    """Debug routing configuration for entity type"""
    print("\n3. Routing Configuration")
    print("-" * 30)
    
    routing_config = config_loader.get_config("entity_routing")
    if entity_type in routing_config["routing"]["presidio_primary"]["entities"]:
        threshold = routing_config["routing"]["presidio_primary"]["thresholds"][entity_type]
        print(f"Entity type routed to Presidio primary")
        print(f"Confidence threshold: {threshold}")
    elif entity_type in routing_config["routing"]["ensemble_required"]["entities"]:
        config = routing_config["routing"]["ensemble_required"]["confidence_thresholds"][entity_type]
        print(f"Entity type requires ensemble")
        print(f"Minimum combined confidence: {config['minimum_combined']}")
        print(f"Presidio weight: {config['presidio_weight']}")
        print(f"spaCy weight: {config['spacy_weight']}")
    else:
        print(f"WARNING: {entity_type} not found in routing configuration")

def debug_validation_config(config_loader: Any, entity_type: str):
    """Debug validation configuration for entity type"""
    print("\n4. Validation Configuration")
    print("-" * 30)
    
    validation_params = config_loader.get_config("validation_params")
    if entity_type in validation_params:
        print("Validation parameters:")
        params = validation_params[entity_type]
        for key, value in params.items():
            print(f"- {key}: {value}")
    else:
        print(f"WARNING: No validation parameters found for {entity_type}")

###############################################################################
# SECTION 4: RESULTS DEBUG FUNCTIONS
###############################################################################
def debug_final_results(ensemble, text: str, entity_type: str):
    """Debug final detection results"""
    print("\n5. Final Results")
    print("-" * 30)
    
    final_results = ensemble.detect_entities(text)
    final_entities = [e for e in final_results if e.entity_type == entity_type]
    
    if final_entities:
        for entity in final_entities:
            print(f"Found: {entity.text}")
            print(f"Final confidence: {entity.confidence}")
            print(f"Position: {entity.start}-{entity.end}")
            print(f"Source: {getattr(entity, 'source', 'unknown')}")
    else:
        print("No entities in final results")

###############################################################################
# SECTION 5: MAIN PIPELINE DEBUG FUNCTION
###############################################################################
def debug_entity_pipeline(test_id: str, entity_type: str):
    """Debug the full pipeline for a specific entity type."""
    print(f"\nDebugging pipeline for {entity_type} in {test_id}")
    print("=" * 50)
    
    try:
        # Import after path setup
        from app.utils.config_loader import ConfigLoader
        from evaluation.evaluate import StageOneEvaluator
        
        # Initialize components
        config_loader = ConfigLoader()
        config_loader.load_all_configs()
        evaluator = StageOneEvaluator()
        
        # Load document
        text = evaluator._load_test_document(test_id)
        print(f"\nLoaded document ({len(text)} chars)")
        
        # Run debug stages
        debug_presidio_stage(evaluator.ensemble.presidio_detector, text, entity_type)
        debug_spacy_stage(evaluator.ensemble.spacy_detector, text, entity_type)
        debug_routing_config(config_loader, entity_type)
        debug_validation_config(config_loader, entity_type)
        debug_final_results(evaluator.ensemble, text, entity_type)
            
    except Exception as e:
        print(f"Error during pipeline debug: {str(e)}")
        import traceback
        print(traceback.format_exc())

###############################################################################
# SECTION 6: COMMAND LINE INTERFACE
###############################################################################
def main():
    """Command line interface for pipeline debugger"""
    if len(sys.argv) != 3:
        print("Usage: python tools/pipeline_debug.py <test_id> <entity_type>")
        print("\nExample: python tools/pipeline_debug.py alexis_rivera EMAIL_ADDRESS")
        sys.exit(1)
        
    test_id = sys.argv[1]
    entity_type = sys.argv[2]
    debug_entity_pipeline(test_id, entity_type)

if __name__ == "__main__":
    main()