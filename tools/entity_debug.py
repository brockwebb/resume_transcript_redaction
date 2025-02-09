# tools/entity_debug.py

import sys
from pathlib import Path
import yaml
import fitz  # PyMuPDF
from typing import List, Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

class EntityDebugger:
    """Debug tool focused on single entity type detection"""
    
    def __init__(self, entity_type: str):
        self.entity_type = entity_type
        print(f"\nInitializing EntityDebugger for {entity_type}")
        
        # Setup paths
        self.test_suite_dir = project_root / "data" / "test_suite"
        print(f"Test suite directory: {self.test_suite_dir}")
        
    def load_ground_truth(self, test_id: str) -> List[Dict]:
        """Load ground truth annotations with error handling"""
        annotation_path = self.test_suite_dir / "annotations" / f"{test_id}.yaml"
        print(f"\nLooking for annotations at: {annotation_path}")
        
        if not annotation_path.exists():
            print(f"ERROR: Annotation file not found: {annotation_path}")
            return []
            
        try:
            with open(annotation_path) as f:
                annotations = yaml.safe_load(f)
                print(f"Successfully loaded annotations for {test_id}")
                return annotations.get("entities", [])
        except Exception as e:
            print(f"ERROR loading annotations: {str(e)}")
            return []
            
    def load_document(self, test_id: str) -> str:
        """Load document content with error handling"""
        pdf_path = self.test_suite_dir / "originals" / f"{test_id}.pdf"
        print(f"\nLooking for document at: {pdf_path}")
        
        if not pdf_path.exists():
            print(f"ERROR: Document not found: {pdf_path}")
            return ""
            
        try:
            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text() + "\n"
            print(f"Successfully loaded document content")
            return text
        except Exception as e:
            print(f"ERROR loading document: {str(e)}")
            return ""

    def analyze_entity(self, test_id: str) -> Dict[str, Any]:
        """Analyze detection for a specific entity type in a test case"""
        print(f"\nAnalyzing {self.entity_type} in test case {test_id}")
        
        # Load ground truth
        truth_entities = self.load_ground_truth(test_id)
        if not truth_entities:
            print("No ground truth entities found")
            return {}
            
        # Filter for our entity type
        truth_entities = [
            e for e in truth_entities 
            if e.get("type") == self.entity_type
        ]
        print(f"\nFound {len(truth_entities)} ground truth {self.entity_type} entities")
        
        # Load document
        text = self.load_document(test_id)
        if not text:
            print("No document content loaded")
            return {}
            
        try:
            # Import detection components
            from evaluation.evaluate import StageOneEvaluator
            evaluator = StageOneEvaluator()
            
            # Get Presidio results
            presidio_entities = evaluator.ensemble.presidio_detector.detect_entities(text)
            presidio_entities = [
                e for e in presidio_entities 
                if e.entity_type == self.entity_type
            ]
            print(f"\nPresidio found {len(presidio_entities)} {self.entity_type} entities")
            
            # Get spaCy results
            spacy_entities = evaluator.ensemble.spacy_detector.detect_entities(text)
            spacy_entities = [
                e for e in spacy_entities 
                if e.entity_type == self.entity_type
            ]
            print(f"spaCy found {len(spacy_entities)} {self.entity_type} entities")
            
            # Get ensemble results
            detected = evaluator.ensemble.detect_entities(text)
            detected_entities = [
                d for d in detected 
                if d.entity_type == self.entity_type
            ]
            print(f"Ensemble produced {len(detected_entities)} final {self.entity_type} entities")
            
            return {
                "test_id": test_id,
                "truth": truth_entities,
                "presidio": presidio_entities,
                "spacy": spacy_entities,
                "ensemble": detected_entities
            }
            
        except Exception as e:
            print(f"ERROR during analysis: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return {}

def main():
    """Debug a specific entity type"""
    if len(sys.argv) < 3:
        print("Usage: python -m tools.entity_debug <entity_type> <test_id>")
        print("\nExample: python -m tools.entity_debug EMAIL_ADDRESS alexis_rivera_corrected")
        sys.exit(1)
        
    entity_type = sys.argv[1]
    test_id = sys.argv[2]
    
    debugger = EntityDebugger(entity_type)
    results = debugger.analyze_entity(test_id)
    
    if results:
        print("\nAnalysis Results:")
        print("=" * 50)
        
        print("\nGround Truth Entities:")
        for e in results["truth"]:
            print(f"- {e['text']} (pos: {e['location']['start_char']}-{e['location']['end_char']})")
            
        if results.get("presidio"):
            print("\nPresidio Detections:")
            for e in results["presidio"]:
                print(f"- {e.text} (pos: {e.start}-{e.end}, conf: {e.confidence:.2f})")
            
        if results.get("spacy"):
            print("\nspaCy Detections:")
            for e in results["spacy"]:
                print(f"- {e.text} (pos: {e.start}-{e.end})")
            
        if results.get("ensemble"):
            print("\nEnsemble Final Results:")
            for e in results["ensemble"]:
                print(f"- {e.text} (pos: {e.start}-{e.end}, conf: {e.confidence:.2f}, source: {getattr(e, 'source', 'unknown')})")

if __name__ == "__main__":
    main()