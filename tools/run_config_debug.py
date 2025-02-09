#!/usr/bin/env python3
# tools/run_config_debug.py

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from tools.config_debug import ConfigDebugger

def main():
    """Run configuration debugging"""
    print("Resume Redaction System - Configuration Debugger")
    print("=" * 50)
    
    debugger = ConfigDebugger(project_root)
    
    print("\nChecking configuration structure...")
    structure_results = debugger.verify_config_structure()
    
    print("\nConfiguration file status:")
    for name, exists in structure_results.items():
        status = "✓" if exists else "✗"
        print(f"{status} {name}")
        
    print("\nChecking configuration references...")
    reference_results = debugger.verify_config_references()
    
    print("\nConfiguration reference status:")
    for name, valid in reference_results.items():
        status = "✓" if valid else "✗"
        print(f"{status} {name}")
        
    print("\nChecking detector configurations...")
    detector_results = debugger.verify_detectors_config()
    
    print("\nDetector configuration status:")
    for name, valid in detector_results.items():
        status = "✓" if valid else "✗"
        print(f"{status} {name}")
        
    print("\nGenerating suggestions...")
    suggestions = debugger.suggest_fixes({
        'structure': structure_results,
        'references': reference_results,
        'detectors': detector_results
    })
    
    if suggestions != "All configurations appear valid.":
        print("\nRequired fixes:")
        print(suggestions)
    else:
        print("\nAll configurations are valid!")

if __name__ == "__main__":
    main()