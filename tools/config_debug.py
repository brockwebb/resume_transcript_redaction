# config_debug.py

import os
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional

class ConfigDebugger:
    """Utility for debugging configuration loading issues"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.app_config_dir = self.project_root / "app" / "config"
        self.redactor_config_dir = self.project_root / "redactor" / "config"
        
    def verify_config_structure(self) -> Dict[str, bool]:
        """Verify all required configuration files exist and are valid"""
        required_files = {
            'app_config': {
                'path': self.app_config_dir / "config.yaml",
                'type': 'yaml'
            },
            'entity_routing': {
                'path': self.redactor_config_dir / "entity_routing.yaml",
                'type': 'yaml'
            },
            'validation_params': {
                'path': self.redactor_config_dir / "validation_params.json",
                'type': 'json'
            },
            'validation_config': {
                'path': self.redactor_config_dir / "validation_config.json",
                'type': 'json'
            }
        }
        
        results = {}
        for name, info in required_files.items():
            file_exists = info['path'].exists()
            if file_exists:
                try:
                    if info['type'] == 'yaml':
                        with open(info['path']) as f:
                            yaml.safe_load(f)
                    else:
                        with open(info['path']) as f:
                            json.load(f)
                    results[name] = True
                except Exception as e:
                    results[name] = False
                    print(f"Error parsing {name}: {str(e)}")
            else:
                results[name] = False
                print(f"Missing file: {info['path']}")
                
        return results
    
    def verify_config_references(self) -> Dict[str, bool]:
        """Verify that configuration files reference each other correctly"""
        try:
            # Load main config
            with open(self.app_config_dir / "config.yaml") as f:
                main_config = yaml.safe_load(f)
            
            # Check config paths
            paths = main_config.get('config_paths', {})
            results = {}
            
            for key, path in paths.items():
                # Convert relative paths to absolute
                if path.startswith('./'):
                    path = self.project_root / path[2:]
                    
                results[f"{key}_path"] = Path(path).exists()
                
            return results
        except Exception as e:
            print(f"Error checking config references: {str(e)}")
            return {'main_config_load': False}
    
    def verify_detectors_config(self) -> Dict[str, bool]:
        """Verify detector-specific configurations"""
        try:
            # Load entity routing config
            with open(self.redactor_config_dir / "entity_routing.yaml") as f:
                routing_config = yaml.safe_load(f)
            
            results = {
                'has_presidio_config': 'presidio_primary' in routing_config.get('routing', {}),
                'has_spacy_config': 'spacy_primary' in routing_config.get('routing', {}),
                'has_ensemble_config': 'ensemble_required' in routing_config.get('routing', {})
            }
            
            # Check specific required fields
            if results['has_presidio_config']:
                presidio_config = routing_config['routing']['presidio_primary']
                results['presidio_has_entities'] = bool(presidio_config.get('entities'))
                results['presidio_has_thresholds'] = bool(presidio_config.get('thresholds'))
            
            if results['has_spacy_config']:
                spacy_config = routing_config['routing']['spacy_primary']
                results['spacy_has_entities'] = bool(spacy_config.get('entities'))
                results['spacy_has_thresholds'] = bool(spacy_config.get('thresholds'))
            
            return results
        except Exception as e:
            print(f"Error checking detector configs: {str(e)}")
            return {'routing_config_load': False}
    
    def suggest_fixes(self, results: Dict[str, Dict[str, bool]]) -> str:
        """Generate suggestions based on verification results"""
        suggestions = []
        
        # Check basic config structure
        for config_name, exists in results['structure'].items():
            if not exists:
                suggestions.append(f"- Create missing configuration file: {config_name}")
        
        # Check config references
        for ref_name, valid in results['references'].items():
            if not valid:
                suggestions.append(f"- Fix broken configuration reference: {ref_name}")
        
        # Check detector configurations
        detector_results = results['detectors']
        if not detector_results.get('routing_config_load', True):
            suggestions.append("- Fix entity_routing.yaml format")
        else:
            if not detector_results.get('has_presidio_config', False):
                suggestions.append("- Add missing presidio_primary configuration")
            if not detector_results.get('has_spacy_config', False):
                suggestions.append("- Add missing spacy_primary configuration")
            if not detector_results.get('has_ensemble_config', False):
                suggestions.append("- Add missing ensemble_required configuration")
        
        return "\n".join(suggestions) if suggestions else "All configurations appear valid."

def main():
    # Assuming script is run from project root
    debugger = ConfigDebugger(".")
    
    print("Checking configuration structure...")
    structure_results = debugger.verify_config_structure()
    
    print("\nChecking configuration references...")
    reference_results = debugger.verify_config_references()
    
    print("\nChecking detector configurations...")
    detector_results = debugger.verify_detectors_config()
    
    print("\nGenerating suggestions...")
    suggestions = debugger.suggest_fixes({
        'structure': structure_results,
        'references': reference_results,
        'detectors': detector_results
    })
    
    print("\nSuggested fixes:")
    print(suggestions)

if __name__ == "__main__":
    main()