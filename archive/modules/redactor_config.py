import yaml
from typing import Dict, Any, List, Set

def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Load YAML configuration from a file.
    """
    try:
        with open(path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML config from {path}: {e}")
        return {}

def load_trigger_words(path: str) -> Set[str]:
    """
    Load trigger words from a YAML file.
    """
    try:
        config = load_yaml_config(path)
        return set(config.get("trigger_words", []))
    except Exception as e:
        print(f"Error loading trigger words from {path}: {e}")
        return set()

def load_detection_patterns(path: str) -> List[Dict[str, Any]]:
    """
    Load detection patterns from a YAML file.
    """
    try:
        config = load_yaml_config(path)
        return config.get("patterns", [])
    except Exception as e:
        print(f"Error loading detection patterns from {path}: {e}")
        return []
