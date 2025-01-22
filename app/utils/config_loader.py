# app/utils/config_loader.py

import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Union
from app.utils.logger import RedactionLogger


class ConfigLoader:
    """Enhanced configuration loader with robust YAML handling and dynamic access."""

    def __init__(self):
        """Initialize ConfigLoader with logging and path setup."""
        self.logger = RedactionLogger(
            name="config_loader",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True,
        )

        # Setup paths
        self.app_dir = Path(__file__).parent.parent
        self.project_root = self.app_dir.parent
        self.config_dir = self.app_dir / "config"
        self.redactor_config_dir = self.project_root / "redactor" / "config"

        # Debug path information
        self.logger.debug(f"Project root: {self.project_root}")
        self.logger.debug(f"App config dir: {self.config_dir}")
        self.logger.debug(f"Redactor config dir: {self.redactor_config_dir}")

        self.configs: Dict[str, Optional[Dict]] = {}
        self.loaded_paths: Dict[str, Path] = {}

    def _safe_yaml_load(self, file_path: Path) -> Optional[Dict]:
        """Safely load YAML file with error handling.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Dict if successful, None if failed
        """
        try:
            self.logger.debug(f"Attempting to load YAML from: {file_path}")
            if not file_path.exists():
                self.logger.error(f"File does not exist: {file_path}")
                return None

            with open(file_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                if not isinstance(config_data, dict):
                    self.logger.error(f"Invalid YAML structure in {file_path} - expected dict, got {type(config_data)}")
                    return None
                self.logger.debug(f"Successfully loaded YAML from {file_path}")
                return config_data
        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {file_path}: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error loading {file_path}: {e}")
            return None

    def load_all_configs(self) -> bool:
        """Load all configuration files and store them in the configs dictionary."""
        self.logger.info("Starting configuration loading process...")

        # Load main configuration
        main_config_path = self.config_dir / "config.yaml"
        self.logger.debug(f"Loading main config from: {main_config_path}")
        
        main_config = self._safe_yaml_load(main_config_path)
        if not main_config:
            self.logger.error("Failed to load main configuration - stopping config load")
            return False

        self.configs["main"] = main_config
        self.loaded_paths["main"] = main_config_path

        # Define and load additional configurations
        config_files = {
            "detection": main_config.get("redaction_patterns_path"),
            "confidential": main_config.get("confidential_terms_path"),
            "word_filters": main_config.get("word_filters_path"),
            "entity_routing": main_config.get("entity_routing_path"),
            "core_patterns": main_config.get("core_patterns_path")
        }

        self.logger.debug("Config paths from main config:")
        for name, path in config_files.items():
            self.logger.debug(f"  {name}: {path}")

        for name, path in config_files.items():
            resolved_path = self._resolve_path(path)
            if not resolved_path:
                self.logger.error(f"Could not resolve path for {name} config")
                continue
            self._load_single_config(name, resolved_path)

        # Log final loading status
        loaded_configs = [name for name, config in self.configs.items() if config is not None]
        self.logger.info(f"Configuration loading complete. Loaded configs: {loaded_configs}")
        self.logger.debug("Loaded paths:")
        for name, path in self.loaded_paths.items():
            self.logger.debug(f"  {name}: {path}")

        return True

    def _resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        """Resolve relative or absolute file paths with detailed debugging."""
        if not path_str:
            self.logger.error("Received empty path string")
            return None

        self.logger.debug(f"Resolving path: {path_str}")
        
        # Try absolute path
        path = Path(path_str)
        if path.is_absolute():
            self.logger.debug(f"Checking absolute path: {path}")
            if path.exists():
                self.logger.debug("Found file at absolute path")
                return path
            else:
                self.logger.debug("Absolute path does not exist")
        
        # Try relative to app directory
        app_path = self.app_dir / path_str.replace("../", "")
        self.logger.debug(f"Checking relative to app dir: {app_path}")
        if app_path.exists():
            self.logger.debug("Found file relative to app directory")
            return app_path
        
        # Try relative to redactor config directory
        redactor_path = self.redactor_config_dir / Path(path_str).name
        self.logger.debug(f"Checking in redactor config dir: {redactor_path}")
        if redactor_path.exists():
            self.logger.debug("Found file in redactor config directory")
            return redactor_path

        self.logger.error(f"Path resolution failed for: {path_str}")
        self.logger.debug("Tried locations:")
        self.logger.debug(f"  1. Absolute: {path}")
        self.logger.debug(f"  2. App relative: {app_path}")
        self.logger.debug(f"  3. Redactor config: {redactor_path}")
        return None

    def _load_single_config(self, name: str, filepath: Path) -> None:
        """Load a single YAML configuration file with enhanced error reporting."""
        self.logger.debug(f"Loading {name} configuration from {filepath}")
        
        config_data = self._safe_yaml_load(filepath)
        if config_data:
            self.configs[name] = config_data
            self.loaded_paths[name] = filepath
            self.logger.info(f"Successfully loaded {name} configuration")
            self.logger.debug(f"Config structure for {name}:")
            self._log_config_structure(config_data)
        else:
            self.logger.error(f"Failed to load {name} configuration")
            if name in self.configs:
                self.logger.debug(f"Removing failed config '{name}' from loaded configs")
                del self.configs[name]

    def _log_config_structure(self, config: Dict, prefix: str = "") -> None:
        """Recursively log configuration structure for debugging."""
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.debug(f"{prefix}{key}:")
                self._log_config_structure(value, prefix + "  ")
            else:
                self.logger.debug(f"{prefix}{key}: {type(value)}")

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific configuration by name with debug info."""
        if config_name not in self.configs:
            self.logger.debug(f"Requested config '{config_name}' not found in loaded configs: {list(self.configs.keys())}")
            return None
            
        config = self.configs.get(config_name)
        self.logger.debug(f"Retrieved config '{config_name}' {'successfully' if config else 'but it was None'}")
        return config

    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Get configuration value with detailed path traversal logging."""
        config = self.configs.get(config_name, {})
        self.logger.debug(f"Accessing {config_name}: {'found' if config else 'not found'}")
        
        if key:
            self.logger.debug(f"Looking for key: {key}")
            keys = key.split(".")
            current = config
            path = []
            
            for k in keys:
                path.append(k)
                if isinstance(current, dict) and k in current:
                    current = current[k]
                    self.logger.debug(f"Found key at path: {'.'.join(path)}")
                else:
                    self.logger.debug(f"Key not found at path: {'.'.join(path)}")
                    return default
                    
            return current
        return config or default

    def validate_config(self, config_name: str, required_keys: Union[str, list]) -> bool:
        """Validate configuration with detailed reporting."""
        config = self.configs.get(config_name)
        if not config:
            self.logger.error(f"Cannot validate {config_name} - configuration not loaded")
            return False

        if isinstance(required_keys, str):
            required_keys = [required_keys]

        self.logger.debug(f"Validating {config_name} configuration")
        self.logger.debug(f"Required keys: {required_keys}")
        
        missing_keys = []
        for key in required_keys:
            if key not in config:
                missing_keys.append(key)
                self.logger.error(f"Missing required key '{key}' in {config_name}")

        if missing_keys:
            self.logger.error(f"Validation failed for {config_name}. Missing keys: {missing_keys}")
            return False

        self.logger.debug(f"Validation successful for {config_name}")
        return True