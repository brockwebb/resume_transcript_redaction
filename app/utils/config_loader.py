import yaml
import json
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

        # Setup important paths
        self.app_dir = Path(__file__).parent.parent
        self.project_root = self.app_dir.parent
        self.config_dir = self.app_dir / "config"
        self.redactor_config_dir = self.project_root / "redactor" / "config"

        self.logger.debug(f"Project root: {self.project_root}")
        self.logger.debug(f"App config dir: {self.config_dir}")
        self.logger.debug(f"Redactor config dir: {self.redactor_config_dir}")

        # Initialize storage for configurations and a flag for load status.
        self.configs: Dict[str, Optional[Dict]] = {}
        self.loaded_paths: Dict[str, Path] = {}
        self._loaded = False

    def _safe_yaml_load(self, file_path: Path) -> Optional[Dict]:
        """Safely load a YAML file."""
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

    def _load_single_json_config(self, name: str, filepath: Path) -> None:
        """Load a single JSON configuration file."""
        try:
            self.logger.debug(f"Loading {name} configuration from {filepath}")
            with open(filepath, "r", encoding="utf-8") as file:
                config_data = json.load(file)
            self.configs[name] = config_data
            self.loaded_paths[name] = filepath
            self.logger.info(f"Successfully loaded {name} configuration")
            self.logger.debug(f"Config structure for {name}: {type(config_data)}")
        except Exception as e:
            self.logger.error(f"Failed to load {name} configuration from {filepath}: {e}")

    def _load_single_config(self, name: str, filepath: Path) -> None:
        """Load a single YAML configuration file."""
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

    def _resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        """Resolve a file path, checking several locations."""
        if not path_str:
            self.logger.error("Received empty path string")
            return None
        self.logger.debug(f"Resolving path: {path_str}")
        path = Path(path_str)
        if path.is_absolute():
            self.logger.debug(f"Checking absolute path: {path}")
            if path.exists():
                self.logger.debug("Found file at absolute path")
                return path
            else:
                self.logger.debug("Absolute path does not exist")
        # Try relative to the app directory
        app_path = self.app_dir / path_str.replace("../", "")
        self.logger.debug(f"Checking relative to app dir: {app_path}")
        if app_path.exists():
            self.logger.debug("Found file relative to app directory")
            return app_path
        # Finally, try the redactor config directory
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

    def load_all_configs(self) -> bool:
        """Load all configuration files, if not already loaded."""
        if self._loaded:
            self.logger.info("Configurations already loaded; skipping reload.")
            return True

        self.logger.info("Starting configuration loading process...")

        # Load the main configuration file.
        main_config_path = self.config_dir / "config.yaml"
        self.logger.debug(f"Loading main config from: {main_config_path}")
        main_config = self._safe_yaml_load(main_config_path)
        if not main_config:
            self.logger.error("Failed to load main configuration - stopping config load")
            return False

        self.configs["main"] = main_config
        self.loaded_paths["main"] = main_config_path

        # Load direct sections from main config.
        for section in ["core_settings"]:
            if section in main_config:
                self.logger.debug(f"Loading {section} from main config")
                self.configs[section] = main_config[section]
                self.loaded_paths[section] = main_config_path

        # Define additional config file paths.
        config_files = {
            "entity_routing": main_config.get("entity_routing_path"),
            "validation_params": main_config.get("validation_params_path"),
            "validation_config": main_config.get("validation_config"),
            "word_filters_path": main_config.get("word_filters_path"),
        }
        self.logger.debug("Config paths from main config:")
        for name, path in config_files.items():
            self.logger.debug(f"  {name}: {path}")

        # Attempt to load each additional config.
        for name, path in config_files.items():
            resolved_path = self._resolve_path(path)
            if not resolved_path:
                self.logger.error(f"Could not resolve path for {name} config")
                continue
            if resolved_path.suffix == ".json":
                self._load_single_json_config(name, resolved_path)
            else:
                self._load_single_config(name, resolved_path)

        loaded_configs = [name for name, config in self.configs.items() if config is not None]
        self.logger.info(f"Configuration loading complete. Loaded configs: {loaded_configs}")
        self.logger.debug("Loaded paths:")
        for name, path in self.loaded_paths.items():
            self.logger.debug(f"  {name}: {path}")
        self._loaded = True
        return True

    def _log_config_structure(self, config: Dict, prefix: str = "") -> None:
        """Recursively log the structure of a configuration."""
        for key, value in config.items():
            if isinstance(value, dict):
                self.logger.debug(f"{prefix}{key}:")
                self._log_config_structure(value, prefix + "  ")
            else:
                self.logger.debug(f"{prefix}{key}: {type(value)}")

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific configuration."""
        if config_name not in self.configs:
            self.logger.debug(f"Requested config '{config_name}' not found in loaded configs: {list(self.configs.keys())}")
            return None
        config = self.configs.get(config_name)
        self.logger.debug(f"Retrieved config '{config_name}' {'successfully' if config else 'but it was None'}")
        return config

    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """Retrieve a configuration value, optionally following a dotted key path."""
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
        """Validate that a configuration contains the required keys."""
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

# ---------------------------------------------------------------------------
# Create a global instance of the configuration loader.
# All other modules should import and use this instance.
global_config_loader = ConfigLoader()

if __name__ == "__main__":
    if global_config_loader.load_all_configs():
        routing_config = global_config_loader.get_config("entity_routing")
        print("Loaded Entity Routing Configuration:", routing_config)
        if global_config_loader.validate_config("entity_routing", ["routing", "settings"]):
            print("Configuration validation successful")
