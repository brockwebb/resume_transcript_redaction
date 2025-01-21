#app/utils/config_loader.py

import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Union
from app.utils.logger import RedactionLogger


class ConfigLoader:
    """Enhanced configuration loader with robust YAML handling and dynamic access."""

    def __init__(self):
        self.logger = RedactionLogger(
            name="config_loader",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True,
        )

        self.app_dir = Path(__file__).parent.parent
        self.config_dir = self.app_dir / "config"
        self.redactor_config_dir = (
            Path(__file__).parent.parent.parent / "redactor" / "config"
        )

        self.configs: Dict[str, Optional[Dict]] = {}
        self.loaded_paths: Dict[str, Path] = {}

    def _safe_yaml_load(self, file_path: Path) -> Optional[Dict]:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file)
                if not isinstance(config_data, dict):
                    self.logger.error(f"Invalid YAML structure in {file_path}")
                    return None
                return config_data
        except Exception as e:
            self.logger.error(f"Error loading {file_path}: {e}")
            return None

    def load_all_configs(self) -> bool:
        """
        Load all configuration files and store them in the configs dictionary.

        Returns:
            bool: True if all main configurations are loaded successfully, False otherwise.
        """
        self.logger.info("Loading all configuration files...")

        # Load the main configuration first
        main_config = self._safe_yaml_load(self.config_dir / "config.yaml")
        if not main_config:
            self.logger.error("Failed to load main configuration")
            return False

        self.configs["main"] = main_config

        # Load additional configurations defined in the main config
        config_files = {
            "detection": main_config.get("redaction_patterns_path"),
            "confidential": main_config.get("confidential_terms_path"),
            "word_filters": main_config.get("word_filters_path"),
            "entity_routing": main_config.get("entity_routing_path"),
            "core_patterns": main_config.get("core_patterns_path")
        }

        for name, path in config_files.items():
            resolved_path = self._resolve_path(path)
            if not resolved_path:
                self.logger.error(f"Invalid path for {name} configuration")
                continue
            self._load_single_config(name, resolved_path)

        return True

    def _resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        """
        Resolve relative or absolute file paths.

        Args:
            path_str (str): File path to resolve.

        Returns:
            Path: Resolved Path object if the path exists, None otherwise.
        """
        if not path_str:
            return None
        path = Path(path_str)
        if path.is_absolute() and path.exists():
            return path
        app_path = self.app_dir / path
        if app_path.exists():
            return app_path
        redactor_path = self.redactor_config_dir / path.name
        if redactor_path.exists():
            return redactor_path
        self.logger.error(f"Could not resolve path: {path_str}")
        return None

    def _load_single_config(self, name: str, filepath: Path) -> None:
        """
        Load a single YAML configuration file.

        Args:
            name (str): Configuration name to use as a key.
            filepath (Path): Path to the configuration file.
        """
        config_data = self._safe_yaml_load(filepath)
        if config_data:
            self.configs[name] = config_data
            self.loaded_paths[name] = filepath
            self.logger.info(f"Successfully loaded {name} configuration")

    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Dynamically retrieve a configuration or specific keys within it.

        Args:
            config_name (str): The name of the configuration to retrieve.
            key (str, optional): Specific key within the configuration. Defaults to None.
            default (Any, optional): Default value if the key or configuration does not exist. Defaults to None.

        Returns:
            Any: The configuration or specific key value.
        """
        config = self.configs.get(config_name, {})
        if key:
            keys = key.split(".")
            for k in keys:
                if isinstance(config, dict) and k in config:
                    config = config[k]
                else:
                    return default
        return config or default

    def get_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific configuration by name.
    
        Args:
            config_name (str): The name of the configuration to retrieve.
    
        Returns:
            Optional[Dict[str, Any]]: The configuration dictionary if found, else None.
        """
        return self.configs.get(config_name)

    
    def validate_config(self, config_name: str, required_keys: Union[str, list]) -> bool:
        """
        Validate if a configuration contains required keys.

        Args:
            config_name (str): The name of the configuration to validate.
            required_keys (list or str): A single key or list of required keys.

        Returns:
            bool: True if all required keys are present, False otherwise.
        """
        config = self.configs.get(config_name)
        if not config:
            self.logger.error(f"{config_name} configuration not loaded")
            return False

        if isinstance(required_keys, str):
            required_keys = [required_keys]

        for key in required_keys:
            if key not in config:
                self.logger.error(f"Missing required key '{key}' in {config_name} configuration")
                return False

        return True
