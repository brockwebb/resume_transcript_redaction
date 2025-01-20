import yaml
from pathlib import Path
from typing import Dict, Optional, Any, Union
from app.utils.logger import RedactionLogger

class ConfigLoader:
    """Enhanced configuration loader with robust YAML handling."""

    def __init__(self):
        self.logger = RedactionLogger(
            name="config_loader",
            log_level="DEBUG",
            log_dir="logs",
            console_output=True,
        )

        # Get directory paths
        self.app_dir = Path(__file__).parent.parent
        self.config_dir = self.app_dir / "config"
        self.redactor_config_dir = Path(__file__).parent.parent.parent / "redactor" / "config"

        # Initialize storage
        self.configs: Dict[str, Optional[Dict]] = {}
        self.load_status: Dict[str, bool] = {}
        
        # Track loaded paths to avoid duplicates
        self.loaded_paths: Dict[str, Path] = {}

    def _safe_yaml_load(self, file_path: Path) -> Optional[Dict]:
        """
        Safely load YAML file with error handling and type validation.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Loaded configuration dict or None if loading fails
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                # Use safe_load to prevent arbitrary code execution
                config_data = yaml.safe_load(file)
                
                # Verify we got a dictionary
                if not isinstance(config_data, dict):
                    self.logger.error(
                        f"Invalid YAML structure in {file_path}. Expected dict, got {type(config_data)}"
                    )
                    return None
                    
                return config_data

        except yaml.YAMLError as e:
            self.logger.error(f"YAML parsing error in {file_path}: {str(e)}")
            if hasattr(e, 'problem_mark'):
                mark = e.problem_mark
                self.logger.error(
                    f"Error position: line {mark.line + 1}, column {mark.column + 1}"
                )
            return None
            
        except Exception as e:
            self.logger.error(f"Error reading {file_path}: {str(e)}")
            return None

    def _validate_loaded_config(self, config: Dict, config_name: str) -> bool:
        """
        Validate loaded configuration structure.
        
        Args:
            config: Configuration dictionary to validate
            config_name: Name of configuration for logging
            
        Returns:
            bool: Whether configuration is valid
        """
        try:
            if config_name == "entity_routing":
                required_keys = {"routing", "settings"}
                if not all(key in config for key in required_keys):
                    self.logger.error(
                        f"Missing required keys in entity_routing config. "
                        f"Required: {required_keys}"
                    )
                    return False
                    
                # Validate routing section structure
                routing = config.get("routing", {})
                if not isinstance(routing, dict):
                    self.logger.error("Routing section must be a dictionary")
                    return False
                    
                required_sections = {
                    "presidio_primary", 
                    "spacy_primary", 
                    "ensemble_required"
                }
                if not all(key in routing for key in required_sections):
                    self.logger.error(
                        f"Missing required sections in routing config. "
                        f"Required: {required_sections}"
                    )
                    return False

            # Add other config-specific validation as needed
            return True

        except Exception as e:
            self.logger.error(
                f"Error validating {config_name} configuration: {str(e)}"
            )
            return False

    def get_config_value(
        self, 
        config_name: str, 
        *keys: str, 
        default: Any = None
    ) -> Any:
        """
        Safely get nested configuration value.
        
        Args:
            config_name: Name of configuration to access
            *keys: Sequence of keys to access nested value
            default: Default value if path doesn't exist
            
        Returns:
            Configuration value or default
        """
        try:
            config = self.configs.get(config_name)
            if config is None:
                return default

            current = config
            for key in keys:
                if not isinstance(current, dict):
                    return default
                current = current.get(key)
                if current is None:
                    return default
                    
            return current

        except Exception as e:
            self.logger.error(
                f"Error accessing config {config_name} at keys {keys}: {str(e)}"
            )
            return default

    def load_all_configs(self) -> bool:
        """
        Load all configuration files.
        
        Returns:
            bool: Whether all critical configs loaded successfully
        """
        self.logger.info("Loading all configuration files...")
        
        # First load main config
        main_config = self._safe_yaml_load(self.config_dir / "config.yaml")
        if not main_config:
            self.logger.error("Failed to load main configuration")
            return False
            
        self.configs["main"] = main_config

        # Get paths from main config
        config_files = {
            "detection": self._resolve_path(main_config.get("redaction_patterns_path")),
            "confidential": self._resolve_path(main_config.get("confidential_terms_path")),
            "word_filters": self._resolve_path(main_config.get("word_filters_path")),
            "entity_routing": self._resolve_path(main_config.get("entity_routing_path"))
        }

        # Load each config
        for config_name, filepath in config_files.items():
            if not filepath:
                self.logger.error(f"Invalid path for {config_name} configuration")
                continue
                
            success = self._load_single_config(config_name, filepath)
            self.load_status[config_name] = success

            if not success and config_name in ["detection", "entity_routing"]:
                return False

        return True

    def _resolve_path(self, path_str: Optional[str]) -> Optional[Path]:
        """Resolve relative path from config to absolute path."""
        if not path_str:
            return None
            
        path = Path(path_str)
        if path.is_absolute():
            return path
            
        # Try relative to app dir first
        app_path = self.app_dir / path
        if app_path.exists():
            return app_path
            
        # Try relative to redactor config dir
        redactor_path = self.redactor_config_dir / path.name
        if redactor_path.exists():
            return redactor_path
            
        self.logger.error(f"Could not resolve path: {path_str}")
        return None

    def _load_single_config(self, config_name: str, filepath: Path) -> bool:
        """
        Load and validate a single configuration file.
        
        Args:
            config_name: Name of configuration to load
            filepath: Path to configuration file
            
        Returns:
            bool: Whether loading was successful
        """
        try:
            self.logger.debug(f"Loading config: {config_name} from {filepath}")
            
            config_data = self._safe_yaml_load(filepath)
            if config_data is None:
                return False

            if not self._validate_loaded_config(config_data, config_name):
                return False

            self.configs[config_name] = config_data
            self.loaded_paths[config_name] = filepath
            self.logger.info(f"Successfully loaded {config_name} configuration")
            return True

        except Exception as e:
            self.logger.error(f"Unexpected error loading {filepath}: {str(e)}")
            return False

    def get_config(self, config_name: str) -> Optional[Dict]:
        """Safely get a configuration by name."""
        return self.configs.get(config_name)

    def get_config_path(self, config_name: str) -> Optional[Path]:
        """Get the file path for a loaded configuration."""
        return self.loaded_paths.get(config_name)