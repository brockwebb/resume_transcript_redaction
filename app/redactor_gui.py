import yaml
from pathlib import Path
import streamlit as st
import os
from typing import Dict, Optional
from redactor_logic import RedactionProcessor

class ConfigLoader:
    """Handles loading and validation of all configuration files"""

    def __init__(self):
        self.config_files = {
            'main': 'config.yaml',
            'confidential': 'confidential_terms.yaml',
            'word_filters': 'custom_word_filters.yaml',
            'detection': 'detection_patterns.yaml'
        }
        self.configs: Dict[str, Optional[dict]] = {}
        self.load_status: Dict[str, bool] = {}

    def load_all_configs(self) -> bool:
        """
        Load all configuration files and validate their structure.
        Returns True if all critical configs loaded successfully.
        """
        for config_name, filename in self.config_files.items():
            success = self._load_single_config(config_name, filename)
            self.load_status[config_name] = success

            if not success and config_name in ['main', 'confidential']:  # Critical configs
                st.error(f"Failed to load critical configuration: {filename}")
                return False

        return True

    def _load_single_config(self, config_name: str, filename: str) -> bool:
        """Load and validate a single configuration file."""
        try:
            with open(filename, 'r') as file:
                config_data = yaml.safe_load(file)

            # Validate config structure based on type
            if config_name == 'main' and not self._validate_main_config(config_data):
                st.error(f"Invalid structure in {filename}")
                return False

            if config_name == 'confidential' and not self._validate_confidential_config(config_data):
                st.error(f"Invalid structure in {filename}")
                return False

            self.configs[config_name] = config_data
            return True

        except FileNotFoundError:
            st.warning(f"Configuration file not found: {filename}")
            self.configs[config_name] = None
            return config_name not in ['main', 'confidential']

        except yaml.YAMLError as e:
            st.error(f"Error parsing {filename}: {str(e)}")
            self.configs[config_name] = None
            return False

        except Exception as e:
            st.error(f"Unexpected error loading {filename}: {str(e)}")
            self.configs[config_name] = None
            return False

    def _validate_main_config(self, config: dict) -> bool:
        """Validate main configuration structure."""
        required_keys = ['input_folder', 'output_folder', 'model_directory', 'confidence_threshold']
        return all(key in config for key in required_keys)

    def _validate_confidential_config(self, config: dict) -> bool:
        """Validate confidential terms configuration structure."""
        required_keys = ['identity_terms', 'race_ethnicity_terms', 'protected_categories', 'custom_patterns']
        return all(key in config for key in required_keys)

    def get_config(self, config_name: str) -> Optional[dict]:
        """Safely get a configuration by name."""
        return self.configs.get(config_name)

    def get_load_status(self) -> Dict[str, bool]:
        """Get the loading status of all configurations."""
        return self.load_status


def display_gui():
    """Main GUI for the Resume Redaction Web App."""
    st.title("Resume Redaction Web App")

    # Initialize config loader
    config_loader = ConfigLoader()

    # Load all configurations
    if not config_loader.load_all_configs():
        st.error("Failed to load critical configurations. Please check the configuration files.")
        return

    # Display configuration status
    with st.expander("Configuration Status"):
        status = config_loader.get_load_status()
        for config_name, success in status.items():
            if success:
                st.success(f"{config_name}: Loaded successfully")
            else:
                st.error(f"{config_name}: Failed to load")

    # Get configurations
    main_config = config_loader.get_config('main')
    confidential_config = config_loader.get_config('confidential')
    word_filters = config_loader.get_config('word_filters')

    # Initialize processor with loaded configurations
    try:
        processor = RedactionProcessor(
            custom_patterns=confidential_config.get('custom_patterns', []),
            keep_words_path=config_loader.config_files['word_filters'],
            confidence_threshold=main_config.get('confidence_threshold', 0.75),
            model_directory=main_config.get('model_directory')
        )
    except AttributeError as e:
        st.error(f"Failed to initialize redaction processor due to missing attribute: {str(e)}")
        return
    except Exception as e:
        st.error(f"Failed to initialize redaction processor: {str(e)}")
        return

    # GUI Components
    st.header("Step 1: Configuration")
    col1, col2 = st.columns(2)

    with col1:
        input_dir = st.text_input("Input Directory", main_config.get('input_folder', ''))
        output_dir = st.text_input("Output Directory", main_config.get('output_folder', ''))

    with col2:
        mode = st.radio("Select Output Mode:", options=["Blackline", "Highlight"], index=0)
        redact_color = (0, 0, 0) if mode == "Blackline" else (1, 1, 0)
        redact_opacity = 1.0 if mode == "Blackline" else 0.5

    # Step 2: Custom Word Filters
    st.header("Step 2: Custom Word Filters")
    col1, col2 = st.columns(2)

    with col1:
        current_keep_words = ', '.join(word_filters.get('keep_words', []))
        manual_keep_words = st.text_area("Whitelist (Words to Keep)", 
                                         value=current_keep_words,
                                         placeholder="Enter words to keep (comma-separated)")
        if st.button("Update Whitelist"):
            new_keep_words = [word.strip() for word in manual_keep_words.split(",") if word.strip()]
            processor.keep_words.update(new_keep_words)
            st.success(f"Added {len(new_keep_words)} words to the whitelist.")

    with col2:
        current_discard_words = ', '.join(word_filters.get('discard_words', []))
        manual_discard_words = st.text_area("Blacklist (Words to Remove)", 
                                            value=current_discard_words,
                                            placeholder="Enter words to remove (comma-separated)")
        if st.button("Update Blacklist"):
            new_discard_words = [word.strip() for word in manual_discard_words.split(",") if word.strip()]
            processor.discard_words.update(new_discard_words)
            st.success(f"Added {len(new_discard_words)} words to the blacklist.")

    # Step 3: Start Redaction
    st.header("Step 3: Start Redaction")
    if st.button("Start Redaction"):
        if not os.path.exists(input_dir):
            st.error("Input directory does not exist.")
        elif not os.path.exists(output_dir):
            st.error("Output directory does not exist.")
        else:
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("No PDF files found in the input directory.")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_stats = {"processed": 0, "total_words": 0, "total_redacted": 0}

                for idx, pdf_file in enumerate(pdf_files, 1):
                    input_path = os.path.join(input_dir, pdf_file)
                    output_path = os.path.join(output_dir, f"confidential_{idx}.pdf")

                    try:
                        status_text.text(f"Processing file {idx} of {len(pdf_files)}: {pdf_file}")
                        stats = processor.process_pdf(
                            input_path=input_path,
                            output_path=output_path,
                            redact_color=redact_color,
                            redact_opacity=redact_opacity
                        )

                        total_stats["processed"] += 1
                        total_stats["total_words"] += stats["total_words"]
                        total_stats["total_redacted"] += stats["redacted_words"]

                        progress = idx / len(pdf_files)
                        progress_bar.progress(progress)

                    except Exception as e:
                        st.error(f"Error processing {pdf_file}: {str(e)}")

                status_text.empty()

                st.success("Redaction complete!")
                st.write(f"Processed {total_stats['processed']} files")
                st.write(f"Total words processed: {total_stats['total_words']}")
                st.write(f"Total words redacted: {total_stats['total_redacted']}")
                if total_stats['total_words'] > 0:
                    redaction_rate = (total_stats['total_redacted'] / total_stats['total_words'] * 100)
                    st.write(f"Overall redaction rate: {redaction_rate:.2f}%")

if __name__ == "__main__":
    display_gui()

