# app/redactor_gui.py

# Standard library imports
import sys
import os
import yaml
from pathlib import Path

# Resolve project root path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Third-party imports
import streamlit as st

# Local imports
from redactor.redactor_logic import RedactionProcessor
from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger

# Set page config
st.set_page_config(
    page_title="Resume Redaction System",
    page_icon="📄",
    layout="wide"
)

def generate_anonymous_filename(index: int) -> str:
    """Generate anonymous filename in format 'confidential_N.pdf'."""
    return f"confidential_{index:03d}.pdf"

def save_filename_mapping(mapping_dict: dict, output_dir: str):
    """Save mapping of original filenames to anonymous filenames."""
    mapping_path = Path(output_dir) / "filename_mapping.yaml"
    try:
        with open(mapping_path, 'w') as f:
            yaml.dump(mapping_dict, f, default_flow_style=False)
        return True
    except Exception as e:
        st.error(f"Error saving filename mapping: {e}")
        return False

def resolve_paths(config_loader):
    """Resolve all relative paths in configuration to absolute paths."""
    main_config = config_loader.get_config('main')
    if not main_config:
        raise ValueError("Failed to load main configuration")

    project_root = Path(__file__).parent.parent
    
    paths = {
        'input_dir': project_root / main_config.get('input_folder', 'data/redact_input').replace('../', ''),
        'output_dir': project_root / main_config.get('output_folder', 'data/redact_output').replace('../', ''),
    }

    config_paths = {
        'detection_patterns': project_root / main_config.get('redaction_patterns_path', 'redactor/config/detection_patterns.yaml').replace('../', ''),
        'confidential_terms': project_root / main_config.get('confidential_terms_path', 'redactor/config/confidential_terms.yaml').replace('../', ''),
        'word_filters': project_root / main_config.get('word_filters_path', 'redactor/config/custom_word_filters.yaml').replace('../', ''),
        'entity_routing': project_root / main_config.get('entity_routing_path', 'redactor/config/entity_routing.yaml').replace('../', ''),
        'core_patterns': project_root / main_config.get('core_patterns_path', 'redactor/config/core_patterns.yaml').replace('../', '')
    }

    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths, config_paths

# Main GUI function
def main():
    st.title("Resume Redaction Web App")

    try:
        # Initialize config loader and logger first
        config_loader = ConfigLoader()
        config_loader.load_all_configs()
        logger = config_loader.logger

        # Resolve paths
        paths, config_paths = resolve_paths(config_loader)

        # Get main config
        main_config = config_loader.get_config('main')
        if not main_config:
            st.error("Failed to load main configuration")
            logger.error("Failed to load main configuration")
            return

        # Load core patterns
        patterns = config_loader.get_config('core_patterns')
        if not patterns:
            st.error("Failed to load core patterns configuration")
            logger.error("Failed to load core patterns configuration")
            return

        # Initialize RedactionProcessor
        try:
            processor = RedactionProcessor(
                custom_patterns=patterns,
                keep_words_path=str(config_paths['word_filters']),
                confidence_threshold=main_config.get('confidence_threshold', 0.8),
                trigger_words_path=str(config_paths['word_filters']),
                logger=logger
            )
        except Exception as e:
            st.error(f"Failed to initialize RedactionProcessor: {str(e)}")
            logger.error(f"RedactionProcessor initialization error: {str(e)}")
            return

        # Create main layout
        st.header("Step 1: Configuration")
        col1, col2 = st.columns(2)

        with col1:
            input_dir = st.text_input("Input Directory", str(paths['input_dir']))
            output_dir = st.text_input("Output Directory", str(paths['output_dir']))

        with col2:
            mode = st.radio("Select Output Mode:", options=["Blackline", "Highlight"], index=0)
            redact_color = (0, 0, 0) if mode == "Blackline" else (1, 1, 0)
            redact_opacity = 1.0 if mode == "Blackline" else 0.5

        # Process files
        st.header("Step 2: Process Resumes")
        if st.button("Start Redaction"):
            if not os.path.exists(input_dir):
                st.error("Input directory does not exist.")
                return
            if not os.path.exists(output_dir):
                st.error("Output directory does not exist.")
                return

            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("No PDF files found in the input directory.")
                return

            # Create filename mapping
            filename_mapping = {}
            
            # Process files with progress bar
            progress_bar = st.progress(0)
            for idx, pdf_file in enumerate(pdf_files, start=1):
                input_path = os.path.join(input_dir, pdf_file)
                anonymous_filename = generate_anonymous_filename(idx)
                output_path = os.path.join(output_dir, anonymous_filename)
                
                # Add to mapping
                filename_mapping[anonymous_filename] = pdf_file

                try:
                    stats = processor.process_pdf(
                        input_path=input_path,
                        output_path=output_path,
                        redact_color=redact_color,
                        redact_opacity=redact_opacity,
                    )
                    st.write(f"Processed {pdf_file} → {anonymous_filename}")
                    st.write(f"Statistics: {stats}")
                    progress_bar.progress(idx / len(pdf_files))
                except Exception as e:
                    st.error(f"Error processing {pdf_file}: {str(e)}")
                    logger.error(f"Failed to process {pdf_file}: {str(e)}")

            # Save the filename mapping
            if save_filename_mapping(filename_mapping, output_dir):
                st.success(f"Redaction complete! Processed {len(pdf_files)} files.")
                st.info("Filename mapping saved to 'filename_mapping.yaml' in the output directory.")
            else:
                st.warning("Redaction complete but failed to save filename mapping.")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()