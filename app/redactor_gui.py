# app/redactor_gui.py
import streamlit as st
import os
from typing import Dict, Optional
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Use absolute imports
from redactor.redactor_logic import RedactionProcessor
from utils.config_loader import ConfigLoader  
from utils.logger import RedactionLogger


def resolve_paths(config_loader):
    """Resolve all relative paths in configuration to absolute paths."""
    main_config = config_loader.get_config('main')
    project_root = Path(__file__).parent.parent

    # Resolve core directories
    paths = {
        'input_dir': project_root / main_config.get('input_folder').replace('../', ''),
        'output_dir': project_root / main_config.get('output_folder').replace('../', ''),
        'models_dir': project_root / main_config.get('models_dir').replace('../', '')
    }

    # Resolve config paths
    config_paths = {
        'redaction_patterns': project_root / main_config.get('redaction_patterns_path').replace('../', ''),
        'confidential_terms': project_root / main_config.get('confidential_terms_path').replace('../', ''),
        'word_filters': project_root / main_config.get('word_filters_path').replace('../', '')
    }

    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths, config_paths

def display_gui():
    """Main GUI for the Resume Redaction Web App."""
    st.title("Resume Redaction Web App")

    # Initialize config loader
    config_loader = ConfigLoader()
    logger = config_loader.logger

    # Load configurations and resolve paths
    if not config_loader.load_all_configs():
        st.error("Failed to load critical configurations. Please check the configuration files.")
        return

    # Resolve paths
    try:
        paths, config_paths = resolve_paths(config_loader)
    except Exception as e:
        st.error(f"Failed to resolve paths: {str(e)}")
        logger.error(f"Path resolution failed: {str(e)}")
        return

    # Load word filters configuration - ADD THIS LINE HERE
    word_filters = config_loader.get_config('word_filters')

    # Display configuration status
    with st.expander("Configuration Status"):
        st.write("Directories:")
        for name, path in paths.items():
            if path.exists():
                st.success(f"{name}: {path}")
            else:
                st.error(f"{name}: {path} (not found)")

        st.write("Configuration Files:")
        for name, path in config_paths.items():
            if path.exists():
                st.success(f"{name}: {path}")
            else:
                st.error(f"{name}: {path} (not found)")

    # Initialize processor with resolved paths
    try:
        processor = RedactionProcessor(
            custom_patterns=config_loader.get_config('detection'),  # Remove the .get('custom_patterns', [])
            keep_words_path=str(config_paths['word_filters']),
            confidence_threshold=config_loader.get_config('main').get('confidence_threshold', 0.75),
            detection_patterns_path=str(config_paths['redaction_patterns']),
            trigger_words_path=str(config_paths['word_filters']),
            logger=logger
        )
    except Exception as e:
        st.error(f"Failed to initialize redaction processor: {str(e)}")
        logger.error(f"Processor initialization failed: {str(e)}")
        return

    # GUI Components
    st.header("Step 1: Configuration")
    col1, col2 = st.columns(2)

    with col1:
        input_dir = st.text_input("Input Directory", str(paths['input_dir']))
        output_dir = st.text_input("Output Directory", str(paths['output_dir']))

    with col2:
        mode = st.radio("Select Output Mode:", options=["Blackline", "Highlight"], index=0)
        redact_color = (0, 0, 0) if mode == "Blackline" else (1, 1, 0)
        redact_opacity = 1.0 if mode == "Blackline" else 0.5

    # Step 2: Custom Word Filters
    st.header("Step 2: Custom Word Filters")
    col1, col2 = st.columns(2)

    with col1:
        # Safely get keep_words with fallback to empty list if None
        current_keep_words = ', '.join(word_filters.get('keep_words', []) if word_filters else [])
        manual_keep_words = st.text_area("Whitelist (Words to Keep)", 
                                       value=current_keep_words,
                                       placeholder="Enter words to keep (comma-separated)")
        if st.button("Update Whitelist"):
            new_keep_words = [word.strip() for word in manual_keep_words.split(",") if word.strip()]
            processor.keep_words.update(new_keep_words)
            logger.info(f"Added {len(new_keep_words)} words to the whitelist")
            st.success(f"Added {len(new_keep_words)} words to the whitelist.")

    with col2:
        # Safely get discard_words with fallback to empty list if None
        current_discard_words = ', '.join(word_filters.get('discard_words', []) if word_filters else [])
        manual_discard_words = st.text_area("Blacklist (Words to Remove)", 
                                          value=current_discard_words,
                                          placeholder="Enter words to remove (comma-separated)")
        if st.button("Update Blacklist"):
            new_discard_words = [word.strip() for word in manual_discard_words.split(",") if word.strip()]
            processor.discard_words.update(new_discard_words)
            logger.info(f"Added {len(new_discard_words)} words to the blacklist")
            st.success(f"Added {len(new_discard_words)} words to the blacklist.")
    # Step 3: Start Redaction
    st.header("Step 3: Start Redaction")
    if st.button("Start Redaction"):
        if not os.path.exists(input_dir):
            st.error("Input directory does not exist.")
            logger.error(f"Input directory does not exist: {input_dir}")
        elif not os.path.exists(output_dir):
            st.error("Output directory does not exist.")
            logger.error(f"Output directory does not exist: {output_dir}")
        else:
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            if not pdf_files:
                st.warning("No PDF files found in the input directory.")
                logger.warning(f"No PDF files found in: {input_dir}")
            else:
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_stats = {"processed": 0, "total_words": 0, "total_redacted": 0}

                for idx, pdf_file in enumerate(pdf_files, 1):
                    input_path = os.path.join(input_dir, pdf_file)
                    output_path = os.path.join(output_dir, f"confidential_{idx}.pdf")

                    try:
                        status_text.text(f"Processing file {idx} of {len(pdf_files)}: {pdf_file}")
                        logger.info(f"Processing file {idx}/{len(pdf_files)}: {pdf_file}")
                        
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
                        logger.info(f"Completed {pdf_file} - Words: {stats['total_words']}, Redacted: {stats['redacted_words']}")

                    except Exception as e:
                        st.error(f"Error processing {pdf_file}: {str(e)}")
                        logger.error(f"Error processing {pdf_file}: {str(e)}")

                status_text.empty()

                st.success("Redaction complete!")
                st.write(f"Processed {total_stats['processed']} files")
                st.write(f"Total words processed: {total_stats['total_words']}")
                st.write(f"Total words redacted: {total_stats['total_redacted']}")
                if total_stats['total_words'] > 0:
                    redaction_rate = (total_stats['total_redacted'] / total_stats['total_words'] * 100)
                    st.write(f"Overall redaction rate: {redaction_rate:.2f}%")
                    logger.info(f"Processing complete - Rate: {redaction_rate:.2f}%")

if __name__ == "__main__":
    display_gui()