import yaml
from pathlib import Path
import streamlit as st
import os
from redactor_logic import RedactionProcessor

# Load configuration from config.yaml
CONFIG_PATH = Path("config.yaml")

def load_config():
    """Load configuration from config.yaml."""
    try:
        with open(CONFIG_PATH, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading configuration: {e}")
        return {}

config = load_config()

CONFIDENTIAL_TERMS_PATH = Path("confidential_terms.yaml")

def load_confidential_terms():
    """Load confidential terms from YAML."""
    try:
        with open(CONFIDENTIAL_TERMS_PATH, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        st.error(f"Error loading confidential terms: {e}")
        return {}

def display_gui():
    """Main GUI for the Resume Redaction App."""
    # Load configuration
    input_dir = config.get("input_folder", "data/redact_input")
    output_dir = config.get("output_folder", "data/redact_output")
    model_directory = config.get("model_directory", "models/jobbert_model")
    test_mode = config.get("test_mode", False)
    confidence_threshold = config.get("confidence_threshold", 0.75)
    pii_types = config.get("pii_types", [])

    # Load confidential terms
    confidential_data = load_confidential_terms()
    custom_patterns = confidential_data.get("custom_patterns", [])

    # Initialize the processor with custom patterns and word filters
    processor = RedactionProcessor(
        custom_patterns=custom_patterns,
        keep_words_path="custom_word_filters.yaml",
        redact_color=(0.8, 0.8, 0.8) if test_mode else (0, 0, 0),
        redact_opacity=0.3 if test_mode else 1.0,
        pii_types=pii_types,
        model_directory=model_directory
    )

    # Add all categories of terms to custom terms
    all_terms = confidential_data.get("identity_terms", []) + \
                confidential_data.get("race_ethnicity_terms", []) + \
                confidential_data.get("protected_categories", [])
    processor.add_custom_terms(all_terms)

    st.title("Resume Redaction Web App")
    st.write("Use this tool to redact sensitive information from resumes.")

    # Step 1: Input/Output Directories and Output Mode
    st.header("Step 1: Configuration")
    col1, col2 = st.columns(2)

    with col1:
        input_dir = st.text_input("Input Directory", input_dir)
        output_dir = st.text_input("Output Directory", output_dir)

    with col2:
        mode = st.radio("Select Output Mode:", options=["Blackline", "Highlight"], index=0)
        processor.redact_color = (0.8, 0.8, 0.8) if mode == "Highlight" else (0, 0, 0)
        processor.redact_opacity = 0.3 if mode == "Highlight" else 1.0
        processor.highlight_only = mode == "Highlight"

    # Step 2: Custom Word Filters
    st.header("Step 2: Custom Word Filters")
    col1, col2 = st.columns(2)

    with col1:
        manual_keep_words = st.text_area("Whitelist (Words to Keep)", placeholder="Enter words to keep (comma-separated)")
        if st.button("Update Whitelist"):
            new_keep_words = [word.strip() for word in manual_keep_words.split(",") if word.strip()]
            processor.keep_words.extend(new_keep_words)
            st.success(f"Added {len(new_keep_words)} words to the whitelist.")

    with col2:
        manual_discard_words = st.text_area("Blacklist (Words to Remove)", placeholder="Enter words to remove (comma-separated)")
        if st.button("Update Blacklist"):
            new_discard_words = [word.strip() for word in manual_discard_words.split(",") if word.strip()]
            processor.discard_words.extend(new_discard_words)
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
                # Initialize progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                total_stats = {"processed": 0, "total_words": 0, "total_redacted": 0}

                for idx, pdf_file in enumerate(pdf_files, 1):
                    input_path = os.path.join(input_dir, pdf_file)
                    output_path = os.path.join(output_dir, f"confidential_{idx}.pdf")

                    try:
                        # Update status
                        status_text.text(f"Processing file {idx} of {len(pdf_files)}: {pdf_file}")

                        # Process the file with highlight mode if enabled
                        stats = processor.process_pdf(input_path, output_path, highlight_only=processor.highlight_only)

                        # Update totals
                        total_stats["processed"] += 1
                        total_stats["total_words"] += stats["total_words"]
                        total_stats["total_redacted"] += stats["redacted_words"]

                        # Update progress bar
                        progress = idx / len(pdf_files)
                        progress_bar.progress(progress)

                    except Exception as e:
                        st.error(f"Error processing {pdf_file}: {str(e)}")

                # Clear status text
                status_text.empty()

                # Show final statistics
                st.success("Redaction complete!")
                st.write(f"Processed {total_stats['processed']} files")
                st.write(f"Total words processed: {total_stats['total_words']}")
                st.write(f"Total words redacted: {total_stats['total_redacted']}")
                if total_stats['total_words'] > 0:
                    redaction_rate = (total_stats['total_redacted'] / total_stats['total_words'] * 100)
                    st.write(f"Overall redaction rate: {redaction_rate:.2f}%")

if __name__ == "__main__":
    display_gui()
