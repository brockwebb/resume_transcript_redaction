# app/redactor_gui.py

# Standard library imports
import sys
import os
import yaml
from pathlib import Path
from typing import List, Set

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

def resolve_paths(config_loader: ConfigLoader):
    """Resolve all relative paths in configuration to absolute paths."""
    main_config = config_loader.get_config('main')
    if not main_config:
        return {
            'input_dir': project_root / 'data/redact_input',
            'output_dir': project_root / 'data/redact_output',
        }

    project_root = Path(__file__).parent.parent
    
    paths = {
        'input_dir': project_root / main_config.get('input_folder', 'data/redact_input').replace('../', ''),
        'output_dir': project_root / main_config.get('output_folder', 'data/redact_output').replace('../', ''),
    }

    # Create directories if they don't exist
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)

    return paths

class KeepWordManager:
    """Manages custom keep words for the redaction session."""
    
    def __init__(self, processor: RedactionProcessor):
        """Initialize with a redaction processor."""
        self.processor = processor
        self.custom_keep_words = set()
        
    def add_keep_words(self, words: List[str]):
        """Add custom keep words that won't be redacted."""
        if not words:
            return
            
        # Store custom keep words locally
        self.custom_keep_words.update(words)
        
        # Add to processor's keep_words set
        if hasattr(self.processor, 'keep_words'):
            self.processor.keep_words.update(words)
            st.success(f"Added {len(words)} custom keep words")
        else:
            st.warning("Keep words not supported by this processor")
        
    def get_keep_words(self) -> Set[str]:
        """Get the current set of custom keep words."""
        return self.custom_keep_words

# Main GUI function
def main():
    st.title("Resume Redaction System")
    st.markdown("Automatically detect and redact personal identifiable information from resumes")

    try:
        # Initialize logger first
        logger = RedactionLogger(
            name="redaction_gui",
            log_level="INFO",
            log_dir="logs",
            console_output=True
        )
        
        # Initialize config loader
        config_loader = ConfigLoader()
        config_loader.load_all_configs()
        
        # Get available configs for display
        available_configs = config_loader._cached_configs.keys()
        
        # Resolve paths
        paths = resolve_paths(config_loader)

        # Initialize RedactionProcessor with proper error handling
        try:
            processor = RedactionProcessor(
                config_loader=config_loader,
                logger=logger
            )
            
            # Suppress excessive logging for missing configs
            if hasattr(processor, 'logger') and processor.logger:
                processor.logger.setLevel("WARNING")
                
            # Initialize the keep word manager
            keep_word_manager = KeepWordManager(processor)
            
        except Exception as e:
            st.error(f"Failed to initialize RedactionProcessor: {str(e)}")
            logger.error(f"RedactionProcessor initialization error: {str(e)}")
            return

        # Create sidebar for configuration information
        with st.sidebar:
            st.subheader("System Information")
            st.write(f"Available configurations: {', '.join(available_configs)}")
            
            # Display confidence threshold if available
            main_config = config_loader.get_config('main')
            if main_config and 'confidence_threshold' in main_config:
                st.write(f"Confidence threshold: {main_config['confidence_threshold']}")
                
            # Show entity types being detected
            entity_routing = config_loader.get_config('entity_routing')
            if entity_routing and 'routing' in entity_routing:
                all_entities = set()
                routing = entity_routing['routing']
                
                # Collect all entity types from all detectors
                for detector in ['presidio_primary', 'spacy_primary', 'ensemble_required']:
                    if detector in routing and 'entities' in routing[detector]:
                        all_entities.update(routing[detector]['entities'])
                
                st.write("### Entity Types Detected")
                st.write(", ".join(sorted(all_entities)))
                
            # Add system status indicators
            st.success("✓ Config Loader Ready")
            st.success("✓ RedactionProcessor Ready")

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
            
            # Add custom keep words option
            st.subheader("Custom Settings")
            custom_keep_words = st.text_area(
                "Keep Words (terms that should NOT be redacted, one per line)",
                height=100
            )
            
            if st.button("Add Keep Words"):
                if custom_keep_words:
                    words = [word.strip() for word in custom_keep_words.split("\n") if word.strip()]
                    keep_word_manager.add_keep_words(words)
            
            # Show current keep words if any exist
            current_keep_words = keep_word_manager.get_keep_words()
            if current_keep_words:
                st.info(f"Current custom keep words: {', '.join(current_keep_words)}")

        # Process files
        st.header("Step 2: Process Resumes")
        
        # File selection options
        use_file_selector = st.checkbox("Select specific files", value=False)
        
        if use_file_selector and os.path.exists(input_dir):
            pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
            if pdf_files:
                selected_files = st.multiselect("Select files to process:", pdf_files, default=pdf_files)
            else:
                selected_files = []
                st.warning("No PDF files found in input directory")
        else:
            selected_files = None  # Process all files
            
        if st.button("Start Redaction"):
            if not os.path.exists(input_dir):
                st.error("Input directory does not exist.")
                return
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
                st.info(f"Created output directory: {output_dir}")

            # Get list of PDF files to process
            if selected_files is not None:
                pdf_files = selected_files
            else:
                pdf_files = [f for f in os.listdir(input_dir) if f.endswith('.pdf')]
                
            if not pdf_files:
                st.warning("No PDF files found in the input directory.")
                return

            # Create filename mapping
            filename_mapping = {}
            
            # Process files with progress bar
            progress_bar = st.progress(0)
            st.subheader("Processing Results:")
            results_container = st.container()
            
            with results_container:
                for idx, pdf_file in enumerate(pdf_files, start=1):
                    input_path = os.path.join(input_dir, pdf_file)
                    anonymous_filename = generate_anonymous_filename(idx)
                    output_path = os.path.join(output_dir, anonymous_filename)
                    
                    # Add to mapping
                    filename_mapping[anonymous_filename] = pdf_file
                    
                    st.write(f"Processing {pdf_file}...")
                    
                    try:
                        # Set processor log level to WARNING to suppress debug messages
                        if hasattr(processor, 'logger') and processor.logger:
                            prev_level = processor.logger.level
                            processor.logger.setLevel("WARNING")
                            
                        stats = processor.process_pdf(
                            input_path=input_path,
                            output_path=output_path,
                            redact_color=redact_color,
                            redact_opacity=redact_opacity,
                        )
                        
                        # Restore previous log level
                        if hasattr(processor, 'logger') and processor.logger:
                            processor.logger.setLevel(prev_level)
                            
                        st.write(f"✅ Processed {pdf_file} → {anonymous_filename}")
                        if stats:
                            st.write(f"   - Total words: {stats.get('total_words', 0)}")
                            st.write(f"   - Redacted words: {stats.get('redacted_words', 0)}")
                            redaction_rate = stats.get('redacted_words', 0)/max(stats.get('total_words', 1), 1)*100
                            st.write(f"   - Redaction rate: {redaction_rate:.1f}%")
                            
                        # Update progress bar
                        progress_bar.progress(idx / len(pdf_files))
                    except Exception as e:
                        st.error(f"Error processing {pdf_file}: {str(e)}")
                        logger.error(f"Failed to process {pdf_file}: {str(e)}")

            # Save the filename mapping
            if save_filename_mapping(filename_mapping, output_dir):
                st.success(f"Redaction complete! Processed {len(pdf_files)} files.")
                st.info("Filename mapping saved to 'filename_mapping.yaml' in the output directory.")
                
                # Generate download links for redacted files
                st.header("Step 3: Download Redacted Files")
                st.write("Click on the links below to download individual redacted files:")
                
                for anon_filename, orig_filename in filename_mapping.items():
                    output_path = os.path.join(output_dir, anon_filename)
                    if os.path.exists(output_path):
                        with open(output_path, "rb") as f:
                            file_bytes = f.read()
                            st.download_button(
                                label=f"Download {anon_filename} (was {orig_filename})",
                                data=file_bytes,
                                file_name=anon_filename,
                                mime="application/pdf"
                            )
            else:
                st.warning("Redaction complete but failed to save filename mapping.")

    except Exception as e:
        st.error(f"Application error: {str(e)}")
        if 'logger' in locals():
            logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()
