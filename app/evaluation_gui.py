# app/evaluation_gui.py

# ===== 1. IMPORTS =====
import logging
from pathlib import Path
import io
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from evaluation.evaluate import StageOneEvaluator

# ===== 2. INITIALIZATION AND CACHING =====
@st.cache_resource
def init_components():
    """Initialize core components with caching"""
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    logger = RedactionLogger(name="evaluation", log_level="DEBUG", log_dir="logs")
    evaluator = StageOneEvaluator()
    return config_loader, logger, evaluator

@st.cache_data
def load_supported_entity_types(config_path):
    """Load supported entity types from routing configuration"""
    try:
        with open(config_path, 'r') as f:
            routing_config = yaml.safe_load(f)
        
        entity_types = set()
        for category in ['presidio_primary', 'spacy_primary', 'ensemble_required']:
            entity_types.update(
                routing_config['routing'].get(category, {}).get('entities', [])
            )
        return sorted(list(entity_types))
    except Exception as e:
        st.error(f"Error loading entity types: {e}")
        return []

@st.cache_data
def load_test_manifest():
    """Load and cache test manifest"""
    try:
        manifest_path = Path("data/test_suite/test_manifest.yaml")
        with open(manifest_path) as f:
            return yaml.safe_load(f)
    except Exception as e:
        st.error(f"Error loading test manifest: {e}")
        return {"test_cases": []}

# ===== 3. METRICS AND ANALYSIS =====
def calculate_metrics(truth_entities, detected_entities):
    """Calculate precision, recall, and F1 score"""
    matched_entities = [
        (t, d) for t in truth_entities 
        for d in detected_entities 
        if t.text.strip() == d.text.strip()
    ]
    
    precision = len(matched_entities) / len(detected_entities) if detected_entities else 0
    recall = len(matched_entities) / len(truth_entities) if truth_entities else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'matched_count': len(matched_entities),
        'total_truth': len(truth_entities),
        'total_detected': len(detected_entities)
    }

def display_metrics(metrics):
    """Display metrics in Streamlit"""
    st.subheader("Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Ground Truth", metrics['total_truth'])
    with col2:
        st.metric("Detected", metrics['total_detected'])
    with col3:
        st.metric("Matched", metrics['matched_count'])
    with col4:
        st.metric("F1 Score", f"{metrics['f1_score']:.2%}")

def display_confidence_distribution(detected_entities):
    """Display confidence score distribution"""
    st.subheader("Confidence Distribution")
    confidence_data = [d.confidence for d in detected_entities]
    
    if confidence_data:
        fig, ax = plt.subplots()
        ax.hist(confidence_data, bins=10, edgecolor='black')
        ax.set_title("Confidence Distribution")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Count")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("No entities detected for confidence distribution")

# ===== 4. DETAILED ANALYSIS =====
def analyze_detections(truth_entities, detected_entities, text):
    """Analyze detection matches and mismatches"""
    # Match entities
    matched = []
    position_mismatches = []
    missed = list(truth_entities)
    extra = list(detected_entities)
    
    for truth in truth_entities:
        # Get correct attribute names for truth entity
        truth_start = getattr(truth, 'start_char', getattr(truth, 'start', 0))
        truth_end = getattr(truth, 'end_char', getattr(truth, 'end', 0))
        
        for detected in detected_entities:
            # Get correct attribute names for detected entity
            detected_start = getattr(detected, 'start_char', getattr(detected, 'start', 0))
            detected_end = getattr(detected, 'end_char', getattr(detected, 'end', 0))
            
            if truth.text.strip() == detected.text.strip():
                # Text matches, check position
                if truth_start == detected_start and truth_end == detected_end:
                    matched.append((truth, detected))
                    if truth in missed: missed.remove(truth)
                    if detected in extra: extra.remove(detected)
                else:
                    position_mismatches.append((truth, detected))
                    if truth in missed: missed.remove(truth)
                    if detected in extra: extra.remove(detected)
                    
    return {
        'matched': matched,
        'position_mismatches': position_mismatches,
        'missed': missed,
        'extra': extra
    }

# Also update display function to handle both attribute names
def display_detailed_breakdown(analysis_results, text):
    """Display detailed breakdown of matches and mismatches"""
    st.subheader("Detailed Analysis")
    
    # Helper function to get position attributes
    def get_positions(entity):
        start = getattr(entity, 'start_char', getattr(entity, 'start', 0))
        end = getattr(entity, 'end_char', getattr(entity, 'end', 0))
        return start, end
    
    # Matched Entities
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Exact Matches")
        if analysis_results['matched']:
            for truth, detected in analysis_results['matched']:
                truth_start, truth_end = get_positions(truth)
                st.success(
                    f"Text: {truth.text}\n"
                    f"Position: {truth_start}-{truth_end}\n"
                    f"Confidence: {detected.confidence:.2f}"
                )
        else:
            st.info("No exact matches")
            
    with col2:
        st.write("### Position Mismatches")
        if analysis_results['position_mismatches']:
            for truth, detected in analysis_results['position_mismatches']:
                truth_start, truth_end = get_positions(truth)
                detected_start, detected_end = get_positions(detected)
                st.warning(
                    f"Text: {truth.text}\n"
                    f"Truth: {truth_start}-{truth_end}\n"
                    f"Detected: {detected_start}-{detected_end}\n"
                    f"Confidence: {detected.confidence:.2f}"
                )
        else:
            st.info("No position mismatches")
    
    # Missed Entities
    st.write("### Missed Entities")
    if analysis_results['missed']:
        for entity in analysis_results['missed']:
            start, end = get_positions(entity)
            context_start = max(0, start - 20)
            context_end = min(len(text), end + 20)
            context = text[context_start:context_end]
            
            st.error(
                f"Text: {entity.text}\n"
                f"Position: {start}-{end}\n"
                f"Context: ...{context}..."
            )
    else:
        st.info("No missed entities")

# ===== 5. MAIN ANALYSIS AND GUI =====
def analyze_entity_type(test_id: str, entity_type: str, evaluator: StageOneEvaluator):
    """Main entity analysis function with enhanced date debugging"""
    try:
        # Load document and ground truth
        text = evaluator._load_test_document(test_id)
        truth = evaluator._load_ground_truth(test_id)
        
        # Filter for entity type
        truth_entities = [t for t in truth if t.entity_type == entity_type]
        
        st.write("### Initial Ground Truth")
        for t in truth_entities:
            st.write(f"""
            Ground Truth Date:
            - Text: {t.text}
            - Start: {getattr(t, 'start_char', getattr(t, 'start', 'N/A'))}
            - End: {getattr(t, 'end_char', getattr(t, 'end', 'N/A'))}
            """)
            
        # Show raw detector results
        st.write("### Raw Detector Results")
        
        # Get Presidio results
        presidio_dates = [e for e in evaluator.ensemble.presidio_detector.detect_entities(text) 
                         if e.entity_type == entity_type]
        
        st.write("Presidio Detections:")
        for d in presidio_dates:
            st.write(f"""
            - Text: {d.text}
            - Start: {d.start}
            - End: {d.end}
            - Confidence: {d.confidence}
            """)
            
        # Get spaCy results
        spacy_dates = [e for e in evaluator.ensemble.spacy_detector.detect_entities(text)
                      if e.entity_type == entity_type]
        
        st.write("spaCy Detections:")
        for d in spacy_dates:
            st.write(f"""
            - Text: {d.text}
            - Start: {d.start}
            - End: {d.end}
            - Confidence: {d.confidence}
            """)
            
        # Get final ensemble results
        st.write("### Ensemble Final Results")
        detected = evaluator.ensemble.detect_entities(text)
        detected_entities = [d for d in detected if d.entity_type == entity_type]
        
        if detected_entities:
            for d in detected_entities:
                st.write(f"""
                Final Detection:
                - Text: {d.text}
                - Start: {d.start}
                - End: {d.end}
                - Confidence: {d.confidence}
                - Source: {d.source}
                - Metadata: {d.metadata}
                """)
        else:
            st.warning("No dates passed ensemble validation")
            
        # Show validation settings
        st.write("### Validation Settings")
        routing_config = evaluator.config_loader.get_config("entity_routing")
        if routing_config:
            date_settings = routing_config["routing"]["ensemble_required"]["confidence_thresholds"]["DATE_TIME"]
            st.write(f"""
            Date Detection Settings:
            - Minimum Combined Confidence: {date_settings['minimum_combined']}
            - Presidio Weight: {date_settings['presidio_weight']}
            - spaCy Weight: {date_settings['spacy_weight']}
            """)
            
        # Calculate and display metrics
        metrics = calculate_metrics(truth_entities, detected_entities)
        display_metrics(metrics)
        
        # Detailed analysis
        analysis_results = analyze_detections(truth_entities, detected_entities, text)
        display_detailed_breakdown(analysis_results, text)
        
        # Show confidence distribution
        display_confidence_distribution(detected_entities)
        
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        raise


def main():
    """Main Streamlit application"""
    st.title("Resume Redaction System - Evaluation Dashboard")
    
    # Initialize components (cached)
    config_loader, logger, evaluator = init_components()
    
    # Load configurations (cached)
    entity_types = load_supported_entity_types(
        Path(__file__).parent.parent / 'redactor/config/entity_routing.yaml'
    )
    manifest = load_test_manifest()
    test_cases = manifest.get('test_cases', [])

    # Sidebar Controls
    st.sidebar.header("Test Configuration")
    
    test_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Run All Tests", "Investigate Entity Type"]
    )

    if test_mode == "Run All Tests":
        if st.sidebar.button("Run Full Test Suite"):
            st.info("Running full test suite...")
            # TODO: Implement full test suite execution
            
    else:  # Investigate Entity Type
        entity_type = st.sidebar.selectbox(
            "Select Entity Type",
            entity_types
        )
        
        test_id = st.sidebar.selectbox(
            "Select Test Case",
            [case['id'] for case in test_cases]
        )
        
        if st.sidebar.button("Analyze"):
            analyze_entity_type(test_id, entity_type, evaluator)

if __name__ == "__main__":
    main()