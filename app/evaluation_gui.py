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
from evaluation.test_runner import TestRunner


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
from evaluation.test_runner import TestRunner

# [Previous imports and functions remain the same]

# ===== 6. TEST SUITE EXECUTION =====
def run_full_test_suite():
    """Run the full validation test suite and display results"""
    st.header("Full Test Suite Execution")
    
    try:
        # Initialize test runner
        test_runner = TestRunner("data/test_suite")
        
        # Load test cases
        test_cases = test_runner.load_test_cases()
        
        # Add a progress bar
        progress_bar = st.progress(0)
        
        # Capture output
        output_container = st.container()
        
        # Redirect stdout to capture print statements
        old_stdout = sys.stdout
        sys.stdout = captured_output = io.StringIO()
        
        try:
            # Run evaluation
            results = test_runner.run_evaluation(test_cases)
            
            # Restore stdout
            sys.stdout = old_stdout
            
            # Update progress bar
            progress_bar.progress(100)
            
            # Display summary results
            st.subheader("Test Run Summary")
            
            # Performance Metrics
            st.write("### Overall Performance Metrics")
            metrics_df = pd.DataFrame.from_dict(results.summary_metrics, orient='index', columns=['Value'])
            st.dataframe(metrics_df)
            
            # Type Performance Analysis
            st.write("### Performance by Entity Type")
            type_analysis = test_runner.analyze_type_performance(results.run_id)
            type_performance_data = []
            
            for entity_type, analysis in type_analysis.items():
                agg_metrics = analysis['aggregate_metrics']
                type_performance_data.append({
                    'Entity Type': entity_type,
                    'Precision': agg_metrics['precision'],
                    'Recall': agg_metrics['recall'],
                    'F1 Score': agg_metrics['f1_score'],
                    'Total Detected': analysis['total_detected']
                })
            
            type_performance_df = pd.DataFrame(type_performance_data)
            st.dataframe(type_performance_df)
            
            # Confidence Distribution
            st.write("### Confidence Score Distribution")
            confidence_analysis = test_runner.analyze_confidence_distribution(results.run_id)
            
            # Create a pie chart for confidence distribution
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
            
            # True Positives Confidence Distribution
            tp_labels = list(confidence_analysis['true_positives'].keys())
            tp_sizes = list(confidence_analysis['true_positives'].values())
            ax1.pie(tp_sizes, labels=tp_labels, autopct='%1.1f%%', title='True Positives')
            
            # False Positives Confidence Distribution
            fp_labels = list(confidence_analysis['false_positives'].keys())
            fp_sizes = list(confidence_analysis['false_positives'].values())
            ax2.pie(fp_sizes, labels=fp_labels, autopct='%1.1f%%', title='False Positives')
            
            st.pyplot(fig)
            
            # Raw output
            with st.expander("Detailed Output"):
                st.text(captured_output.getvalue())
            
        except Exception as e:
            # Restore stdout
            sys.stdout = old_stdout
            
            # Display error
            st.error(f"Error running test suite: {e}")
            st.exception(e)
        
    except FileNotFoundError as e:
        st.error(f"Test suite directory not found - {e}")
        st.error("Make sure you're running from the project root directory")
    except Exception as e:
        st.error(f"Error initializing test runner: {e}")

# ===== 7. VALIDATION TESTS =====
def run_validation_tests():
    """Run the validation tests and display results in a structured, visual format"""
    import sys
    import io
    import json
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import streamlit as st

    try:
        # Initialize test runner
        test_runner = TestRunner("data/test_suite")
        
        # Load test cases
        test_cases = test_runner.load_test_cases()
        
        # Run evaluation
        results = test_runner.run_evaluation(test_cases)
        
        # Debugging: Print raw results to understand structure
        st.write("Debug: Raw Summary Metrics")
        st.json(results.summary_metrics)
        
        # Overall Metrics Section
        st.header("Validation Test Results")
        
        # Overall Performance Metrics
        st.subheader("Overall Performance")
        overall_metrics = results.summary_metrics['overall']
        
        # Create columns for key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Precision", f"{overall_metrics.get('precision', 0):.2f}")
        with col2:
            st.metric("Recall", f"{overall_metrics.get('recall', 0):.2f}")
        with col3:
            st.metric("F1 Score", f"{overall_metrics.get('f1_score', 0):.2f}")
        with col4:
            st.metric("False Positives", f"{overall_metrics.get('false_positives', 0):.2f}")
        
        # Detailed Performance by Detection Type and Entity
        st.subheader("Detailed Performance Breakdown")
        
        # Collect performance data
        performance_data = []
        detection_types = ['presidio_detections', 'spacy_detections', 'ensemble_detections']
        
        for det_type in detection_types:
            type_metrics = results.summary_metrics['by_type'].get(det_type, {})
            for entity_type, metrics in type_metrics.items():
                performance_data.append({
                    'Detection Type': det_type,
                    'Entity Type': entity_type,
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1 Score': metrics.get('f1_score', 0),
                    'Total Detections': metrics.get('total_detections', 0)
                })
        
        # Create DataFrame and display
        df_performance = pd.DataFrame(performance_data)
        st.dataframe(df_performance)
        
        # Confidence Distribution Visualization
        st.subheader("Confidence Distribution")
        
        # Retrieve confidence distribution
        confidence_analysis = test_runner.analyze_confidence_distribution(results.run_id)
        
        # Prepare confidence data
        def prepare_confidence_data(data_dict):
            """Safely prepare confidence distribution data"""
            # Remove any NaN or None values
            filtered_data = {k: v for k, v in data_dict.items() if v is not None and v > 0}
            
            # If no data, return a default
            if not filtered_data:
                return {'No Data': 1}
            
            return filtered_data
        
        # Prepare true positives and false positives data
        tp_data = prepare_confidence_data(confidence_analysis.get('true_positives', {}))
        fp_data = prepare_confidence_data(confidence_analysis.get('false_positives', {}))
        
        # Visualization of confidence distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # True Positives Confidence
        tp_labels = list(tp_data.keys())
        tp_sizes = list(tp_data.values())
        ax1.pie(tp_sizes, labels=tp_labels, autopct='%1.1f%%')
        ax1.set_title('True Positives Confidence')
        
        # False Positives Confidence
        fp_labels = list(fp_data.keys())
        fp_sizes = list(fp_data.values())
        ax2.pie(fp_sizes, labels=fp_labels, autopct='%1.1f%%')
        ax2.set_title('False Positives Confidence')
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error running validation tests: {e}")
        st.exception(e)


# ===== 8. MAIN APPLICATION =====
def main():
    """Main Streamlit application"""
    st.title("Resume Redaction System - Evaluation Dashboard")
    
    # Initialize components
    config_loader = ConfigLoader()
    config_loader.load_all_configs()
    logger = RedactionLogger(name="evaluation", log_level="DEBUG", log_dir="logs")
    evaluator = StageOneEvaluator()

    # Load test manifest
    manifest_path = Path("data/test_suite/test_manifest.yaml")
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)
    test_cases = manifest['test_cases']

    # Sidebar Controls
    st.sidebar.header("Test Configuration")
    
    # Test Mode Selection
    test_mode = st.sidebar.radio(
        "Test Mode",
        ["Run Validation Tests", "Investigate Entity Type"]
    )

    if test_mode == "Run Validation Tests":
        if st.sidebar.button("Run Validation Suite"):
            run_validation_tests()
            
    else:  # Investigate Entity Type
        # Entity Type Selection
        entity_type = st.sidebar.selectbox(
            "Select Entity Type to Investigate",
            ["DATE_TIME", "INTERNET_REFERENCE"]  # Start with these two
        )
        
        # Test Case Selection
        test_id = st.sidebar.selectbox(
            "Select Test Case",
            [case['id'] for case in test_cases]
        )
        
        if st.sidebar.button("Investigate"):
            investigate_entity_detections(test_id, entity_type, evaluator)

if __name__ == "__main__":
    main()