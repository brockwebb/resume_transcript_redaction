# app/evaluation_gui.py

import streamlit as st
import yaml
from pathlib import Path
import pandas as pd
from datetime import datetime

from app.utils.config_loader import ConfigLoader
from app.utils.logger import RedactionLogger
from evaluation.evaluate import StageOneEvaluator

def investigate_entity_detections(test_id: str, entity_type: str, evaluator: StageOneEvaluator):
    """Detailed investigation of entity detection mismatches"""
    st.header(f"{entity_type} Detection Investigation: {test_id}")
    
    # Load annotation file to get location details
    annotation_path = Path(f"data/test_suite/annotations/{test_id}.yaml")
    with open(annotation_path) as f:
        annotations = yaml.safe_load(f)
    
    # Get ground truth entities
    truth = evaluator._load_ground_truth(test_id)
    truth_entities = [t for t in truth if t.entity_type == entity_type]
    
    # Get document text and detected entities
    text = evaluator._load_test_document(test_id)
    detected = evaluator.ensemble.detect_entities(text)
    detected_entities = [d for d in detected if d.entity_type == entity_type]
    
    # Display counts
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Ground Truth Count", len(truth_entities))
    with col2:
        st.metric("Detected Count", len(detected_entities))
    
    # Show side-by-side comparison
    st.subheader("Ground Truth vs Detected")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Ground Truth:")
        truth_df = pd.DataFrame([{
            "Text": t.text,
            "Start": next(
                ann['location']['start_char'] 
                for ann in annotations['entities'] 
                if ann['text'] == t.text and ann['type'] == t.entity_type
            ),
            "End": next(
                ann['location']['end_char'] 
                for ann in annotations['entities'] 
                if ann['text'] == t.text and ann['type'] == t.entity_type
            ),
            "Context": f"...{text[max(0, next(ann['location']['start_char'] for ann in annotations['entities'] if ann['text'] == t.text and ann['type'] == t.entity_type)-20):min(len(text), next(ann['location']['end_char'] for ann in annotations['entities'] if ann['text'] == t.text and ann['type'] == t.entity_type)+20)]}..."
        } for t in truth_entities])
        st.dataframe(truth_df)
    
    with col2:
        st.write("Detected:")
        detected_df = pd.DataFrame([{
            "Text": d.text,
            "Start": d.start,
            "End": d.end,
            "Confidence": f"{d.confidence:.2f}",
            "Context": f"...{text[max(0, d.start-20):min(len(text), d.end+20)]}..."
        } for d in detected_entities])
        st.dataframe(detected_df)
    
    # Analyze mismatches
    st.subheader("Mismatch Analysis")
    
    # Text matches ignoring position
    text_matches = []
    position_mismatches = []
    missed_entities = []
    
    for truth_entity in truth_entities:
        # Find the corresponding annotation
        truth_annotation = next(
            (ann for ann in annotations['entities'] 
             if ann['text'] == truth_entity.text and ann['type'] == truth_entity.entity_type), 
            None
        )
        
        if truth_annotation:
            truth_start = truth_annotation['location']['start_char']
            truth_end = truth_annotation['location']['end_char']
        
            found = False
            for detected_entity in detected_entities:
                if truth_entity.text.strip() == detected_entity.text.strip():
                    found = True
                    if (truth_start != detected_entity.start or 
                        truth_end != detected_entity.end):
                        position_mismatches.append((truth_entity, detected_entity, truth_start, truth_end))
                    else:
                        text_matches.append((truth_entity, detected_entity))
                    break
            if not found:
                missed_entities.append(truth_entity)
    
    # Show matches and mismatches
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"💚 Exact Matches: {len(text_matches)}")
        for truth, detected in text_matches:
            # Find the annotation for this truth entity
            truth_annotation = next(
                ann['location'] 
                for ann in annotations['entities'] 
                if ann['text'] == truth.text and ann['type'] == truth.entity_type
            )
            st.success(
                f"Text: {truth.text}\n"
                f"Position: {truth_annotation['start_char']}-{truth_annotation['end_char']}\n"
                f"Confidence: {detected.confidence:.2f}"
            )
            
    with col2:
        st.write(f"🟡 Position Mismatches: {len(position_mismatches)}")
        for truth, detected, truth_start, truth_end in position_mismatches:
            st.warning(
                f"Text: {truth.text}\n"
                f"Truth pos: {truth_start}-{truth_end}\n"
                f"Detected pos: {detected.start}-{detected.end}\n"
                f"Confidence: {detected.confidence:.2f}"
            )
    
    if missed_entities:
        st.write(f"❌ Completely Missed: {len(missed_entities)}")
        for entity in missed_entities:
            # Find the annotation for this missed entity
            truth_annotation = next(
                ann['location'] 
                for ann in annotations['entities'] 
                if ann['text'] == entity.text and ann['type'] == entity.entity_type
            )
            st.error(
                f"Text: {entity.text}\n"
                f"Position: {truth_annotation['start_char']}-{truth_annotation['end_char']}\n"
                f"Context: ...{text[max(0, truth_annotation['start_char']-20):min(len(text), truth_annotation['end_char']+20)]}..."
            )
    
    # Rest of the function remains the same...

    
    
    # Show additional detections
    extra_detections = []
    for detected_entity in detected_entities:
        if not any(truth_entity.text.strip() == detected_entity.text.strip() 
                  for truth_entity in truth_entities):
            extra_detections.append(detected_entity)
    
    if extra_detections:
        st.write(f"ℹ️ Additional Detections: {len(extra_detections)}")
        for entity in extra_detections:
            st.info(
                f"Text: {entity.text}\n"
                f"Position: {entity.start}-{entity.end}\n"
                f"Confidence: {entity.confidence:.2f}\n"
                f"Context: ...{text[max(0, entity.start-20):min(len(text), entity.end+20)]}..."
            )

def main():
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
        ["Run All Tests", "Investigate Entity Type"]
    )

    if test_mode == "Run All Tests":
        selected_tests = [case['id'] for case in test_cases]
        if st.sidebar.button("Run All Tests"):
            # Your existing full test suite code here
            st.info("Running full test suite...")
            
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