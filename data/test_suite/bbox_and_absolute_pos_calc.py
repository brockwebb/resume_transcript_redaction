#!/usr/bin/env python3
"""
annotation_file_checker.py

This script opens each PDF in the "originals" folder and checks the annotation
positions from YAML files (located in the "annotations" folder). It groups annotations
by both text and type (e.g. DATE_TIME) and collects all occurrences for that text
in the PDF (bounding boxes and character positions). It then assigns an incremental 
"instance" number (1, 2, …) to each annotation in that group and updates its location
based on the corresponding occurrence in reading order. If there aren’t enough occurrences,
the annotation is flagged for review.

The location information includes:
  - Absolute bounding box coordinates (x0, y0, x1, y1) on the page.
  - Absolute character positions in the document (not just the page) for the occurrence.

The corrected YAML (with an added "instance" field) is saved to the "corrected_annotations" folder.
"""

import yaml
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import logging
import re
from collections import defaultdict

def setup_logger(level: str):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

def load_yaml(yaml_file: Path) -> Dict[str, Any]:
    try:
        with yaml_file.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not data or "entities" not in data:
            raise ValueError("Invalid YAML structure: Missing 'entities' key.")
        return data
    except Exception as e:
        logging.error(f"Failed to load YAML {yaml_file.name}: {e}")
        return {}

def save_yaml(data: Dict[str, Any], corrected_path: Path):
    with corrected_path.open("w", encoding="utf-8") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)

def compute_cumulative_offsets(doc: fitz.Document) -> Dict[int, int]:
    """
    Compute a mapping from page number to the cumulative character offset.
    For page N, cumulative_offsets[N] is the total number of characters in all pages before N.
    """
    cumulative_offsets = {}
    total = 0
    for page_num, page in enumerate(doc, start=1):
        cumulative_offsets[page_num] = total
        page_text = page.get_text()
        total += len(page_text)
    return cumulative_offsets

def get_occurrences_for_text(doc: fitz.Document, text: str, cumulative_offsets: Dict[int, int]) -> List[Dict]:
    """
    Search the entire document for all occurrences of the given text.
    For each occurrence, record:
      - page number,
      - bounding box (absolute coordinates in the page),
      - absolute character offsets (global across the document).
    Returns a list of occurrence dictionaries sorted in reading order.
    """
    occurrences = []
    pattern = re.escape(text)
    
    for page_num, page in enumerate(doc, start=1):
        # Get bounding boxes (in reading order) using page.search_for.
        rects = page.search_for(text)
        # Get full text of the page.
        page_text = page.get_text()
        # Use regex to find all occurrences in the page text.
        char_matches = list(re.finditer(pattern, page_text))
        count = min(len(rects), len(char_matches))
        for i in range(count):
            match = char_matches[i]
            abs_start = cumulative_offsets[page_num] + match.start()
            abs_end = cumulative_offsets[page_num] + match.end()
            occurrence = {
                "page": page_num,
                "bbox": rects[i],
                "start_char": abs_start,
                "end_char": abs_end,
            }
            occurrences.append(occurrence)
    
    # Sort occurrences by page, then by vertical (y0) then horizontal (x0) order.
    occurrences.sort(key=lambda occ: (occ["page"], occ["bbox"].y0, occ["bbox"].x0))
    return occurrences

def correct_yaml_for_pdf(pdf_path: Path, yaml_path: Path, output_dir: Path) -> int:
    """
    For a given PDF and YAML file, group annotations by (text, type) and for each group:
      - Collect all PDF occurrences (bounding boxes and character offsets) for that text.
      - Compute absolute character offsets using cumulative offsets.
      - Assign each YAML annotation an incremental instance number (1,2,…) based on its position in the group.
      - Update the YAML "location" field with the occurrence data.
    
    If a group has fewer occurrences than annotations, the annotation is flagged for review.
    
    The corrected YAML is saved in output_dir.
    """
    doc = fitz.open(pdf_path)
    cumulative_offsets = compute_cumulative_offsets(doc)
    
    yaml_data = load_yaml(yaml_path)
    if not yaml_data:
        return 0
    annotations = yaml_data.get("entities", [])
    corrections = 0
    issues = defaultdict(int)
    
    # Group annotations by (text, type) tuple.
    groups: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for ann in annotations:
        if not isinstance(ann, dict):
            logging.error(f"Invalid annotation format in {yaml_path.name}: {ann}")
            issues["invalid_annotation"] += 1
            continue
        if "text" not in ann or "type" not in ann:
            ann["needs_review"] = True
            ann["review_reason"] = "Missing 'text' or 'type' key"
            issues["missing_key"] += 1
            continue
        key = (ann["text"], ann["type"])
        groups[key].append(ann)
    
    # For each (text, type) group, collect occurrences in the PDF and assign instance numbers.
    for (text, entity_type), group_annotations in groups.items():
        occurrences = get_occurrences_for_text(doc, text, cumulative_offsets)
        logging.debug(f"Found {len(occurrences)} occurrences of '{text}' (type {entity_type})")
        
        # Sort the YAML annotations in the group by their existing location start_char (if available)
        # or simply by the order they appear.
        group_annotations.sort(key=lambda a: a.get("location", {}).get("start_char", 0))
        
        for idx, ann in enumerate(group_annotations):
            instance = idx + 1  # instance numbering starts at 1
            ann["instance"] = instance  # add instance field to the output
            if idx < len(occurrences):
                occ = occurrences[idx]
                current_location = {
                    "page": occ["page"],
                    "x0": occ["bbox"].x0,
                    "y0": occ["bbox"].y0,
                    "x1": occ["bbox"].x1,
                    "y1": occ["bbox"].y1,
                    "start_char": occ["start_char"],
                    "end_char": occ["end_char"],
                }
                if ann.get("location", {}) != current_location:
                    ann["location"] = current_location
                    corrections += 1
                    issues["mismatched_positions"] += 1
                    logging.debug(f"Updated '{text}' (type {entity_type}) instance {instance} on page {occ['page']}")
            else:
                ann["needs_review"] = True
                ann["review_reason"] = f"Only found {len(occurrences)} occurrence(s) in PDF."
                issues["text_not_found"] += 1
                logging.warning(f"Not enough occurrences for '{text}' (type {entity_type}), instance {instance} requested but only {len(occurrences)} found")
    
    corrected_yaml_path = output_dir / f"{yaml_path.stem}_corrected.yaml"
    save_yaml(yaml_data, corrected_yaml_path)
    
    logging.info(f"Processed {len(annotations)} annotations in {pdf_path.name}")
    logging.info(f"Summary for {pdf_path.name}: {corrections} corrections, issues: {dict(issues)}")
    return corrections

def main(input_dir: Path, yaml_dir: Path, output_dir: Path, log_level: str):
    setup_logger(log_level)
    input_files = list(input_dir.glob("*.pdf"))
    yaml_files = {y.stem: y for y in yaml_dir.glob("*.yaml")}
    output_dir.mkdir(exist_ok=True)

    results = []
    for pdf_path in input_files:
        yaml_path = yaml_files.get(pdf_path.stem)
        if not yaml_path:
            logging.warning(f"No YAML found for {pdf_path.name}")
            continue

        corrections = correct_yaml_for_pdf(pdf_path, yaml_path, output_dir)
        results.append((pdf_path.name, corrections))

    logging.info("\nCorrection Summary:")
    for pdf_name, corr in results:
        logging.info(f"{pdf_name}: {corr} corrections")

if __name__ == "__main__":
    # Directories for PDFs, YAML annotations, and output
    input_dir = Path("./originals")
    yaml_dir = Path("./annotations")
    output_dir = Path("./corrected_annotations")
    log_level = "INFO"  # Options: DEBUG, INFO, WARNING
    main(input_dir, yaml_dir, output_dir, log_level)
