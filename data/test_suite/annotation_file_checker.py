# data/test_suite/annotation_file_checker.py
# Opens each resume in the originals folder and checks the character positions as listed
# Used chatGPT to do the original yaml file creation and there may have been inconsistent results
# becasue this is the testing truth set, we need to verify these positional details

import yaml
import fitz
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import re

def setup_logger(level: str):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(levelname)s: %(message)s")

def load_yaml(yaml_file: Path) -> List[Dict]:
    try:
        with yaml_file.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)
        if not data or "entities" not in data:
            raise ValueError("Invalid YAML structure: Missing 'entities' key.")
        return data.get("entities", [])
    except Exception as e:
        logging.error(f"Failed to load YAML {yaml_file.name}: {e}")
        return []

def save_yaml(original_yaml: Path, annotations: List[Dict], corrected_path: Path):
    corrected_data = {"file_name": corrected_path.stem, "entities": annotations}
    with corrected_path.open("w", encoding="utf-8") as file:
        yaml.dump(corrected_data, file, default_flow_style=False, sort_keys=False)

def find_text_in_pdf(page_text: str, text_to_find: str) -> Tuple[int, int]:
    # Initial exact search
    start_char = page_text.find(text_to_find)
    if start_char != -1:
        return start_char, start_char + len(text_to_find)
    
    def normalize_text(text: str) -> str:
        normalized = re.sub(r'\s+', ' ', text)
        normalized = normalized.strip()
        return normalized
    
    normalized_page = normalize_text(page_text)
    normalized_search = normalize_text(text_to_find)
    
    # Try normalized exact match
    start_char = normalized_page.find(normalized_search)
    if start_char != -1:
        return start_char, start_char + len(normalized_search)
    
    # Fuzzy matching for minor differences
    search_words = normalized_search.split()
    page_words = normalized_page.split()
    
    best_match_start = -1
    best_match_len = 0
    
    for i in range(len(page_words)):
        matches = 0
        for j in range(len(search_words)):
            if (i + j) < len(page_words) and page_words[i + j] == search_words[j]:
                matches += 1
            else:
                break
        if matches > best_match_len:
            best_match_len = matches
            best_match_start = i
    
    # Good partial match threshold (80%)
    if best_match_len >= len(search_words) * 0.8:
        char_start = 0
        for i in range(best_match_start):
            char_start += len(page_words[i]) + 1
        char_end = char_start
        for i in range(best_match_len):
            char_end += len(page_words[best_match_start + i]) + 1
        return char_start, char_end - 1
        
    return -1, -1

def correct_yaml_for_pdf(pdf_path: Path, yaml_path: Path, output_dir: Path) -> int:
    doc = fitz.open(pdf_path)
    annotations = load_yaml(yaml_path)
    corrections = 0
    issues = {
        "missing_text_key": 0,
        "text_not_found": 0,
        "invalid_annotation": 0,
        "mismatched_positions": 0
    }

    for annotation in annotations:
        if not isinstance(annotation, dict):
            logging.error(f"Invalid annotation format in {yaml_path.name}: {annotation}")
            issues["invalid_annotation"] += 1
            continue

        if "text" not in annotation:
            annotation["needs_review"] = True
            annotation["review_reason"] = "Missing 'text' key"
            issues["missing_text_key"] += 1
            continue

        text_to_find = annotation["text"]
        found = False
        original_positions = annotation.get("location", {})

        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            start_char, end_char = find_text_in_pdf(page_text, text_to_find)

            if start_char != -1:
                current_positions = {
                    "page": page_num,
                    "start_char": start_char,
                    "end_char": end_char
                }
                
                if current_positions != original_positions:
                    annotation["location"] = current_positions
                    corrections += 1
                    issues["mismatched_positions"] += 1
                    logging.debug(f"Updated position for '{text_to_find}' on page {page_num}")
                found = True
                break

        if not found:
            annotation["needs_review"] = True
            annotation["review_reason"] = f"Text not found in PDF"
            issues["text_not_found"] += 1
            logging.warning(f"Text not found: '{text_to_find}'")

    corrected_yaml_path = output_dir / f"{yaml_path.stem}_corrected.yaml"
    save_yaml(yaml_path, annotations, corrected_yaml_path)
    
    logging.info(f"Processed {len(annotations)} annotations in {pdf_path.name}")
    logging.info(f"Summary: {corrections} corrections, {issues}")
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
    for pdf_name, corrections in results:
        logging.info(f"{pdf_name}: {corrections} corrections")

if __name__ == "__main__":
    input_dir = Path("./originals")
    yaml_dir = Path("./annotations")
    output_dir = Path("./corrected_annotations")
    log_level = "INFO" # DEBUG INFO WARNING in order of  Most - > Least output
    main(input_dir, yaml_dir, output_dir, log_level)