import string
import fitz
from typing import List

def normalize_text(text: str) -> str:
    """
    Normalize text by stripping punctuation and converting to lowercase.
    """
    return text.translate(str.maketrans('', '', string.punctuation)).strip().lower()

def combine_bboxes(bboxes: List[fitz.Rect]) -> fitz.Rect:
    """
    Combines multiple bounding boxes into one unified rectangle.
    """
    if not bboxes:
        return None

    x0 = min(bbox.x0 for bbox in bboxes)
    y0 = min(bbox.y0 for bbox in bboxes)
    x1 = max(bbox.x1 for bbox in bboxes)
    y1 = max(bbox.y1 for bbox in bboxes)
    return fitz.Rect(x0, y0, x1, y1)

def aggregate_line_bboxes(bboxes: List[fitz.Rect]) -> fitz.Rect:
    """
    Combines bounding boxes for a single line into one rectangle.
    """
    return combine_bboxes(bboxes)
