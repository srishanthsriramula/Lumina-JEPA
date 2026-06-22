"""
metrics/ocr_area.py — OCR area ratio metric.

Motivation
----------
In a document, only some regions carry readable text that an OCR engine
would process.  Knowing the fraction of the page area occupied by text-
bearing elements helps calibrate:

* How much of the page the graph must explicitly represent.
* Whether the document is text-heavy (article) or graphics-heavy (poster).
* Downstream decisions like whether to run OCR at all on certain regions.

We partition layout categories into two groups:

    text-bearing :  title, text, caption, list, header, footer
    non-text     :  table, figure  (these need specialised extractors)

The **OCR area ratio** is:

    ocr_area_ratio = Σ area(text elements) / (page_width × page_height)

Overlapping elements are handled correctly using a sweep-line union-area
algorithm so that shared pixels are not double-counted.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Set, Tuple

from data.loader import BBox, LayoutElement, PageLayout


# ---------------------------------------------------------------------------
# Category partitions
# ---------------------------------------------------------------------------

TEXT_CATEGORIES: Set[str] = {"title", "text", "caption", "list", "header", "footer"}
NON_TEXT_CATEGORIES: Set[str] = {"table", "figure"}


# ---------------------------------------------------------------------------
# Overlap-aware area computation (axis-aligned rectangle union)
# ---------------------------------------------------------------------------

def _union_area(boxes: List[BBox]) -> float:
    """Compute the area of the union of axis-aligned rectangles.

    Uses a coordinate-compression + sweep-line approach that is exact and
    runs in O(n² log n) — perfectly fine for the tens-of-elements scale of
    document pages.

    Parameters
    ----------
    boxes : list[BBox]
        Axis-aligned bounding boxes.

    Returns
    -------
    float
        Total union area (no double-counting of overlaps).
    """
    if not boxes:
        return 0.0

    # Collect all unique x-coordinates and y-coordinates
    xs: List[float] = []
    ys: List[float] = []
    for b in boxes:
        xs.extend([b.x, b.x2])
        ys.extend([b.y, b.y2])

    xs = sorted(set(xs))
    ys = sorted(set(ys))

    if len(xs) < 2 or len(ys) < 2:
        # Degenerate: all boxes are zero-width or zero-height
        return 0.0

    total_area = 0.0

    # Sweep over each cell in the compressed grid
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            # Centre of the cell (used to test containment)
            cx = (xs[i] + xs[i + 1]) / 2.0
            cy = (ys[j] + ys[j + 1]) / 2.0

            # If any box contains this cell, add its area
            for b in boxes:
                if b.x <= cx <= b.x2 and b.y <= cy <= b.y2:
                    cell_area = (xs[i + 1] - xs[i]) * (ys[j + 1] - ys[j])
                    total_area += cell_area
                    break  # only count once

    return total_area


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_ocr_area_ratio(page: PageLayout) -> float:
    """Fraction of the page area covered by text-bearing elements.

    Parameters
    ----------
    page : PageLayout
        A fully-annotated document page.

    Returns
    -------
    float
        Value in [0, 1].  0 means no text regions; 1 means the entire
        page is text.  Returns 0.0 for pages with zero area.
    """
    page_area = page.width * page.height
    if page_area <= 0:
        return 0.0

    text_boxes = [
        elem.bbox for elem in page.elements if elem.category in TEXT_CATEGORIES
    ]

    if not text_boxes:
        return 0.0

    union = _union_area(text_boxes)
    # Clamp to [0, 1] — union can slightly exceed page_area if elements
    # extend beyond the page boundary in noisy annotations.
    return min(union / page_area, 1.0)


def compute_non_text_area_ratio(page: PageLayout) -> float:
    """Fraction of the page covered by non-text elements (table, figure).

    Useful for understanding the graphics density of a page.
    """
    page_area = page.width * page.height
    if page_area <= 0:
        return 0.0

    non_text_boxes = [
        elem.bbox for elem in page.elements if elem.category in NON_TEXT_CATEGORIES
    ]

    if not non_text_boxes:
        return 0.0

    union = _union_area(non_text_boxes)
    return min(union / page_area, 1.0)


def compute_ocr_area_stats(pages: List[PageLayout]) -> Dict[str, float]:
    """Descriptive statistics of OCR area ratios across a corpus.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``min``, ``max``, ``std``, ``count``.
    """
    if not pages:
        raise ValueError("Cannot compute OCR area stats on an empty list.")

    ratios = [compute_ocr_area_ratio(p) for p in pages]

    return {
        "mean": statistics.mean(ratios),
        "median": statistics.median(ratios),
        "min": min(ratios),
        "max": max(ratios),
        "std": statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
        "count": len(ratios),
    }
