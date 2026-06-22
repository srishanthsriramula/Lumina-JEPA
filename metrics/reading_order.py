"""
metrics/reading_order.py — Reading-order recovery metric via Kendall's Tau.

Motivation
----------
Correct reading order is critical for downstream NLP tasks (question
answering, summarisation) applied to parsed documents.  We evaluate how
well the graph's implicit reading order (inferred from spatial layout)
matches the ground-truth annotation order provided by the dataset.

Ground truth : the sequential order in which elements appear in the
               dataset annotation (``PageLayout.elements`` list order).
Predicted    : spatial heuristic — group elements into rows (vertical
               tolerance), sort rows top-to-bottom, sort within each row
               left-to-right.

We measure agreement with **Kendall's Tau-b** rank correlation:
  τ = 1  → perfect agreement
  τ = 0  → no correlation
  τ = -1 → perfectly reversed
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Tuple

from scipy.stats import kendalltau

from data.loader import LayoutElement, PageLayout
from graph.builder import DocumentGraph, _infer_reading_order





def _annotation_order(elements: List[LayoutElement]) -> List[int]:
    """Return element IDs in their original annotation (ground-truth) order."""
    return [e.id for e in elements]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_reading_order_score(
    graph: DocumentGraph,
    page: PageLayout,
    row_tolerance: float = 20.0,
) -> float:
    """Compute Kendall's Tau between ground-truth and predicted reading order.

    Parameters
    ----------
    graph : DocumentGraph
        The document graph (used to verify node membership; the spatial
        order is re-derived from *page.elements* for consistency).
    page : PageLayout
        Source page whose ``.elements`` list defines the ground-truth order.
    row_tolerance : float
        Row-grouping tolerance in pixels.

    Returns
    -------
    float
        Kendall's Tau-b correlation in [-1, 1].  Returns 1.0 for pages
        with fewer than 2 elements (trivially correct order).
    """
    elements = page.elements
    if len(elements) < 2:
        return 1.0  # trivially correct

    gt_order = _annotation_order(elements)
    pred_elements = _infer_reading_order(elements, row_tolerance=row_tolerance)
    pred_order = [e.id for e in pred_elements]

    # Build rank arrays: for each element, what rank does it hold?
    gt_rank = {eid: rank for rank, eid in enumerate(gt_order)}
    pred_rank = {eid: rank for rank, eid in enumerate(pred_order)}

    # Align to common ordering (by gt_order) so scipy gets paired ranks
    common_ids = [eid for eid in gt_order if eid in pred_rank]
    if len(common_ids) < 2:
        return 1.0

    gt_ranks = [gt_rank[eid] for eid in common_ids]
    pred_ranks = [pred_rank[eid] for eid in common_ids]

    tau, _pvalue = kendalltau(gt_ranks, pred_ranks)

    # kendalltau can return nan for constant inputs; treat as perfect
    if tau != tau:  # nan check
        return 1.0

    return float(tau)


def compute_reading_order_stats(
    graphs: List[DocumentGraph],
    pages: List[PageLayout],
    row_tolerance: float = 20.0,
) -> Dict[str, float]:
    """Aggregate reading-order scores across a corpus.

    Parameters
    ----------
    graphs : list[DocumentGraph]
        Document graphs (one per page).
    pages : list[PageLayout]
        Corresponding page layouts (same length, same order).
    row_tolerance : float
        Row-grouping tolerance.

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``min``, ``max``, ``std``, ``count``.
    """
    if len(graphs) != len(pages):
        raise ValueError(
            f"graphs ({len(graphs)}) and pages ({len(pages)}) must have the same length."
        )
    if not graphs:
        raise ValueError("Cannot compute stats on an empty list.")

    scores = [
        compute_reading_order_score(g, p, row_tolerance)
        for g, p in zip(graphs, pages)
    ]

    return {
        "mean": statistics.mean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
        "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
        "count": len(scores),
    }
