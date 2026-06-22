"""
metrics/compression.py — Graph compression ratio metric.

Motivation
----------
A core thesis of the graph-first approach is that document structure can be
captured with far fewer "tokens" (graph nodes) than the flat grid of visual
tokens emitted by a vision encoder (typically 4096 patches for a 2048×2048
image at patch-size 32).  The **compression ratio** quantifies this:

    compression_ratio = visual_tokens / num_graph_nodes

A ratio of e.g. 100× means the graph represents the page with 100× fewer
elements than a naïve patch tokeniser.  Higher is better (more compression)
as long as downstream metrics (reading-order, information retention) remain
healthy.
"""

from __future__ import annotations

import statistics
from typing import Dict, List

from graph.builder import DocumentGraph


# ---------------------------------------------------------------------------
# Single-graph compression ratio
# ---------------------------------------------------------------------------

def compute_compression(
    graph: DocumentGraph,
    visual_tokens: int = 4096,
) -> float:
    """Return ``visual_tokens / num_nodes`` for a single document graph.

    Parameters
    ----------
    graph : DocumentGraph
        The document graph whose node count serves as the "graph token" count.
    visual_tokens : int
        Baseline number of visual tokens a typical vision encoder would
        produce for the same page (default 4096 = 64×64 patch grid).

    Returns
    -------
    float
        The compression ratio.  Returns ``float('inf')`` when the graph
        has zero nodes (degenerate/empty page).
    """
    if graph.num_nodes == 0:
        return float("inf")
    return visual_tokens / graph.num_nodes


# ---------------------------------------------------------------------------
# Aggregate statistics over a corpus
# ---------------------------------------------------------------------------

def compute_compression_stats(
    graphs: List[DocumentGraph],
    visual_tokens: int = 4096,
) -> Dict[str, float]:
    """Compute descriptive statistics of compression ratios over many graphs.

    Parameters
    ----------
    graphs : list[DocumentGraph]
        Collection of document graphs (e.g. all pages in a dataset split).
    visual_tokens : int
        Baseline visual-token count (same semantics as in
        :func:`compute_compression`).

    Returns
    -------
    dict
        Keys: ``mean``, ``median``, ``min``, ``max``, ``std``, ``count``.
        All values are plain floats suitable for JSON serialisation.

    Raises
    ------
    ValueError
        If *graphs* is empty.
    """
    if not graphs:
        raise ValueError("Cannot compute compression stats on an empty list of graphs.")

    ratios = [compute_compression(g, visual_tokens) for g in graphs]

    # Filter out infinite ratios (empty pages) for robust statistics
    finite_ratios = [r for r in ratios if r != float("inf")]
    if not finite_ratios:
        return {
            "mean": float("inf"),
            "median": float("inf"),
            "min": float("inf"),
            "max": float("inf"),
            "std": 0.0,
            "count": len(ratios),
            "empty_pages": len(ratios),
        }

    return {
        "mean": statistics.mean(finite_ratios),
        "median": statistics.median(finite_ratios),
        "min": min(finite_ratios),
        "max": max(finite_ratios),
        "std": statistics.stdev(finite_ratios) if len(finite_ratios) > 1 else 0.0,
        "count": len(ratios),
        "empty_pages": len(ratios) - len(finite_ratios),
    }
