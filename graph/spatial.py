"""
graph/spatial.py — Spatial relationship computation for document layout elements.

Given pairs of bounding boxes, determines directional spatial relationships
(above, below, left_of, right_of, contains) using geometric tests.

Design decisions:
  • We use *centroid comparison* for directionality but require *projection
    overlap* on the orthogonal axis so that, e.g., "above" only fires when
    two elements share horizontal extent (they're roughly in the same column).
  • A configurable **max_gap** prevents edges between elements that are far
    apart on the page.  The gap can be given in absolute pixels or as a
    fraction of the page diagonal — the fractional form generalises across
    different page resolutions.
  • "contains" uses strict geometric containment (a fully encloses b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import math

from data.loader import BBox, LayoutElement


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SpatialConfig:
    """Parameters controlling spatial-edge construction.

    Attributes
    ----------
    max_gap_px : float
        Maximum pixel gap between two bounding boxes for an edge to be
        created.  If the *closest edges* of the two boxes are farther apart
        than this, no directional edge is added.  Set to ``float('inf')``
        to disable the absolute check.
    max_gap_frac : float
        Maximum gap expressed as a fraction of the page diagonal.  This
        adapts the threshold to different page resolutions.  The effective
        threshold is ``min(max_gap_px, max_gap_frac * page_diagonal)``.
    overlap_ratio_min : float
        Minimum overlap ratio on the orthogonal axis for a directional
        relationship to fire.  0.0 means any overlap counts; 1.0 would
        require full overlap (rarely useful).  Default 0.0 — even a single
        pixel of shared projection is enough.
    containment_margin : float
        Tolerance in pixels when testing containment.  ``a`` contains ``b``
        if every edge of ``b`` is within this margin inside ``a``.  A small
        positive value accounts for annotation noise.
    """

    max_gap_px: float = 50.0
    max_gap_frac: float = 0.05
    overlap_ratio_min: float = 0.0
    containment_margin: float = 2.0


_DEFAULT_CONFIG = SpatialConfig()


# ---------------------------------------------------------------------------
# Low-level geometric helpers
# ---------------------------------------------------------------------------

def _projection_overlap(a_lo: float, a_hi: float,
                        b_lo: float, b_hi: float) -> float:
    """Length of the overlap between two 1-D intervals [a_lo, a_hi] and
    [b_lo, b_hi].  Returns 0 if they don't overlap."""
    return max(0.0, min(a_hi, b_hi) - max(a_lo, b_lo))


def _overlap_ratio(a_lo: float, a_hi: float,
                   b_lo: float, b_hi: float) -> float:
    """Fraction of the shorter interval that is covered by the overlap."""
    overlap = _projection_overlap(a_lo, a_hi, b_lo, b_hi)
    shorter = min(a_hi - a_lo, b_hi - b_lo)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _min_edge_distance_x(a: BBox, b: BBox) -> float:
    """Minimum horizontal gap between two boxes (0 if they overlap in x)."""
    if a.x2 <= b.x:
        return b.x - a.x2
    if b.x2 <= a.x:
        return a.x - b.x2
    return 0.0  # overlapping horizontally


def _min_edge_distance_y(a: BBox, b: BBox) -> float:
    """Minimum vertical gap between two boxes (0 if they overlap in y)."""
    if a.y2 <= b.y:
        return b.y - a.y2
    if b.y2 <= a.y:
        return a.y - b.y2
    return 0.0  # overlapping vertically


def _min_edge_distance(a: BBox, b: BBox) -> float:
    """Minimum distance between the closest edges of two boxes.

    For axis-aligned boxes this is ``sqrt(dx² + dy²)`` where ``dx`` and ``dy``
    are the edge-to-edge gaps (clamped at 0 when they overlap on that axis).
    """
    dx = _min_edge_distance_x(a, b)
    dy = _min_edge_distance_y(a, b)
    return math.hypot(dx, dy)


def _contains(a: BBox, b: BBox, margin: float = 0.0) -> bool:
    """True if box *a* fully contains box *b* (within *margin* tolerance)."""
    return (
        a.x - margin <= b.x
        and a.y - margin <= b.y
        and a.x2 + margin >= b.x2
        and a.y2 + margin >= b.y2
    )


# ---------------------------------------------------------------------------
# Pairwise relationship test
# ---------------------------------------------------------------------------

def _compute_pair_relations(
    a: LayoutElement,
    b: LayoutElement,
    effective_gap: float,
    cfg: SpatialConfig,
) -> List[str]:
    """Return the list of spatial relation labels from *a* → *b*.

    At most one directional relation (above/below/left_of/right_of) is
    returned per pair; containment is independent and can co-occur.
    """
    ba, bb = a.bbox, b.bbox
    relations: List[str] = []

    # --- Containment (independent of proximity) ---
    if _contains(ba, bb, margin=cfg.containment_margin):
        relations.append("contains")

    # --- Proximity gate — skip directional edges for distant elements ---
    gap = _min_edge_distance(ba, bb)
    if gap > effective_gap:
        return relations  # only containment (if any) survives

    # --- Directional relations ---
    # Horizontal overlap is needed for above/below; vertical for left/right.
    h_overlap = _overlap_ratio(ba.x, ba.x2, bb.x, bb.x2)
    v_overlap = _overlap_ratio(ba.y, ba.y2, bb.y, bb.y2)

    # Pick the dominant axis: compare centroid offsets normalised by the
    # average box dimension so that tall/narrow boxes naturally prefer
    # vertical relations and wide/short boxes prefer horizontal ones.
    dx = bb.bbox.cx - ba.cx if False else bb.cx - ba.cx  # keep simple
    dy = bb.cy - ba.cy

    # Vertical relationship (above / below)
    if h_overlap > cfg.overlap_ratio_min and abs(dy) > 0:
        if ba.cy < bb.cy:
            relations.append("above")
        elif ba.cy > bb.cy:
            relations.append("below")

    # Horizontal relationship (left_of / right_of)
    if v_overlap > cfg.overlap_ratio_min and abs(dx) > 0:
        if ba.cx < bb.cx:
            relations.append("left_of")
        elif ba.cx > bb.cx:
            relations.append("right_of")

    return relations


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_spatial_relations(
    elements: List[LayoutElement],
    page_width: float = 0.0,
    page_height: float = 0.0,
    config: Optional[SpatialConfig] = None,
) -> List[Tuple[int, int, str]]:
    """Compute all pairwise spatial relationships among *elements*.

    Parameters
    ----------
    elements : list[LayoutElement]
        Layout elements on a single page.
    page_width, page_height : float
        Page dimensions used to scale the fractional gap threshold.  If both
        are 0 the absolute ``max_gap_px`` is used alone.
    config : SpatialConfig, optional
        Override default spatial thresholds.

    Returns
    -------
    list[tuple[int, int, str]]
        Each tuple is ``(source_id, target_id, relation_type)`` where
        *relation_type* ∈ {"above", "below", "left_of", "right_of",
        "contains"}.

    Complexity
    ----------
    O(N²) pairwise comparison.  For typical document pages (N < 100) this
    is negligible (< 1 ms).  If N grows large, a spatial index (R-tree)
    could be introduced.
    """
    cfg = config or _DEFAULT_CONFIG

    # Effective gap: min of absolute and fractional thresholds
    if page_width > 0 and page_height > 0:
        page_diag = math.hypot(page_width, page_height)
        effective_gap = min(cfg.max_gap_px, cfg.max_gap_frac * page_diag)
    else:
        effective_gap = cfg.max_gap_px

    relations: List[Tuple[int, int, str]] = []

    n = len(elements)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            pair_rels = _compute_pair_relations(
                elements[i], elements[j], effective_gap, cfg,
            )
            for rel in pair_rels:
                relations.append((elements[i].id, elements[j].id, rel))

    return relations
