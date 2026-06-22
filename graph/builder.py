"""
graph/builder.py — Document graph construction from page layouts.

Converts a PageLayout into a directed graph (NetworkX DiGraph) where nodes
represent layout elements and edges encode explicit spatial relationships
(above, below, left_of, right_of, contains) plus reading-order links.

Design notes
------------
• **Why explicit spatial relations instead of kNN?**  kNN on centroids is
  symmetric and ignores orientation.  Explicit directional edges (above,
  left_of, …) give downstream GNNs a richer, asymmetric message-passing
  structure that mirrors how humans parse documents: "title is *above* body
  text" carries more information than "title is *near* body text."

• **Reading order** is inferred geometrically (top-to-bottom rows, left-to-
  right within each row).  This simple heuristic works for single-column and
  many multi-column Western documents.  Column detection can be layered on
  later.

• The module is intentionally pure Python / NetworkX — no GPU required —
  so graph construction can run offline on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import networkx as nx

from data.loader import LayoutElement, PageLayout
from graph.spatial import SpatialConfig, compute_spatial_relations


# ---------------------------------------------------------------------------
# DocumentGraph dataclass
# ---------------------------------------------------------------------------

@dataclass
class DocumentGraph:
    """Lightweight wrapper around an ``nx.DiGraph`` built from a page layout.

    Attributes
    ----------
    graph : nx.DiGraph
        The directed graph.  Each node has attributes ``type`` (category
        string), ``bbox`` (x, y, w, h tuple), and ``area``.  Each edge has
        ``edge_type`` ∈ {"above", "below", "left_of", "right_of",
        "contains", "reading_order"}.
    page_id : str
        Identifier of the source page.
    num_nodes : int
        Number of layout-element nodes.
    num_edges : int
        Number of edges (spatial + reading-order).
    node_types : dict[str, int]
        Mapping ``{category: count}`` for quick statistics.
    """

    graph: nx.DiGraph
    page_id: str
    num_nodes: int
    num_edges: int
    node_types: Dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _add_nodes(G: nx.DiGraph, elements: List[LayoutElement]) -> None:
    """Add one node per layout element with standard attributes."""
    for elem in elements:
        G.add_node(
            elem.id,
            type=elem.category,
            bbox=(elem.bbox.x, elem.bbox.y, elem.bbox.w, elem.bbox.h),
            area=elem.area,
        )


def _add_spatial_edges(
    G: nx.DiGraph,
    elements: List[LayoutElement],
    page_width: float,
    page_height: float,
    spatial_config: Optional[SpatialConfig],
) -> None:
    """Compute and add directed spatial-relationship edges via :mod:`graph.spatial`."""
    relations = compute_spatial_relations(
        elements,
        page_width=page_width,
        page_height=page_height,
        config=spatial_config,
    )
    for src_id, tgt_id, rel_type in relations:
        # If an edge already exists between this pair we still add the new
        # relation — NetworkX DiGraph replaces the dict, so we store all
        # relation types for a pair as a list under "edge_types" and keep
        # the first as the primary "edge_type".
        if G.has_edge(src_id, tgt_id):
            existing = G[src_id][tgt_id].setdefault("edge_types", [])
            if rel_type not in existing:
                existing.append(rel_type)
        else:
            G.add_edge(
                src_id,
                tgt_id,
                edge_type=rel_type,
                edge_types=[rel_type],
            )


def _infer_reading_order(
    elements: List[LayoutElement],
    row_tolerance: float,
) -> List[LayoutElement]:
    """Sort elements into reading order using recursive XY-cut.

    Recursively splits the document along empty horizontal or vertical gaps.
    Handles multi-column layouts and cross-column headers elegantly.
    Falls back to a row-based sort if no clear cuts are found.
    """
    if len(elements) <= 1:
        return list(elements)

    def xy_cut(elems: List[LayoutElement]) -> List[LayoutElement]:
        if len(elems) <= 1:
            return elems
            
        # Try horizontal cut (split top and bottom)
        elems_y = sorted(elems, key=lambda e: e.bbox.y)
        for i in range(1, len(elems_y)):
            max_bottom = max(e.bbox.y2 for e in elems_y[:i])
            # 2.0px tolerance to ignore slight overlapping artifacts
            if elems_y[i].bbox.y > max_bottom - 2.0:
                return xy_cut(elems_y[:i]) + xy_cut(elems_y[i:])
                
        # Try vertical cut (split left and right)
        elems_x = sorted(elems, key=lambda e: e.bbox.x)
        for i in range(1, len(elems_x)):
            max_right = max(e.bbox.x2 for e in elems_x[:i])
            if elems_x[i].bbox.x > max_right - 2.0:
                return xy_cut(elems_x[:i]) + xy_cut(elems_x[i:])
                
        # No clear cut found, fall back to simple row-based sorting
        return _row_sort(elems, row_tolerance)

    def _row_sort(elems: List[LayoutElement], tol: float) -> List[LayoutElement]:
        sorted_elems = sorted(elems, key=lambda e: (e.bbox.cy, e.bbox.cx))
        rows = []
        current_row = [sorted_elems[0]]
        for elem in sorted_elems[1:]:
            if abs(elem.bbox.cy - current_row[0].bbox.cy) <= tol:
                current_row.append(elem)
            else:
                rows.append(current_row)
                current_row = [elem]
        rows.append(current_row)
        ordered = []
        for row in rows:
            ordered.extend(sorted(row, key=lambda e: e.bbox.cx))
        return ordered

    return xy_cut(elements)


def _add_reading_order_edges(
    G: nx.DiGraph,
    elements: List[LayoutElement],
    row_tolerance: float = 20.0,
) -> None:
    """Add sequential reading-order edges between consecutive elements."""
    if len(elements) <= 1:
        return

    ordered = _infer_reading_order(elements, row_tolerance)

    for prev_elem, next_elem in zip(ordered[:-1], ordered[1:]):
        if G.has_edge(prev_elem.id, next_elem.id):
            existing = G[prev_elem.id][next_elem.id].setdefault("edge_types", [])
            if "reading_order" not in existing:
                existing.append("reading_order")
        else:
            G.add_edge(
                prev_elem.id,
                next_elem.id,
                edge_type="reading_order",
                edge_types=["reading_order"],
            )


def _node_type_counts(elements: List[LayoutElement]) -> Dict[str, int]:
    """Count elements per category."""
    counts: Dict[str, int] = {}
    for elem in elements:
        counts[elem.category] = counts.get(elem.category, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_graph(
    page: PageLayout,
    spatial_config: Optional[SpatialConfig] = None,
    row_tolerance: float = 20.0,
    raw_image = None,
    visual_encoder = None,
) -> DocumentGraph:
    """Construct a :class:`DocumentGraph` from a single :class:`PageLayout`.

    Parameters
    ----------
    page : PageLayout
        Source page with annotated layout elements.
    spatial_config : SpatialConfig, optional
        Override default proximity / overlap thresholds.
    row_tolerance : float
        Vertical-centre tolerance (px) for same-row grouping when
        inferring reading order.

    Returns
    -------
    DocumentGraph
    """
    G = nx.DiGraph()

    # --- Nodes ---
    _add_nodes(G, page.elements)
    
    # --- Multimodal Selective Vision ---
    if raw_image is not None and visual_encoder is not None:
        visual_elems = [e for e in page.elements if e.category in ('figure', 'picture', 'table')]
        if visual_elems:
            bboxes = [(e.bbox.x, e.bbox.y, e.bbox.w, e.bbox.h) for e in visual_elems]
            vis_features = visual_encoder(raw_image, bboxes).cpu()
            for elem, feat in zip(visual_elems, vis_features):
                G.nodes[elem.id]['visual_features'] = feat # Keep as tensor

    # --- Spatial edges (above, below, left_of, right_of, contains) ---
    _add_spatial_edges(G, page.elements, page.width, page.height, spatial_config)

    # --- Reading-order edges ---
    _add_reading_order_edges(G, page.elements, row_tolerance=row_tolerance)

    return DocumentGraph(
        graph=G,
        page_id=page.page_id,
        num_nodes=G.number_of_nodes(),
        num_edges=G.number_of_edges(),
        node_types=_node_type_counts(page.elements),
    )


def build_graphs(
    pages: List[PageLayout],
    spatial_config: Optional[SpatialConfig] = None,
    row_tolerance: float = 20.0,
    raw_images: Optional[List] = None,
    visual_encoder = None,
) -> List[DocumentGraph]:
    """Batch-build :class:`DocumentGraph` objects for multiple pages.

    Parameters
    ----------
    pages : list[PageLayout]
        Source pages.
    spatial_config : SpatialConfig, optional
        Shared spatial thresholds for all pages.
    row_tolerance : float
        Reading-order row tolerance (px).

    Returns
    -------
    list[DocumentGraph]
    """
    raw_images = raw_images or [None] * len(pages)
    return [
        build_graph(
            page, 
            spatial_config=spatial_config, 
            row_tolerance=row_tolerance,
            raw_image=img,
            visual_encoder=visual_encoder
        )
        for page, img in zip(pages, raw_images)
    ]
