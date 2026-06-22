"""
Vision-SARVAM: Centralized Configuration
=========================================

All project-wide constants, paths, dataset URLs, graph construction
hyperparameters, evaluation metric targets, and type definitions live here.

Design rationale:
- Single source of truth avoids magic numbers scattered across modules.
- dataclass-based configs enable easy serialization and CLI overrides later.
- Metric targets are derived from the core hypothesis: a graph
  representation can retain ≥95% of a document's semantic information
  while compressing the visual token count by ~20×.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List


# ──────────────────────────────────────────────
# 1. Directory layout
# ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR: Path = PROJECT_ROOT / "data_store"
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"
CACHE_DIR: Path = PROJECT_ROOT / ".cache"

# Dataset-specific directories (used by data/download.py and data/loader.py)
PUBLAYNET_DIR: Path = DATA_DIR / "publaynet"
DOCLAYNET_DIR: Path = DATA_DIR / "doclaynet"

# Ensure critical dirs exist at import time
for _dir in (DATA_DIR, OUTPUT_DIR, CACHE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────
# 2. Dataset configuration
# ──────────────────────────────────────────────
@dataclass
class DatasetConfig:
    """Configuration for a single document-layout dataset."""

    name: str
    url: str
    subset_size: int  # number of images to use in experiments
    description: str = ""


PUBLAYNET = DatasetConfig(
    name="PubLayNet",
    url="https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz",
    subset_size=1000,
    description=(
        "Large-scale dataset for document layout analysis, "
        "automatically annotated from PubMed Central™ articles."
    ),
)

DOCLAYNET = DatasetConfig(
    name="DocLayNet",
    url="https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.01/DocLayNet_core.zip",
    subset_size=500,
    description=(
        "Human-annotated dataset with 11 document categories and "
        "fine-grained layout classes from IBM Research."
    ),
)

DATASETS: Dict[str, DatasetConfig] = {
    "publaynet": PUBLAYNET,
    "doclaynet": DOCLAYNET,
}


# ──────────────────────────────────────────────
# 3. Graph construction parameters
# ──────────────────────────────────────────────
@dataclass
class GraphConstructionParams:
    """
    Hyperparameters that control how a document page is converted into a graph.

    * overlap_threshold  — minimum IoU between two bounding boxes to create
                           a 'contains' edge (parent-child nesting).
    * proximity_threshold — maximum normalized distance (fraction of page
                            diagonal) for two nodes to be considered spatial
                            neighbours and receive a directional edge.
    """

    overlap_threshold: float = 0.3
    proximity_threshold: float = 0.15


GRAPH_PARAMS = GraphConstructionParams()


# ──────────────────────────────────────────────
# 4. Metric targets (success criteria)
# ──────────────────────────────────────────────
@dataclass
class MetricTargets:
    """
    Quantitative targets that define "success" for Phase 1.

    * compression_target      — desired ratio  visual_tokens / graph_tokens.
                                A value of 20 means the graph should use
                                ≤ 1/20th of the tokens a flat patch grid needs.
    * reading_order_target    — Kendall τ ≥ 0.95 between predicted and
                                ground-truth reading order.
    * node_prediction_target  — F1 ≥ 0.90 for node-type classification
                                from graph features alone.
    * ocr_area_target         — ≤ 30% of the page area needs to be sent
                                to OCR when guided by the graph.
    """

    compression_target: float = 20.0
    reading_order_target: float = 0.95
    node_prediction_target: float = 0.90
    ocr_area_target: float = 0.30


METRIC_TARGETS = MetricTargets()


# ──────────────────────────────────────────────
# 5. Visual token budget
# ──────────────────────────────────────────────
# A 1024×1024 image with 16×16 patches yields 4096 visual tokens.
# The graph representation should beat this significantly.
IMAGE_SIZE: int = 1024
PATCH_SIZE: int = 16
VISUAL_TOKEN_COUNT: int = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 4096


# ──────────────────────────────────────────────
# 6. Node & edge type vocabularies
# ──────────────────────────────────────────────
class NodeType(str, Enum):
    """Semantic categories for document layout elements."""

    TITLE = "title"
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"
    CAPTION = "caption"
    LIST = "list"
    HEADER = "header"
    FOOTER = "footer"


class EdgeType(str, Enum):
    """Spatial / hierarchical relationships between layout elements."""

    ABOVE = "above"
    BELOW = "below"
    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    CONTAINS = "contains"


NODE_TYPES: List[str] = [nt.value for nt in NodeType]
EDGE_TYPES: List[str] = [et.value for et in EdgeType]

NUM_NODE_TYPES: int = len(NODE_TYPES)
NUM_EDGE_TYPES: int = len(EDGE_TYPES)
