"""
metrics/information_retention.py — Missing-node-type prediction metric.

Motivation
----------
A good graph representation should encode enough *context* around each node
that a masked node's category can be predicted from its neighbours alone.
This is the graph analogue of masked-language-modelling: if we hide one
node's label, can the surrounding structure (neighbour types, relative
positions, edge types) tell us what it was?

For the MVP we avoid a full GNN and instead build a tabular feature vector
for each node from its 1-hop neighbourhood, then train a lightweight
classifier (sklearn RandomForest or MLP).

Feature vector for each node
----------------------------
1. **Neighbour type histogram** — one-hot counts of each category among
   the node's direct neighbours (in + out edges).
2. **Relative position stats** — mean Δx, mean Δy, mean distance to
   neighbours (normalised by page diagonal).
3. **Edge-type counts** — how many spatial vs. reading-order edges.
4. **Node degree** — in-degree + out-degree.

The target is the node's own category (multi-class classification).
Accuracy on a held-out 20 % test split is the **information retention score**.
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from config import NODE_TYPES as ALL_CATEGORIES
from graph.builder import DocumentGraph


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_node_features(
    dg: DocumentGraph,
    category_list: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    """Build feature matrix X and label vector y for all nodes in one graph.

    Parameters
    ----------
    dg : DocumentGraph
        A single document graph.
    category_list : list[str]
        Canonical ordered list of category names (used for one-hot encoding).

    Returns
    -------
    X : ndarray, shape (n_nodes, n_features)
    y : ndarray, shape (n_nodes,)  — string category labels
    """
    G = dg.graph
    cat_to_idx = {c: i for i, c in enumerate(category_list)}
    n_cats = len(category_list)

    X_rows: List[np.ndarray] = []
    y_labels: List[str] = []

    for node_id in G.nodes:
        attrs = G.nodes[node_id]
        category = attrs.get("type", "unknown")
        bbox = attrs.get("bbox", (0.0, 0.0, 0.0, 0.0))
        cx = bbox[0] + bbox[2] / 2.0
        cy = bbox[1] + bbox[3] / 2.0

        # ---- Neighbour features ----
        # Collect all neighbours (predecessors + successors in the DiGraph)
        neighbours = set(G.predecessors(node_id)) | set(G.successors(node_id))

        # 1) Neighbour-type histogram
        neigh_hist = np.zeros(n_cats, dtype=np.float32)
        for nid in neighbours:
            ncat = G.nodes[nid].get("type", "unknown")
            idx = cat_to_idx.get(ncat)
            if idx is not None:
                neigh_hist[idx] += 1.0

        # 2) Relative position statistics
        dx_list, dy_list, dist_list = [], [], []
        for nid in neighbours:
            n_bbox = G.nodes[nid].get("bbox", (0.0, 0.0, 0.0, 0.0))
            ncx = n_bbox[0] + n_bbox[2] / 2.0
            ncy = n_bbox[1] + n_bbox[3] / 2.0
            dx = ncx - cx
            dy = ncy - cy
            dx_list.append(dx)
            dy_list.append(dy)
            dist_list.append(np.sqrt(dx ** 2 + dy ** 2))

        if dx_list:
            rel_pos = np.array(
                [np.mean(dx_list), np.mean(dy_list), np.mean(dist_list)],
                dtype=np.float32,
            )
        else:
            rel_pos = np.zeros(3, dtype=np.float32)

        # 3) Edge-type counts (spatial vs reading_order)
        spatial_count = 0
        reading_count = 0
        for _, _, edata in G.edges(node_id, data=True):
            etype = edata.get("edge_type", "")
            if etype == "spatial":
                spatial_count += 1
            elif etype == "reading_order":
                reading_count += 1
        for _, _, edata in G.in_edges(node_id, data=True):
            etype = edata.get("edge_type", "")
            if etype == "spatial":
                spatial_count += 1
            elif etype == "reading_order":
                reading_count += 1
        edge_feats = np.array(
            [spatial_count, reading_count], dtype=np.float32
        )

        # 4) Degree
        degree = np.array(
            [G.in_degree(node_id), G.out_degree(node_id)], dtype=np.float32
        )

        # Concatenate
        feat = np.concatenate([neigh_hist, rel_pos, edge_feats, degree])
        X_rows.append(feat)
        y_labels.append(category)

    if not X_rows:
        return np.empty((0, n_cats + 7)), np.empty(0, dtype=object)

    return np.vstack(X_rows), np.array(y_labels, dtype=object)


def _extract_features_corpus(
    graphs: List[DocumentGraph],
    category_list: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features from all graphs and stack into single arrays."""
    if category_list is None:
        category_list = ALL_CATEGORIES

    all_X, all_y = [], []
    for dg in graphs:
        X, y = _extract_node_features(dg, category_list)
        if X.shape[0] > 0:
            all_X.append(X)
            all_y.append(y)

    if not all_X:
        return np.empty((0, len(category_list) + 7)), np.empty(0, dtype=object)

    return np.vstack(all_X), np.concatenate(all_y)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def evaluate_information_retention(
    graphs: List[DocumentGraph],
    classifier: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    category_list: Optional[List[str]] = None,
) -> float:
    """Train a classifier to predict masked node types; return accuracy.

    This simulates "masking" each node's own category and predicting it
    purely from neighbourhood context.  In practice we train on 80 %
    of nodes and evaluate accuracy on the remaining 20 %.

    Parameters
    ----------
    graphs : list[DocumentGraph]
        Corpus of document graphs.
    classifier : str
        ``'random_forest'`` or ``'mlp'``.
    test_size : float
        Fraction of nodes used for evaluation.
    random_state : int
        Seed for reproducibility.
    category_list : list[str] | None
        Canonical category ordering; defaults to ``ALL_CATEGORIES``.

    Returns
    -------
    float
        Classification accuracy on the test split, in [0, 1].
        Returns 0.0 if there are too few samples or only one class.
    """
    if category_list is None:
        category_list = ALL_CATEGORIES

    X, y = _extract_features_corpus(graphs, category_list)

    if X.shape[0] < 5:
        warnings.warn(
            "Too few nodes to evaluate information retention "
            f"({X.shape[0]} nodes). Returning 0.0."
        )
        return 0.0

    # Need at least 2 classes
    unique_classes = np.unique(y)
    if len(unique_classes) < 2:
        warnings.warn(
            "Only one category present in the corpus. Returning 0.0."
        )
        return 0.0

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state
    )

    if len(X_test) == 0:
        return 0.0

    # Train classifier
    if classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=random_state,
            early_stopping=True,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress convergence warnings for MVP
        clf.fit(X_train, y_train)

    accuracy = float(clf.score(X_test, y_test))
    return accuracy


def evaluate_information_retention_detailed(
    graphs: List[DocumentGraph],
    classifier: str = "random_forest",
    test_size: float = 0.2,
    random_state: int = 42,
    category_list: Optional[List[str]] = None,
) -> Dict[str, object]:
    """Like :func:`evaluate_information_retention` but returns per-class detail.

    Returns
    -------
    dict
        ``accuracy``, ``per_class`` (precision / recall / f1 per category),
        ``n_train``, ``n_test``.
    """
    from sklearn.metrics import classification_report

    if category_list is None:
        category_list = ALL_CATEGORIES

    X, y = _extract_features_corpus(graphs, category_list)

    if X.shape[0] < 5 or len(np.unique(y)) < 2:
        return {"accuracy": 0.0, "per_class": {}, "n_train": 0, "n_test": 0}

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=test_size, random_state=random_state
    )

    if classifier == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(64, 32),
            max_iter=300,
            random_state=random_state,
            early_stopping=True,
        )
    else:
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=random_state,
            n_jobs=-1,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)

    accuracy = float(clf.score(X_test, y_test))

    y_pred = clf.predict(X_test)
    report = classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "per_class": {
            k: v for k, v in report.items()
            if k not in ("accuracy", "macro avg", "weighted avg")
        },
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
