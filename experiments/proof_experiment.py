"""
experiments/proof_experiment.py — Main experiment orchestrator for Vision-SARVAM.

Runs all four proof-of-concept metrics on a configurable slice of a document
layout dataset and produces:
  • A summary table printed to stdout (with pass/fail against targets).
  • A JSON results file saved to the output directory.

Usage
-----
    python -m experiments.proof_experiment --dataset publaynet --limit 1000
    python -m experiments.proof_experiment --dataset docbank --limit 500 --output results/

The orchestrator generates synthetic PageLayouts when no real dataset loader
is wired up yet (MVP mode).  Replace ``_load_pages_synthetic`` with real
dataset loading once data pipelines are ready.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ── Project imports ──────────────────────────────────────────────────────
from data.loader import BBox, LayoutElement, PageLayout
from graph.builder import DocumentGraph, build_graph
from metrics.compression import compute_compression, compute_compression_stats
from metrics.information_retention import (
    evaluate_information_retention,
    evaluate_information_retention_detailed,
)
from metrics.ocr_area import compute_ocr_area_ratio, compute_ocr_area_stats
from metrics.reading_order import compute_reading_order_score, compute_reading_order_stats


# =====================================================================
#  Configuration & target thresholds
# =====================================================================

# These are the "pass" thresholds for each metric.
# Chosen to reflect realistic expectations for the graph-first approach.
METRIC_TARGETS: Dict[str, Dict[str, float]] = {
    "compression_ratio_mean": {"target": 50.0, "direction": "higher_is_better"},
    "reading_order_tau_mean": {"target": 0.6, "direction": "higher_is_better"},
    "ocr_area_ratio_mean": {"target": 0.0, "direction": "informational"},  # no pass/fail
    "information_retention_acc": {"target": 0.5, "direction": "higher_is_better"},
}

DEFAULT_OUTPUT_DIR = "output"


# =====================================================================
#  Synthetic data generator (for MVP / no-real-dataset mode)
# =====================================================================

def _generate_synthetic_page(
    page_idx: int,
    dataset: str = "synthetic",
    page_width: float = 612.0,
    page_height: float = 792.0,
    rng: random.Random | None = None,
) -> PageLayout:
    """Create a realistic-ish synthetic page layout for testing.

    Generates between 5 and 30 layout elements with plausible bounding boxes
    and category distributions inspired by real academic papers.
    """
    if rng is None:
        rng = random.Random(page_idx)

    # Category weights (roughly matching PubLayNet distribution)
    categories = ["text", "text", "text", "text", "title", "figure", "table", "list", "caption"]

    n_elements = rng.randint(5, 30)
    elements: List[LayoutElement] = []

    for eid in range(n_elements):
        cat = rng.choice(categories)

        # Generate plausible bounding boxes
        w = rng.uniform(50, page_width * 0.9)
        h = rng.uniform(20, page_height * 0.3)
        x = rng.uniform(0, max(page_width - w, 1))
        y = rng.uniform(0, max(page_height - h, 1))

        bbox = BBox(x=x, y=y, w=w, h=h)
        elements.append(
            LayoutElement(id=eid, category=cat, bbox=bbox, area=bbox.area)
        )

    return PageLayout(
        page_id=f"{dataset}_{page_idx:06d}",
        width=page_width,
        height=page_height,
        elements=elements,
        dataset=dataset,
    )


def load_pages(
    dataset: str,
    limit: int,
    seed: int = 42,
) -> List[PageLayout]:
    """Load pages from a dataset.

    Currently generates synthetic data.  Replace the body of this function
    with real dataset loading (e.g. COCO-format JSON for PubLayNet) when
    data pipelines are ready.

    Parameters
    ----------
    dataset : str
        Dataset name (``'publaynet'``, ``'docbank'``, ``'synthetic'``).
    limit : int
        Maximum number of pages to load.
    seed : int
        Random seed for synthetic generation.

    Returns
    -------
    list[PageLayout]
    """
    from data.loader import load_all_pages
    try:
        return load_all_pages(dataset, split="val", limit=limit)
    except FileNotFoundError:
        print(f"Warning: Dataset {dataset} not found locally. Falling back to synthetic data.")
        rng = random.Random(seed)
        return [_generate_synthetic_page(i, dataset=dataset, rng=rng) for i in range(limit)]


# =====================================================================
#  Experiment runner
# =====================================================================

def run_experiment(
    dataset: str,
    limit: int,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    seed: int = 42,
    verbose: bool = True,
) -> Dict[str, object]:
    """Run the full proof-of-concept experiment.

    Steps
    -----
    1. Load (or generate) pages.
    2. Build document graphs.
    3. Compute all four metrics.
    4. Print a summary table.
    5. Save results to JSON.

    Returns
    -------
    dict
        Complete results dictionary (also saved to disk).
    """
    t_start = time.time()

    # ── 1. Load pages ────────────────────────────────────────────────
    if verbose:
        print(f"[1/4] Loading pages from '{dataset}' (limit={limit})…")
    pages = load_pages(dataset, limit, seed=seed)
    if verbose:
        print(f"       → {len(pages)} pages loaded.")

    # ── 2. Build graphs ──────────────────────────────────────────────
    if verbose:
        print("[2/4] Building document graphs…")
    graphs: List[DocumentGraph] = []
    for page in pages:
        g = build_graph(page)
        graphs.append(g)
    if verbose:
        total_nodes = sum(g.num_nodes for g in graphs)
        total_edges = sum(g.num_edges for g in graphs)
        print(f"       → {len(graphs)} graphs, {total_nodes} nodes, {total_edges} edges.")

    # ── 3. Compute metrics ───────────────────────────────────────────
    if verbose:
        print("[3/4] Computing metrics…")

    # 3a. Compression ratio
    compression_stats = compute_compression_stats(graphs)
    if verbose:
        print(f"       Compression  — mean: {compression_stats['mean']:.1f}×")

    # 3b. Reading order
    reading_order_stats = compute_reading_order_stats(graphs, pages)
    if verbose:
        print(f"       Reading order — mean τ: {reading_order_stats['mean']:.3f}")

    # 3c. OCR area ratio
    ocr_stats = compute_ocr_area_stats(pages)
    if verbose:
        print(f"       OCR area      — mean: {ocr_stats['mean']:.3f}")

    # 3d. Information retention
    if verbose:
        print("       Training information-retention classifier…")
    info_retention = evaluate_information_retention_detailed(graphs)
    if verbose:
        print(f"       Info retention — accuracy: {info_retention['accuracy']:.3f}")

    # ── 4. Assemble results ──────────────────────────────────────────
    elapsed = time.time() - t_start

    results: Dict[str, object] = {
        "dataset": dataset,
        "num_pages": len(pages),
        "elapsed_seconds": round(elapsed, 2),
        "metrics": {
            "compression": compression_stats,
            "reading_order": reading_order_stats,
            "ocr_area": ocr_stats,
            "information_retention": {
                "accuracy": info_retention["accuracy"],
                "n_train": info_retention["n_train"],
                "n_test": info_retention["n_test"],
                "per_class": info_retention.get("per_class", {}),
            },
        },
        "targets": METRIC_TARGETS,
        "pass_fail": {},
    }

    # ── 5. Pass / fail evaluation ────────────────────────────────────
    metric_values = {
        "compression_ratio_mean": compression_stats["mean"],
        "reading_order_tau_mean": reading_order_stats["mean"],
        "ocr_area_ratio_mean": ocr_stats["mean"],
        "information_retention_acc": info_retention["accuracy"],
    }

    for name, spec in METRIC_TARGETS.items():
        val = metric_values.get(name, 0.0)
        if spec["direction"] == "higher_is_better":
            passed = val >= spec["target"]
        elif spec["direction"] == "lower_is_better":
            passed = val <= spec["target"]
        else:
            passed = None  # informational, no pass/fail
        results["pass_fail"][name] = {
            "value": round(val, 4) if isinstance(val, float) else val,
            "target": spec["target"],
            "passed": passed,
        }

    # ── 6. Print summary table ───────────────────────────────────────
    if verbose:
        print()
        _print_summary_table(results)

    # ── 7. Save to disk ─────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    result_path = os.path.join(output_dir, f"results_{dataset}_{limit}.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    if verbose:
        print(f"\nResults saved to {result_path}")

    return results


# =====================================================================
#  Pretty-printing
# =====================================================================

def _print_summary_table(results: Dict[str, object]) -> None:
    """Print a formatted summary table to stdout."""
    pf = results["pass_fail"]
    header = f"{'Metric':<32} {'Value':>10} {'Target':>10} {'Status':>8}"
    sep = "─" * len(header)

    print(sep)
    print(f"  Vision-SARVAM Proof-of-Concept Results")
    print(f"  Dataset: {results['dataset']}  |  Pages: {results['num_pages']}  |  Time: {results['elapsed_seconds']}s")
    print(sep)
    print(header)
    print(sep)

    for name, info in pf.items():
        val_str = f"{info['value']:.4f}" if isinstance(info["value"], float) else str(info["value"])
        tgt_str = f"{info['target']:.2f}" if isinstance(info["target"], float) else str(info["target"])
        if info["passed"] is None:
            status = "  INFO"
        elif info["passed"]:
            status = "  ✅ PASS"
        else:
            status = "  ❌ FAIL"
        print(f"  {name:<30} {val_str:>10} {tgt_str:>10} {status}")

    print(sep)


# =====================================================================
#  CLI entry-point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Vision-SARVAM proof-of-concept experiment runner.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
    python -m experiments.proof_experiment --dataset publaynet --limit 1000
    python -m experiments.proof_experiment --dataset docbank --limit 500 --output results/
        """,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="publaynet",
        help="Dataset name (default: publaynet). Currently uses synthetic data.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of pages to process (default: 100).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )

    args = parser.parse_args()

    run_experiment(
        dataset=args.dataset,
        limit=args.limit,
        output_dir=args.output,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
