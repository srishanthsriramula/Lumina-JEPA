"""
experiments/visualize.py — Visualization dashboard for Vision-SARVAM metrics.

Generates publication-quality plots for all four proof-of-concept metrics
and a combined summary dashboard.  Uses matplotlib + seaborn.

Usage
-----
    # After running the experiment:
    python -m experiments.visualize --results output/results_publaynet_1000.json

    # Or generate plots directly from data:
    python -m experiments.visualize --dataset publaynet --limit 1000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

try:
    import seaborn as sns
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from data.loader import PageLayout
from config import NODE_TYPES as ALL_CATEGORIES
from graph.builder import DocumentGraph, build_graph
from metrics.compression import compute_compression
from metrics.ocr_area import compute_ocr_area_ratio
from metrics.reading_order import compute_reading_order_score


# =====================================================================
#  Colour palette & style constants
# =====================================================================
COLORS = {
    "primary": "#2563EB",     # blue-600
    "secondary": "#7C3AED",   # violet-600
    "accent": "#059669",      # emerald-600
    "warning": "#D97706",     # amber-600
    "danger": "#DC2626",      # red-600
    "neutral": "#6B7280",     # gray-500
    "pass": "#10B981",        # emerald-500
    "fail": "#EF4444",        # red-500
}

DEFAULT_OUTPUT_DIR = "output/plots"


# =====================================================================
#  Individual metric plots
# =====================================================================

def plot_compression_distribution(
    graphs: List[DocumentGraph],
    output_path: str,
    visual_tokens: int = 4096,
) -> str:
    """Histogram of compression ratios across all pages.

    Returns the path to the saved figure.
    """
    ratios = [compute_compression(g, visual_tokens) for g in graphs]
    finite = [r for r in ratios if r != float("inf")]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(finite, bins=40, color=COLORS["primary"], edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(finite), color=COLORS["danger"], linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(finite):.1f}×")
    ax.axvline(np.median(finite), color=COLORS["warning"], linestyle="-.", linewidth=2,
               label=f"Median = {np.median(finite):.1f}×")

    ax.set_xlabel("Compression Ratio (visual_tokens / graph_nodes)")
    ax.set_ylabel("Number of Pages")
    ax.set_title("Graph Compression Ratio Distribution")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_reading_order_distribution(
    graphs: List[DocumentGraph],
    pages: List[PageLayout],
    output_path: str,
) -> str:
    """Histogram of Kendall's Tau reading-order scores."""
    scores = [compute_reading_order_score(g, p) for g, p in zip(graphs, pages)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=40, color=COLORS["secondary"], edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(scores), color=COLORS["danger"], linestyle="--", linewidth=2,
               label=f"Mean τ = {np.mean(scores):.3f}")

    ax.set_xlabel("Kendall's Tau (reading-order agreement)")
    ax.set_ylabel("Number of Pages")
    ax.set_title("Reading Order Recovery — Kendall's Tau Distribution")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_ocr_area_distribution(
    pages: List[PageLayout],
    output_path: str,
) -> str:
    """Histogram of OCR area ratios."""
    ratios = [compute_ocr_area_ratio(p) for p in pages]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(ratios, bins=40, color=COLORS["accent"], edgecolor="white", alpha=0.85)
    ax.axvline(np.mean(ratios), color=COLORS["danger"], linestyle="--", linewidth=2,
               label=f"Mean = {np.mean(ratios):.3f}")

    ax.set_xlabel("OCR Area Ratio (text area / page area)")
    ax.set_ylabel("Number of Pages")
    ax.set_title("OCR Area Ratio Distribution")
    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_node_type_frequency(
    graphs: List[DocumentGraph],
    output_path: str,
) -> str:
    """Bar chart of node-type frequencies across all pages."""
    # Aggregate counts
    total_counts: Dict[str, int] = {}
    for g in graphs:
        for cat, count in g.node_types.items():
            total_counts[cat] = total_counts.get(cat, 0) + count

    # Sort by frequency
    sorted_items = sorted(total_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_items]
    counts = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, counts, color=COLORS["primary"], edgecolor="white", alpha=0.85)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(counts) * 0.01,
            f"{count:,}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Layout Category")
    ax.set_ylabel("Total Count (across all pages)")
    ax.set_title("Node Type Frequency Distribution")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =====================================================================
#  Summary dashboard
# =====================================================================

def plot_summary_dashboard(
    results: Dict[str, object],
    output_path: str,
) -> str:
    """Combined dashboard showing all 4 metrics vs. targets.

    Parameters
    ----------
    results : dict
        The results dict produced by ``proof_experiment.run_experiment``.
    output_path : str
        Path to save the dashboard figure.
    """
    pf = results.get("pass_fail", {})

    metric_names = list(pf.keys())
    values = [pf[m]["value"] for m in metric_names]
    targets = [pf[m]["target"] for m in metric_names]
    passed = [pf[m]["passed"] for m in metric_names]

    # Pretty labels
    labels = {
        "compression_ratio_mean": "Compression\nRatio (mean)",
        "reading_order_tau_mean": "Reading Order\nτ (mean)",
        "ocr_area_ratio_mean": "OCR Area\nRatio (mean)",
        "information_retention_acc": "Info Retention\nAccuracy",
    }
    pretty_names = [labels.get(m, m) for m in metric_names]

    fig, axes = plt.subplots(1, len(metric_names), figsize=(4 * len(metric_names), 5))
    if len(metric_names) == 1:
        axes = [axes]

    for i, (ax, name, val, tgt, ok) in enumerate(
        zip(axes, pretty_names, values, targets, passed)
    ):
        # Determine colour
        if ok is None:
            color = COLORS["neutral"]
            status_text = "INFO"
        elif ok:
            color = COLORS["pass"]
            status_text = "PASS ✅"
        else:
            color = COLORS["fail"]
            status_text = "FAIL ❌"

        # Gauge-like horizontal bar
        ax.barh([0], [val], color=color, height=0.5, alpha=0.8, edgecolor="white")
        if ok is not None:
            ax.axvline(tgt, color=COLORS["danger"], linestyle="--", linewidth=2)
            ax.text(tgt, 0.35, f"Target: {tgt}", ha="center", va="bottom",
                    fontsize=8, color=COLORS["danger"])

        ax.set_xlim(0, max(val, tgt if tgt else val) * 1.3 or 1)
        ax.set_yticks([])
        ax.set_title(name, fontsize=11, fontweight="bold")
        ax.text(
            0.5, -0.15, f"{val:.4f}  ({status_text})",
            transform=ax.transAxes, ha="center", fontsize=10,
            fontweight="bold", color=color,
        )

    fig.suptitle(
        f"Vision-SARVAM — {results.get('dataset', '?')} ({results.get('num_pages', '?')} pages)",
        fontsize=14, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


# =====================================================================
#  Generate all plots from live data
# =====================================================================

def generate_all_plots(
    graphs: List[DocumentGraph],
    pages: List[PageLayout],
    results: Dict[str, object],
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> List[str]:
    """Generate the full suite of plots and return saved paths.

    Parameters
    ----------
    graphs : list[DocumentGraph]
        Document graphs for all pages.
    pages : list[PageLayout]
        Corresponding page layouts.
    results : dict
        Results dict from ``proof_experiment.run_experiment()``.
    output_dir : str
        Directory to save plots.

    Returns
    -------
    list[str]
        Paths to all saved figures.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    saved.append(plot_compression_distribution(
        graphs, os.path.join(output_dir, "compression_distribution.png"),
    ))
    saved.append(plot_reading_order_distribution(
        graphs, pages, os.path.join(output_dir, "reading_order_distribution.png"),
    ))
    saved.append(plot_ocr_area_distribution(
        pages, os.path.join(output_dir, "ocr_area_distribution.png"),
    ))
    saved.append(plot_node_type_frequency(
        graphs, os.path.join(output_dir, "node_type_frequency.png"),
    ))
    saved.append(plot_summary_dashboard(
        results, os.path.join(output_dir, "summary_dashboard.png"),
    ))

    return saved


# =====================================================================
#  CLI: generate plots from a saved results JSON
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visualizations for Vision-SARVAM experiment results.",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--results",
        type=str,
        help="Path to a results JSON file from proof_experiment.",
    )
    group.add_argument(
        "--dataset",
        type=str,
        help="Generate fresh data and plots (synthetic mode).",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if args.results:
        # Load pre-computed results and regenerate dashboard only
        with open(args.results) as f:
            results = json.load(f)
        os.makedirs(args.output, exist_ok=True)
        path = plot_summary_dashboard(
            results, os.path.join(args.output, "summary_dashboard.png")
        )
        print(f"Dashboard saved to {path}")

    else:
        # Run experiment fresh, then generate all plots
        from experiments.proof_experiment import load_pages, run_experiment

        print(f"Running experiment on '{args.dataset}' (limit={args.limit})…")
        results = run_experiment(
            dataset=args.dataset,
            limit=args.limit,
            output_dir=args.output,
            seed=args.seed,
        )

        # Rebuild graphs + pages for per-element plots
        pages = load_pages(args.dataset, args.limit, seed=args.seed)
        graphs = [build_document_graph(p) for p in pages]

        plot_dir = os.path.join(args.output, "plots")
        paths = generate_all_plots(graphs, pages, results, output_dir=plot_dir)
        print(f"\n{len(paths)} plots saved to {plot_dir}/:")
        for p in paths:
            print(f"  • {p}")


if __name__ == "__main__":
    main()
