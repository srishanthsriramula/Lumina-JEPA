# Vision-SARVAM

**Structured Abstraction and Representation for Vision-Augmented Multimodal understanding**

A graph-first document AI system that replaces flat visual token grids with
compact, semantically rich graph representations of document pages.

---

## Research Hypothesis

> A structured graph representation of a document page — where **nodes** are
> layout elements (titles, paragraphs, tables, figures …) and **edges** encode
> spatial/hierarchical relationships — retains **≥ 95 %** of the semantic
> information available in the raw pixel grid while requiring **~20× fewer
> tokens** than a standard Vision Transformer patch embedding.

If validated, this enables document-understanding models that are dramatically
cheaper to run, easier to interpret, and better at preserving reading order.

---

## Target Architecture

```
 ┌─────────────────────────────────────────────────────────────────┐
 │                     Document Page (image)                       │
 └────────────────────────────┬────────────────────────────────────┘
                              │
                   ┌──────────▼──────────┐
                   │  Layout Detection   │
                   │  (Faster-RCNN /     │
                   │   Mask-RCNN)        │
                   └──────────┬──────────┘
                              │  bounding boxes + class labels
                   ┌──────────▼──────────┐
                   │  Graph Construction │
                   │  ─ nodes = regions  │
                   │  ─ edges = spatial  │
                   │    relations        │
                   └──────────┬──────────┘
                              │  Document Graph  (NetworkX / PyG)
            ┌─────────────────┼─────────────────┐
            │                 │                 │
   ┌────────▼───────┐ ┌──────▼───────┐ ┌───────▼────────┐
   │ Compression    │ │ Reading-Order│ │ Node-Type      │
   │ Metric         │ │ Recovery     │ │ Prediction     │
   │ (token count)  │ │ (Kendall τ)  │ │ (F1 score)     │
   └────────────────┘ └──────────────┘ └────────────────┘
```

---

## Phase 1 — Proof of Concept

**Goal:** demonstrate that the graph construction pipeline produces meaningful
representations on real document-layout datasets.

| Step | Description |
|------|-------------|
| **1.1** | Download PubLayNet / DocLayNet subsets via FiftyOne |
| **1.2** | Build document graphs from ground-truth bounding boxes |
| **1.3** | Measure compression ratio vs. 4 096-token ViT baseline |
| **1.4** | Evaluate reading-order recovery (Kendall τ) |
| **1.5** | Train a simple GNN for node-type classification (F1) |
| **1.6** | Compute OCR-area reduction guided by graph regions |

---

## Success Criteria

| Metric | Target | Rationale |
|--------|--------|-----------|
| **Compression ratio** | ≥ 20× | Graph tokens ≪ 4 096 ViT patches |
| **Reading-order τ** | ≥ 0.95 | Near-perfect order from spatial edges |
| **Node-type F1** | ≥ 0.90 | Graph features alone predict layout class |
| **OCR area reduction** | ≤ 30 % of page | Only text-bearing regions sent to OCR |

---

## Repository Layout

```
vision-SARVAM/
├── config.py            # Centralised configuration & constants
├── requirements.txt     # Python dependencies
├── README.md            # ← you are here
├── data/                # Dataset loaders & preprocessing
│   └── __init__.py
├── graph/               # Graph construction & algorithms
│   └── __init__.py
├── metrics/             # Evaluation metric implementations
│   └── __init__.py
└── experiments/         # Runnable experiment scripts
    └── __init__.py
```

---

## Quick Start

```bash
# 1. Clone
git clone <repo-url> vision-SARVAM && cd vision-SARVAM

# 2. Create environment
python -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify config loads
python -c "import config; print(config.VISUAL_TOKEN_COUNT)"
# → 4096
```

---

## License

Research-only — licence TBD.
