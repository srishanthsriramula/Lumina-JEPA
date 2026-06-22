#!/usr/bin/env python3
"""
data/loader.py — Unified COCO-format data loader for Vision-SARVAM.

Loads PubLayNet and DocLayNet annotations into a common set of dataclasses
(`BBox`, `LayoutElement`, `PageLayout`) so that downstream graph-construction
and analysis code doesn't need to know which dataset it's working with.

Design decisions
----------------
* **Category normalisation** — both datasets use different label sets; we map
  them to a shared vocabulary so models can be trained cross-dataset.  The
  canonical types are:

      text, title, list, table, figure, caption, footnote, formula,
      page-header, page-footer, other

  PubLayNet's 5 categories map 1-to-1.  DocLayNet's 11 categories are folded
  where sensible (e.g. ``section-header`` → ``title``, ``picture`` → ``figure``,
  ``list-item`` → ``list``).

* **Lazy JSON loading** — annotation files can be large (~1 GB for PubLayNet
  val); we load once per call and build lookup dicts for efficient grouping by
  image.

* **Generator-based iteration** — `load_pages` yields `PageLayout` objects one
  at a time, keeping memory usage proportional to a single page, not the whole
  dataset.

Usage
-----
    from data.loader import load_pages

    for page in load_pages("publaynet", limit=100):
        print(page.page_id, len(page.elements))
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DOCLAYNET_DIR, PUBLAYNET_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Dataclasses ─────────────────────────────────────────────────────────────


@dataclass
class BBox:
    """Axis-aligned bounding box in absolute page coordinates.

    Coordinates follow the PDF / image convention:
    origin at top-left, x grows right, y grows down.
    COCO format: (x, y) is the **top-left** corner, (w, h) are width and height.
    """

    x: float  # top-left x
    y: float  # top-left y
    w: float  # width
    h: float  # height

    @property
    def x2(self) -> float:
        """Right edge coordinate."""
        return self.x + self.w

    @property
    def y2(self) -> float:
        """Bottom edge coordinate."""
        return self.y + self.h

    @property
    def cx(self) -> float:
        """Horizontal centre."""
        return self.x + self.w / 2.0

    @property
    def cy(self) -> float:
        """Vertical centre."""
        return self.y + self.h / 2.0

    @property
    def area(self) -> float:
        return self.w * self.h


@dataclass
class LayoutElement:
    """A single annotated region on a document page.

    Attributes:
        id:       Unique integer identifier within the page.
        category: Semantic label, normalised to the shared vocabulary
                  (e.g. 'title', 'text', 'figure', 'table').
        bbox:     Bounding box in absolute page coordinates.
        area:     Pre-computed area (== bbox.w * bbox.h for axis-aligned boxes).
    """

    id: int
    category: str
    bbox: BBox
    area: float


@dataclass
class PageLayout:
    """Complete layout annotation for one document page.

    Attributes:
        page_id:  Unique string identifier (e.g. image filename stem).
        width:    Page width in pixels / points.
        height:   Page height in pixels / points.
        elements: All annotated layout elements on this page.
        dataset:  Source dataset name ('publaynet' or 'doclaynet').
    """

    page_id: str
    width: float
    height: float
    elements: List[LayoutElement] = field(default_factory=list)
    dataset: str = ""


# ── Category normalisation ──────────────────────────────────────────────────
# Canonical label set shared across both datasets.  Any category that doesn't
# appear in the mapping falls through to "other".

# PubLayNet categories (from COCO JSON "categories" list):
#   1: text,  2: title,  3: list,  4: table,  5: figure
PUBLAYNET_CATEGORY_MAP: Dict[str, str] = {
    "text": "text",
    "title": "title",
    "list": "list",
    "table": "table",
    "figure": "figure",
}

# DocLayNet categories (from COCO JSON "categories" list):
#   1:  Caption        →  caption
#   2:  Footnote       →  footnote
#   3:  Formula        →  formula
#   4:  List-item      →  list
#   5:  Page-footer    →  page-footer
#   6:  Page-header    →  page-header
#   7:  Picture        →  figure
#   8:  Section-header →  title
#   9:  Table          →  table
#   10: Text           →  text
#   11: Title          →  title
DOCLAYNET_CATEGORY_MAP: Dict[str, str] = {
    "caption": "caption",
    "footnote": "footnote",
    "formula": "formula",
    "list-item": "list",
    "page-footer": "page-footer",
    "page-header": "page-header",
    "picture": "figure",
    "section-header": "title",
    "table": "table",
    "text": "text",
    "title": "title",
}


def _normalise_category(raw: str, dataset: str) -> str:
    """Map a raw category name to the shared canonical label."""
    key = raw.strip().lower()
    if dataset == "publaynet":
        return PUBLAYNET_CATEGORY_MAP.get(key, "other")
    elif dataset == "doclaynet":
        return DOCLAYNET_CATEGORY_MAP.get(key, "other")
    return key


# ── COCO JSON parser ───────────────────────────────────────────────────────


def _load_coco_json(path: Path) -> dict:
    """Load and return a COCO-format annotation JSON."""
    logger.info("Loading COCO JSON: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(
        "  images=%d  annotations=%d  categories=%d",
        len(data.get("images", [])),
        len(data.get("annotations", [])),
        len(data.get("categories", [])),
    )
    return data


def _parse_coco(
    coco: dict,
    dataset: str,
) -> Generator[PageLayout, None, None]:
    """Yield ``PageLayout`` objects from a parsed COCO dict.

    Parameters
    ----------
    coco : dict
        Parsed COCO annotation JSON with ``images``, ``annotations``, and
        ``categories`` top-level keys.
    dataset : str
        ``'publaynet'`` or ``'doclaynet'`` — controls category normalisation.

    Yields
    ------
    PageLayout
        One per image entry, with all matching annotations attached.
    """
    # Build category_id → category_name lookup.
    cat_id_to_name: Dict[int, str] = {
        cat["id"]: cat["name"] for cat in coco.get("categories", [])
    }

    # Group annotations by image_id for O(1) lookup per image.
    anns_by_image: Dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    for img in coco.get("images", []):
        image_id: int = img["id"]
        width: float = float(img.get("width", 0))
        height: float = float(img.get("height", 0))
        file_name: str = img.get("file_name", str(image_id))

        # Use the filename (without extension) as page_id for readability.
        page_id = Path(file_name).stem

        elements: List[LayoutElement] = []
        for ann in anns_by_image.get(image_id, []):
            # COCO bbox is [x, y, w, h] — exactly our BBox layout.
            bx, by, bw, bh = ann["bbox"]
            raw_cat = cat_id_to_name.get(ann["category_id"], "unknown")
            norm_cat = _normalise_category(raw_cat, dataset)
            area = float(ann.get("area", bw * bh))

            elements.append(
                LayoutElement(
                    id=int(ann["id"]),
                    category=norm_cat,
                    bbox=BBox(x=bx, y=by, w=bw, h=bh),
                    area=area,
                )
            )

        yield PageLayout(
            page_id=page_id,
            width=width,
            height=height,
            elements=elements,
            dataset=dataset,
        )


# ── Annotation file discovery ──────────────────────────────────────────────


def _find_annotation_file(dataset: str, split: str = "val") -> Path:
    """Locate the COCO annotation JSON on disk for the given dataset/split.

    Raises ``FileNotFoundError`` with a helpful message if the file is
    missing (the user probably needs to run ``data/download.py`` first).
    """
    if dataset == "publaynet":
        # HF-mirror download stores e.g. publaynet/val.json
        candidates = [
            PUBLAYNET_DIR / f"{split}.json",
            PUBLAYNET_DIR / "publaynet" / f"{split}.json",
        ]
    elif dataset == "doclaynet":
        # DocLayNet ZIP extracts to doclaynet/COCO/{split}.json
        candidates = [
            DOCLAYNET_DIR / "COCO" / f"{split}.json",
            DOCLAYNET_DIR / f"{split}.json",
        ]
    else:
        raise ValueError(
            f"Unknown dataset {dataset!r}. Use 'publaynet' or 'doclaynet'."
        )

    for p in candidates:
        if p.exists():
            return p

    raise FileNotFoundError(
        f"Annotation file for {dataset}/{split} not found.\n"
        f"  Searched: {[str(c) for c in candidates]}\n"
        f"  Run `python -m data.download --dataset {dataset}` first."
    )


# ── Public API ──────────────────────────────────────────────────────────────


def load_pages(
    dataset_name: str,
    split: str = "val",
    limit: Optional[int] = None,
) -> Generator[PageLayout, None, None]:
    """Iterate over page layouts from a COCO-annotated dataset.

    Parameters
    ----------
    dataset_name : str
        ``'publaynet'`` or ``'doclaynet'``.
    split : str
        Annotation split — typically ``'val'`` or ``'test'``.
    limit : int, optional
        If given, stop after yielding this many pages.  Useful for quick
        prototyping runs.

    Yields
    ------
    PageLayout
        One per document page, with normalised category labels.

    Examples
    --------
    >>> pages = list(load_pages("publaynet", limit=5))
    >>> len(pages)
    5
    >>> pages[0].dataset
    'publaynet'
    """
    ann_path = _find_annotation_file(dataset_name, split)
    coco = _load_coco_json(ann_path)

    count = 0
    for page in _parse_coco(coco, dataset=dataset_name):
        yield page
        count += 1
        if limit is not None and count >= limit:
            return

    logger.info("Loaded %d pages from %s/%s", count, dataset_name, split)


def load_all_pages(
    dataset_name: str,
    split: str = "val",
    limit: Optional[int] = None,
) -> List[PageLayout]:
    """Eagerly load all pages into a list.

    Convenience wrapper around :func:`load_pages` for interactive / notebook
    use where a list is more practical than a generator.
    """
    return list(load_pages(dataset_name, split=split, limit=limit))


# ── Quick sanity-check CLI ──────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(description="Inspect a layout dataset.")
    parser.add_argument(
        "dataset",
        choices=["publaynet", "doclaynet"],
        help="Which dataset to load.",
    )
    parser.add_argument(
        "--split",
        default="val",
        help="Annotation split (default: val).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max pages to print (default: 5).",
    )
    args = parser.parse_args()

    for page in load_pages(args.dataset, split=args.split, limit=args.limit):
        print(f"\n{'─' * 60}")
        print(
            f"Page: {page.page_id}  "
            f"({page.width}×{page.height})  [{page.dataset}]"
        )
        print(f"  Elements ({len(page.elements)}):")
        for el in page.elements:
            print(
                f"    [{el.id:6d}] {el.category:14s}  "
                f"bbox=({el.bbox.x:.1f}, {el.bbox.y:.1f}, "
                f"{el.bbox.w:.1f}, {el.bbox.h:.1f})  "
                f"area={el.area:.0f}"
            )
