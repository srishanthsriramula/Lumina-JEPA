#!/usr/bin/env python3
"""
data/download.py — Dataset download script for Vision-SARVAM.

Downloads PubLayNet and DocLayNet document layout datasets in COCO annotation
format.  For MVP speed, only validation/test splits are fetched (much smaller
than full training sets).

Design decisions
----------------
* **Resumable downloads** — we stream to a `*.part` temp file and use HTTP
  `Range` headers so a killed download can pick up where it left off.
* **Atomic rename** — the `.part` file is renamed to the final name only after
  the download completes, so partial files never masquerade as complete ones.
* **ZIP extraction** — DocLayNet ships as a ZIP; PubLayNet annotations ship as
  gzipped JSON.  We handle both transparently.

Datasets
--------
PubLayNet  (https://github.com/ibm-aur-nlp/PubLayNet)
  - COCO-format annotations: text, title, list, table, figure
  - val split ~1 GB JSON (annotations only, no images needed for graph work)

DocLayNet  (https://github.com/DS4SD/DocLayNet)
  - COCO-format annotations: 11 categories including caption, footnote,
    formula, list-item, page-footer, page-header, picture, section-header,
    table, text, title
  - Core dataset ZIP (~1.8 GB) contains COCO/{train,val,test}.json + PNG/

Usage
-----
    python -m data.download                # download both datasets
    python -m data.download --dataset publaynet
    python -m data.download --dataset doclaynet
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import logging
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Resolve project root so we can import config regardless of how the script
# is invoked (standalone vs. module).
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_DIR, DOCLAYNET_DIR, PUBLAYNET_DIR  # noqa: E402

logger = logging.getLogger(__name__)

# ── Download manifest ───────────────────────────────────────────────────────
# Each entry is (url, relative_dest_within_dataset_dir, expected_sha256|None).
#
# PubLayNet: The original IBM DAX CDN hosts the full-dataset tar.gz files.
# For the MVP we only need the *val* annotations JSON (~92 MB gzipped).
# The COCO annotation file can be fetched directly from the DAX CDN.
#
# DocLayNet: The core dataset ZIP contains COCO/*.json + PNG images.
# We download the core ZIP and extract only the COCO annotation JSONs.
# ---------------------------------------------------------------------------

PUBLAYNET_URLS = {
    # val annotations — gzipped COCO JSON
    "val": (
        "https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/publaynet.tar.gz",
        # This is ~1 GB.  We'll provide an alternative smaller-scope approach below.
        None,  # sha256 not pinned for MVP
    ),
}

# Direct annotation JSON URLs extracted from the HuggingFace mirror, which
# lets us skip downloading the full 96-GB tar.
PUBLAYNET_ANNOTATION_URLS = {
    "val": (
        "https://huggingface.co/datasets/jordanparker6/publaynet/resolve/main/val.json",
        "val.json",
        None,
    ),
}

DOCLAYNET_CORE_URL = (
    "https://codait-cos-dax.s3.us.cloud-object-storage.appdomain.cloud/"
    "dax-doclaynet/1.0.0/DocLayNet_core.zip"
)

# ── Helpers ─────────────────────────────────────────────────────────────────


def _sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(
    url: str,
    dest: Path,
    *,
    expected_sha256: Optional[str] = None,
    desc: Optional[str] = None,
    chunk_size: int = 1 << 20,  # 1 MB
    timeout: int = 60,
) -> Path:
    """Download *url* to *dest* with resume support and a tqdm progress bar.

    Parameters
    ----------
    url : str
        Remote URL.
    dest : Path
        Local destination path (final name after download completes).
    expected_sha256 : str, optional
        If given, the file is verified after download.
    desc : str, optional
        Label for the progress bar.
    chunk_size : int
        Bytes per read iteration.
    timeout : int
        HTTP connect/read timeout in seconds.

    Returns
    -------
    Path
        The *dest* path on success.

    Raises
    ------
    RuntimeError
        On checksum mismatch.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already present and checksum matches (or no checksum).
    if dest.exists():
        if expected_sha256 is None or _sha256(dest) == expected_sha256:
            logger.info("Already downloaded: %s", dest)
            return dest
        logger.warning("Checksum mismatch for %s — re-downloading.", dest)

    part = dest.with_suffix(dest.suffix + ".part")
    headers: dict[str, str] = {}
    initial_size = 0

    # Resume support: if a .part file exists, request remaining bytes.
    if part.exists():
        initial_size = part.stat().st_size
        headers["Range"] = f"bytes={initial_size}-"
        logger.info("Resuming download from byte %d", initial_size)

    resp = requests.get(url, headers=headers, stream=True, timeout=timeout)

    # If the server doesn't support Range or returns 200 instead of 206,
    # we start from scratch.
    if resp.status_code == 200:
        initial_size = 0
        mode = "wb"
    elif resp.status_code == 206:
        mode = "ab"
    else:
        resp.raise_for_status()
        mode = "wb"  # unreachable, but keeps mypy happy

    total = int(resp.headers.get("content-length", 0)) + initial_size
    bar_desc = desc or dest.name

    with (
        open(part, mode) as f,
        tqdm(
            total=total or None,
            initial=initial_size,
            unit="B",
            unit_scale=True,
            desc=bar_desc,
        ) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    # Atomic rename.
    part.rename(dest)
    logger.info("Downloaded %s (%d bytes)", dest, dest.stat().st_size)

    # Verify checksum.
    if expected_sha256 is not None:
        actual = _sha256(dest)
        if actual != expected_sha256:
            raise RuntimeError(
                f"SHA-256 mismatch for {dest}:\n"
                f"  expected {expected_sha256}\n"
                f"  got      {actual}"
            )
    return dest


# ── PubLayNet download ──────────────────────────────────────────────────────


def download_publaynet(splits: Optional[list[str]] = None) -> None:
    """Download PubLayNet COCO annotation JSONs (val split by default).

    For the MVP we pull only the lightweight annotation JSON from a
    HuggingFace mirror, avoiding the full 96 GB tar.  The annotation file
    is all we need for graph-construction experiments — no images required.
    """
    splits = splits or ["val"]
    PUBLAYNET_DIR.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in PUBLAYNET_ANNOTATION_URLS:
            logger.warning(
                "Unknown PubLayNet split %r (available: %s)",
                split,
                list(PUBLAYNET_ANNOTATION_URLS.keys()),
            )
            continue

        url, filename, sha = PUBLAYNET_ANNOTATION_URLS[split]
        dest = PUBLAYNET_DIR / filename

        # If the URL points to a .gz file, we'd decompress; the HF mirror
        # already provides plain JSON.
        download_file(url, dest, expected_sha256=sha, desc=f"PubLayNet {split}")

    logger.info("PubLayNet download complete → %s", PUBLAYNET_DIR)


# ── DocLayNet download ──────────────────────────────────────────────────────


def download_doclaynet() -> None:
    """Download DocLayNet core dataset ZIP and extract COCO annotation JSONs.

    The full ZIP is ~1.8 GB and includes PNG images.  We download the whole
    thing (needed for any future image work) but the loader only reads the
    COCO JSON files.
    """
    DOCLAYNET_DIR.mkdir(parents=True, exist_ok=True)
    zip_dest = DOCLAYNET_DIR / "DocLayNet_core.zip"

    # Check if annotations are already extracted.
    coco_dir = DOCLAYNET_DIR / "COCO"
    if coco_dir.exists() and any(coco_dir.glob("*.json")):
        logger.info(
            "DocLayNet COCO annotations already present at %s — skipping.",
            coco_dir,
        )
        return

    download_file(DOCLAYNET_CORE_URL, zip_dest, desc="DocLayNet core ZIP")

    # Extract — pull everything so images are available later.
    logger.info("Extracting %s …", zip_dest)
    with zipfile.ZipFile(zip_dest, "r") as zf:
        # The ZIP may have a top-level directory; we flatten into DOCLAYNET_DIR.
        members = zf.namelist()
        # Detect common prefix (e.g. "DocLayNet_core/")
        prefix = ""
        if members and "/" in members[0]:
            candidate = members[0].split("/")[0] + "/"
            if all(m.startswith(candidate) or m == candidate.rstrip("/") for m in members):
                prefix = candidate

        for member in tqdm(members, desc="Extracting DocLayNet", unit="file"):
            # Strip the common prefix for a cleaner layout.
            rel = member[len(prefix):] if prefix else member
            if not rel or rel.endswith("/"):
                (DOCLAYNET_DIR / rel).mkdir(parents=True, exist_ok=True)
                continue
            dest_path = DOCLAYNET_DIR / rel
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    logger.info("DocLayNet extraction complete → %s", DOCLAYNET_DIR)

    # Optionally remove the ZIP to save disk space (uncomment if desired).
    # zip_dest.unlink()


# ── CLI entry point ─────────────────────────────────────────────────────────


def main() -> None:
    """Download datasets from the command line."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Download PubLayNet and/or DocLayNet for Vision-SARVAM.",
    )
    parser.add_argument(
        "--dataset",
        choices=["publaynet", "doclaynet", "all"],
        default="all",
        help="Which dataset to download (default: all).",
    )
    args = parser.parse_args()

    if args.dataset in ("publaynet", "all"):
        logger.info("═══ Downloading PubLayNet ═══")
        download_publaynet()

    if args.dataset in ("doclaynet", "all"):
        logger.info("═══ Downloading DocLayNet ═══")
        download_doclaynet()

    logger.info("All downloads finished.  Data directory: %s", DATA_DIR)


if __name__ == "__main__":
    main()
