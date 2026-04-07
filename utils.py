"""
utils.py — shared helper functions for the IHC pipeline.
Do not edit without PI approval.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tifffile
import yaml


# ── Config ─────────────────────────────────────────────────────────────────────

def load_config(config_path: str | Path = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load and return the YAML pipeline configuration."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        return yaml.safe_load(f)


# ── Session / progress persistence ─────────────────────────────────────────────

class SessionStore:
    """Simple JSON-backed key-value store so progress survives across Jupyter sessions."""

    def __init__(self, session_dir: str | Path = "session"):
        self._dir = Path(session_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_\-]", "_", key)
        return self._dir / f"{safe}.json"

    def save(self, key: str, value: Any) -> None:
        with open(self._path(key), "w") as f:
            json.dump(value, f, indent=2)

    def load(self, key: str, default: Any = None) -> Any:
        p = self._path(key)
        if not p.exists():
            return default
        with open(p) as f:
            return json.load(f)

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def delete(self, key: str) -> None:
        p = self._path(key)
        if p.exists():
            p.unlink()


# ── Image I/O ──────────────────────────────────────────────────────────────────

def load_tif(path: str | Path) -> np.ndarray:
    """Load a multi-channel TIF → (C, H, W) uint16 array."""
    img = tifffile.imread(str(path))
    if img.ndim == 2:
        img = img[np.newaxis]          # single-channel
    elif img.ndim == 3 and img.shape[-1] in (3, 4):
        img = np.moveaxis(img, -1, 0)  # (H, W, C) → (C, H, W)
    return img.astype(np.uint16)


def extract_channel(img: np.ndarray, ch: int) -> np.ndarray:
    """Return a single channel (H, W) from a (C, H, W) array."""
    return img[ch]


def normalize_channel(
    img: np.ndarray,
    p_low: float = 1.0,
    p_high: float = 99.9,
) -> np.ndarray:
    """Percentile-clip and scale a 2-D image to [0, 1] float32."""
    lo, hi = np.percentile(img, [p_low, p_high])
    img_f = img.astype(np.float32)
    if hi > lo:
        img_f = (img_f - lo) / (hi - lo)
    img_f = np.clip(img_f, 0, 1)
    return img_f


# ── Background subtraction ─────────────────────────────────────────────────────

def subtract_background(img: np.ndarray, radius: int = 50) -> np.ndarray:
    """
    Rolling-ball background subtraction using a morphological opening.
    Works on float32 images in [0, 1].
    """
    from scipy.ndimage import uniform_filter

    bg = uniform_filter(img, size=radius * 2 + 1)
    result = np.clip(img.astype(np.float32) - bg.astype(np.float32), 0, None)
    return result


def gaussian_smooth(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img.astype(np.float32), sigma=sigma)


# ── Slide discovery ────────────────────────────────────────────────────────────

def discover_slides(raw_dir: str | Path) -> List[Path]:
    """
    Return sorted list of multi-channel TIF files under raw_dir,
    matching the naming convention *CH.tif.
    """
    raw_dir = Path(raw_dir)
    tifs = sorted(raw_dir.rglob("*_CH.tif"))
    return tifs


# ── Thumbnail helper ───────────────────────────────────────────────────────────

def make_rgb_thumbnail(
    img: np.ndarray,
    ch_map: Dict[str, int],
    size: Tuple[int, int] = (256, 256),
) -> np.ndarray:
    """
    Create an 8-bit RGB thumbnail from a multi-channel image.
    DAPI → blue, GFP → green, Cy5 → red.
    img: (C, H, W)
    """
    from PIL import Image as PILImage
    h, w = img.shape[1], img.shape[2]

    def _ch(name: str) -> np.ndarray:
        c = ch_map.get(name, -1)
        if c < 0 or c >= img.shape[0]:
            return np.zeros((h, w), np.float32)
        return normalize_channel(img[c])

    r = (_ch("Cy5") * 255).astype(np.uint8)
    g = (_ch("GFP")  * 255).astype(np.uint8)
    b = (_ch("DAPI") * 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    pil = PILImage.fromarray(rgb).resize(size, PILImage.LANCZOS)
    return np.array(pil)


# ── Cell centroid utilities ────────────────────────────────────────────────────

def centroids_from_masks(masks: np.ndarray) -> np.ndarray:
    """
    Given a label mask (H, W) where each unique non-zero integer is one cell,
    return an (N, 2) array of [row, col] centroids.
    """
    from scipy.ndimage import center_of_mass

    labels = np.unique(masks)
    labels = labels[labels > 0]
    if len(labels) == 0:
        return np.empty((0, 2), dtype=float)
    centres = np.array(center_of_mass(masks > 0, masks, labels))
    return centres   # (N, 2) — row, col


def match_cells(
    centroids_a: np.ndarray,
    centroids_b: np.ndarray,
    max_dist_px: float,
) -> np.ndarray:
    """
    Find pairs between two centroid arrays within max_dist_px.
    Returns (M, 2) index pairs [idx_a, idx_b].
    Uses a simple nearest-neighbour approach (sufficient for sparse IHC data).
    """
    from scipy.spatial import cKDTree

    if len(centroids_a) == 0 or len(centroids_b) == 0:
        return np.empty((0, 2), dtype=int)

    tree = cKDTree(centroids_b)
    dists, idxs = tree.query(centroids_a, distance_upper_bound=max_dist_px)
    valid = dists < max_dist_px
    a_idxs = np.where(valid)[0]
    b_idxs = idxs[valid]
    return np.column_stack([a_idxs, b_idxs])


# ── Quality flags ──────────────────────────────────────────────────────────────

QUALITY_FLAGS = ("Good", "Acceptable", "Flagged for PI review")


def validate_quality_flag(flag: str) -> str:
    if flag not in QUALITY_FLAGS:
        raise ValueError(f"Quality flag must be one of {QUALITY_FLAGS}")
    return flag


# ── Pixel → µm conversion ──────────────────────────────────────────────────────

def px_to_um(pixels: float, resolution_um_per_px: float) -> float:
    return pixels * resolution_um_per_px


def area_px_to_mm2(area_px: int, resolution_um_per_px: float) -> float:
    area_um2 = area_px * (resolution_um_per_px ** 2)
    return area_um2 / 1e6
