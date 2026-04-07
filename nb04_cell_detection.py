# nb04_cell_detection.py
# ============================================================
# Notebook 4 — Cell Detection
# Use Cellpose to detect & count cells per fluorescence channel.
# ============================================================
"""
Workflow (per channel: DAPI, GFP/Orexin, Cy5/cFOS):
  1. Pick a test slice, run preview with default params.
  2. Inspect yellow outlines — tune diameter, threshold, min/max size.
  3. Apply to all approved slices.
  4. Approve or flag each result.

Cell masks saved to data/processed/<slide>/<channel>_masks.npy
Centroid CSVs saved to data/processed/<slide>/<channel>_centroids.csv
"""

import sys
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import (
    load_config, SessionStore,
    normalize_channel, centroids_from_masks,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# ── Cellpose import (soft) ─────────────────────────────────────────────────────
try:
    from cellpose import models as cp_models
    CELLPOSE_OK = True
except ImportError:
    print("⚠️  Cellpose not installed.  pip install cellpose")
    CELLPOSE_OK = False

# ── Config & session ──────────────────────────────────────────────────────────
cfg      = load_config()
session  = SessionStore(cfg["paths"]["session"])
PROC_DIR = Path(cfg["paths"]["processed"])
CP_CFG   = cfg["cellpose"]         # channel-level Cellpose settings

# ── Approved slides ───────────────────────────────────────────────────────────
reg_results = session.load("registration_results", default={})
selections  = session.load("slice_selections",    default={})

approved_slides = [
    Path(v["slide"])
    for k, v in selections.items()
    if v["decision"] == "include"
    and reg_results.get(k, {}).get("status") == "approved"
]

# Fallback: use all preprocessed slides if none are registered
if not approved_slides:
    approved_slides = [
        Path(v["slide"])
        for v in selections.values()
        if v["decision"] == "include"
        and (PROC_DIR / Path(v["slide"]).stem / "DAPI.npy").exists()
    ]
    print(f"(No registered slides — using all {len(approved_slides)} preprocessed slides)")

if not approved_slides:
    print("No slides ready for cell detection. Complete nb02/nb03 first.")
    sys.exit(0)

print(f"Slides to process: {len(approved_slides)}")
CHANNELS = list(cfg["channels"].keys())   # ["DAPI", "GFP", "Cy5"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_channel(slide_path: Path, ch: str) -> np.ndarray:
    p = PROC_DIR / slide_path.stem / f"{ch}.npy"
    if not p.exists():
        raise FileNotFoundError(f"Preprocessed channel not found: {p}")
    return np.load(p)


def run_cellpose(img: np.ndarray, ch_name: str, params: dict) -> np.ndarray:
    """Run Cellpose and return integer label masks (H, W)."""
    if not CELLPOSE_OK:
        raise RuntimeError("Cellpose not installed.")

    model_name = params["model"]
    model = cp_models.Cellpose(model_type=model_name, gpu=False)

    # Cellpose expects uint8 or float [0,1]
    img_in = (img * 255).clip(0, 255).astype(np.uint8)

    # Apply intensity threshold — zero out below-threshold pixels
    thr = params.get("intensity_threshold", 0.0)
    img_in[img < thr] = 0

    masks, _, _, _ = model.eval(
        img_in,
        diameter=params["diameter"],
        channels=[0, 0],               # grayscale
        flow_threshold=params["flow_threshold"],
        cellprob_threshold=params["cellprob_threshold"],
        min_size=params["min_size"],
    )

    # Apply max size filter
    max_sz = params.get("max_size", np.inf)
    for label_id in np.unique(masks):
        if label_id == 0:
            continue
        cell_px = np.sum(masks == label_id)
        if cell_px > max_sz:
            masks[masks == label_id] = 0

    return masks.astype(np.int32)


def save_masks_and_centroids(slide_path: Path, ch: str, masks: np.ndarray) -> int:
    """Save masks (.npy) and centroids (.csv). Return cell count."""
    out_dir = PROC_DIR / slide_path.stem
    np.save(out_dir / f"{ch}_masks.npy", masks)

    centroids = centroids_from_masks(masks)
    csv_path  = out_dir / f"{ch}_centroids.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["row_px", "col_px"])
        writer.writerows(centroids.tolist())

    return len(centroids)


def overlay_masks(img: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Return an RGB image with yellow outlines around detected cells."""
    from scipy.ndimage import binary_dilation
    edges = np.zeros_like(masks, dtype=bool)
    for lab in np.unique(masks):
        if lab == 0:
            continue
        cell = masks == lab
        dilated = binary_dilation(cell, iterations=1)
        edges |= (dilated & ~cell)

    rgb = np.stack([img, img, img], axis=-1)
    rgb[edges] = [1.0, 1.0, 0.0]   # yellow outlines
    return np.clip(rgb, 0, 1)


# ── Parameter tuning (interactive) ────────────────────────────────────────────

def tune_channel(slide_path: Path, ch: str) -> dict:
    """Interactive slider tuning for one channel. Returns final params dict."""
    default = dict(CP_CFG[ch])
    params  = dict(default)

    img = _load_channel(slide_path, ch)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    plt.subplots_adjust(bottom=0.35)
    axes[0].set_title(f"{ch} — raw")
    axes[1].set_title(f"{ch} — detections (yellow outlines)")
    axes[0].imshow(img, cmap="gray")
    axes[1].imshow(img, cmap="gray")
    for ax in axes:
        ax.axis("off")

    ax_diam = plt.axes([0.15, 0.25, 0.65, 0.03])
    ax_thr  = plt.axes([0.15, 0.19, 0.65, 0.03])
    ax_min  = plt.axes([0.15, 0.13, 0.65, 0.03])
    ax_max  = plt.axes([0.15, 0.07, 0.65, 0.03])

    s_diam = Slider(ax_diam, "Diameter",  5,  80,  valinit=params["diameter"],           valstep=1)
    s_thr  = Slider(ax_thr,  "Int. thr.", 0,  0.5, valinit=params["intensity_threshold"], valstep=0.01)
    s_min  = Slider(ax_min,  "Min size",  0,  500, valinit=params["min_size"],            valstep=10)
    s_max  = Slider(ax_max,  "Max size", 100, 10000, valinit=params["max_size"],          valstep=100)

    ax_run = plt.axes([0.82, 0.19, 0.12, 0.06])
    btn    = Button(ax_run, "Run preview")

    preview_result = [None]

    def _run(_):
        params.update({
            "diameter":            int(s_diam.val),
            "intensity_threshold": float(s_thr.val),
            "min_size":            int(s_min.val),
            "max_size":            int(s_max.val),
        })
        if CELLPOSE_OK:
            masks = run_cellpose(img, ch, params)
            preview_result[0] = masks
            ov = overlay_masks(img, masks)
            axes[1].imshow(ov)
            axes[1].set_title(f"{ch} — {int(masks.max())} cells detected")
        else:
            axes[1].set_title("Cellpose not installed")
        fig.canvas.draw_idle()

    btn.on_clicked(_run)
    fig.suptitle(f"Tuning {ch} — {slide_path.name}\nClose window when satisfied")
    plt.show()
    plt.close("all")
    return params


# ── Interactive selection ─────────────────────────────────────────────────────
print("\nAvailable slides:")
for i, p in enumerate(approved_slides):
    print(f"  [{i}] {p.name}")

idx = input(f"Select representative slide [0-{len(approved_slides)-1}]: ").strip()
try:
    rep = approved_slides[int(idx)]
except (ValueError, IndexError):
    rep = approved_slides[0]

final_params: dict[str, dict] = {}
for ch in CHANNELS:
    print(f"\nTuning parameters for channel: {ch}")
    final_params[ch] = tune_channel(rep, ch)

session.save("cellpose_params", final_params)
print("\nParameters saved.")


# ── Batch detection ───────────────────────────────────────────────────────────
detection_results: dict = session.load("detection_results", default={})

print("\n" + "=" * 55)
print("  Batch cell detection — all approved slides")
print("=" * 55)

for slide in approved_slides:
    key     = slide.name
    counts  = {}
    flagged = False

    for ch in CHANNELS:
        try:
            img   = _load_channel(slide, ch)
            masks = run_cellpose(img, ch, final_params[ch])
            n     = save_masks_and_centroids(slide, ch, masks)
            counts[ch] = n
            print(f"  {slide.name}  {ch}: {n} cells")

            # Quick visual check
            ov = overlay_masks(img, masks)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(ov)
            ax.set_title(f"{slide.name}  {ch}  ({n} cells)")
            ax.axis("off")
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.2)

        except Exception as e:
            print(f"  ⚠️  {slide.name} / {ch}: {e}")
            counts[ch] = -1
            flagged = True

    plt.close("all")
    flag = input(f"  Flag {slide.name} for PI review? [y/N]: ").strip().lower()
    detection_results[key] = {
        "counts": counts,
        "flag":   "flagged" if flag == "y" or flagged else "ok",
    }
    session.save("detection_results", detection_results)

print("\n  Detection complete.")
print("  → Proceed to nb05_coloc_report.py")
