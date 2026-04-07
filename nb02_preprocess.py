# nb02_preprocess.py
# ============================================================
# Notebook 2 — Preprocess
# Split channels, subtract background, normalise brightness.
# ============================================================
"""
Workflow:
  1. Pick one representative INCLUDED slice for parameter tuning.
  2. Adjust background_radius and gaussian_sigma interactively.
  3. Preview before/after for each channel.
  4. Apply approved parameters to ALL included slices.
  5. Flag any batch results that look wrong.

Outputs saved to data/processed/<slide_name>/<channel>.npy
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import (
    load_config, SessionStore,
    load_tif, extract_channel, normalize_channel,
    subtract_background, gaussian_smooth,
)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ── Config & session ──────────────────────────────────────────────────────────
cfg     = load_config()
session = SessionStore(cfg["paths"]["session"])
ch_map  = cfg["channels"]              # {"DAPI":0,"GFP":1,"Cy5":2}
PP_CFG  = cfg["preprocessing"]
OUT_DIR = Path(cfg["paths"]["processed"])
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load included slides ───────────────────────────────────────────────────────
selections: dict = session.load("slice_selections", default={})
included_paths = [
    Path(v["slide"])
    for v in selections.values()
    if v["decision"] == "include"
]

if not included_paths:
    print("No included slices found. Run nb01 first.")
    sys.exit(0)

print(f"Included slices: {len(included_paths)}")


# ── Interactive parameter tuning ───────────────────────────────────────────────

def tune_parameters(slide_path: Path) -> dict:
    """
    Show an interactive slider window for one slide.
    Returns the chosen parameters.
    """
    print("loading tif...")
    img = load_tif(slide_path)
    print("tif loaded!")
    params = {
        "bg_radius": PP_CFG["background_subtraction_radius"],
        "sigma":     PP_CFG["gaussian_sigma"],
    }

    # Use DAPI for preview (usually clearest signal)
    print("normalizing channels...")
    raw = normalize_channel(
        extract_channel(img, ch_map["DAPI"]),
        PP_CFG["normalize_percentile_low"],
        PP_CFG["normalize_percentile_high"],
    )
    print("normalized channels!")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)
    axes[0].set_title("Before")
    axes[1].set_title("After")
    im0 = axes[0].imshow(raw, cmap="gray", vmin=0, vmax=1)
    im1 = axes[1].imshow(raw, cmap="gray", vmin=0, vmax=1)
    for ax in axes:
        ax.axis("off")

    print("Created plot!")

    ax_rad = plt.axes([0.15, 0.12, 0.65, 0.03])
    ax_sig = plt.axes([0.15, 0.06, 0.65, 0.03])
    s_rad  = Slider(ax_rad, "BG radius", 10, 200, valinit=params["bg_radius"], valstep=5)
    s_sig  = Slider(ax_sig, "Gaussian σ",  0, 5.0, valinit=params["sigma"], valstep=0.5)

    def _update(_):
        processed = subtract_background(raw, int(s_rad.val))
        processed = gaussian_smooth(processed, s_sig.val)
        processed = np.clip(processed, 0, 1)
        im1.set_data(processed)
        fig.canvas.draw_idle()
        params["bg_radius"] = int(s_rad.val)
        params["sigma"]     = float(s_sig.val)

    s_rad.on_changed(_update)
    s_sig.on_changed(_update)

    fig.suptitle(f"Tuning: {slide_path.name}  (DAPI channel)\nClose window when satisfied")
    plt.show()
    return params


# ── Pick representative slice ─────────────────────────────────────────────────
print("\nAvailable included slices:")
for i, p in enumerate(included_paths):
    print(f"  [{i}] {p.name}")

idx = input(f"\nSelect representative slice index [0-{len(included_paths)-1}]: ").strip()
try:
    rep_slide = included_paths[int(idx)]
except (ValueError, IndexError):
    rep_slide = included_paths[0]
    print(f"Invalid — defaulting to [{0}] {rep_slide.name}")

print(f"\nOpening parameter tuner for: {rep_slide.name}")
tuned_params = tune_parameters(rep_slide)
print(f"Chosen params: bg_radius={tuned_params['bg_radius']}, sigma={tuned_params['sigma']}")

session.save("preprocess_params", tuned_params)


# ── Batch preprocessing ────────────────────────────────────────────────────────

def preprocess_slide(slide_path: Path, params: dict, ch_cfg: dict, pp_cfg: dict) -> dict[str, np.ndarray]:
    """Return a dict of channel_name → processed (H, W) float32 array."""
    img    = load_tif(slide_path)
    result = {}
    for ch_name, ch_idx in ch_cfg.items():
        raw  = extract_channel(img, ch_idx)
        norm = normalize_channel(raw, pp_cfg["normalize_percentile_low"],
                                 pp_cfg["normalize_percentile_high"])
        bg   = subtract_background(norm, params["bg_radius"])
        smth = gaussian_smooth(bg, params["sigma"])
        result[ch_name] = np.clip(smth, 0, 1).astype(np.float32)
    return result


print("\n" + "=" * 55)
print("  Batch preprocessing — all included slices")
print("=" * 55)

batch_flags: dict[str, str] = {}

for slide in included_paths:
    out_subdir = OUT_DIR / slide.stem
    out_subdir.mkdir(parents=True, exist_ok=True)

    try:
        channels = preprocess_slide(slide, tuned_params, ch_map, PP_CFG)

        # Save each channel
        for ch_name, arr in channels.items():
            np.save(out_subdir / f"{ch_name}.npy", arr)

        # Quick review plot
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (ch_name, arr) in zip(axes, channels.items()):
            ax.imshow(arr, cmap="gray")
            ax.set_title(ch_name)
            ax.axis("off")
        fig.suptitle(slide.name)
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)

        flag = input(f"  Flag this result? [ok / flag]: ").strip().lower()
        batch_flags[slide.name] = "flagged" if flag == "flag" else "ok"
        plt.close("all")

    except Exception as e:
        print(f"  ⚠️  Error processing {slide.name}: {e}")
        batch_flags[slide.name] = "error"

session.save("preprocess_flags", batch_flags)

ok_count  = sum(1 for v in batch_flags.values() if v == "ok")
bad_count = len(batch_flags) - ok_count
print(f"\n  Done — {ok_count} ok, {bad_count} flagged/error")
print("  → Proceed to nb03_atlas_registration.py")
