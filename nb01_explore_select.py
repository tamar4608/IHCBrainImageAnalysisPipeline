# nb01_explore_select.py
# ============================================================
# Notebook 1 — Explore & Select Slices
# Run this script interactively or convert to Jupyter notebook.
# ============================================================
"""
For each slide, display a thumbnail and prompt the user to:
  - INCLUDE or EXCLUDE the slice
  - Add an optional note (tissue quality, GFP signal, etc.)

Progress is saved automatically to session/slice_selections.json.
"""

import sys
from pathlib import Path

# ── Add scripts/ to path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import load_config, SessionStore, discover_slides, load_tif, make_rgb_thumbnail

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Load config & session ─────────────────────────────────────────────────────
cfg     = load_config()
session = SessionStore(cfg["paths"]["session"])

ch_map = {k: v for k, v in cfg["channels"].items()}   # {"DAPI":0,"GFP":1,"Cy5":2}
RAW_DIR = Path(cfg["paths"]["raw_images"])

# ── Discover slides ───────────────────────────────────────────────────────────
slides = discover_slides(RAW_DIR)
if not slides:
    print(f"⚠️  No slides found under:\n   {RAW_DIR}")
    print("   Edit config/pipeline_config.yaml → paths.raw_images")
    sys.exit(0)

print(f"Found {len(slides)} slide(s).\n")

# ── Load existing selections (resume support) ─────────────────────────────────
selections: dict = session.load("slice_selections", default={})


def _display_thumbnail(slide_path: Path, idx: int, total: int) -> None:
    """Show an RGB thumbnail of the slide in a matplotlib window."""
    img = load_tif(slide_path)
    thumb = make_rgb_thumbnail(img, ch_map)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(thumb)
    ax.set_title(f"[{idx+1}/{total}]  {slide_path.name}", fontsize=9)
    ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color="blue",  label="DAPI (all nuclei)"),
        mpatches.Patch(color="green", label="GFP / Orexin"),
        mpatches.Patch(color="red",   label="Cy5 / cFOS"),
    ]
    ax.legend(handles=patches, loc="lower left", fontsize=7, framealpha=0.7)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.5)


def _prompt_decision(slide_path: Path, prev: dict | None) -> dict:
    """Prompt the user for include/exclude + optional note. Returns a decision dict."""
    if prev:
        print(f"   Previous decision: {prev['decision'].upper()}  |  note: {prev.get('note','—')}")
        redo = input("   Keep this decision? [Y/n]: ").strip().lower()
        if redo != "n":
            return prev

    while True:
        choice = input("   Include or Exclude this slice? [I/E]: ").strip().upper()
        if choice in ("I", "E"):
            break
        print("   Please enter I or E.")

    note = input("   Optional note (press Enter to skip): ").strip()
    return {
        "slide":    str(slide_path),
        "decision": "include" if choice == "I" else "exclude",
        "note":     note,
    }


# ── Main loop ─────────────────────────────────────────────────────────────────
print("=" * 60)
print("  Slice selection — DAPI=blue  GFP=green  Cy5=red")
print("  Your choices are saved after each slide.")
print("=" * 60)

for i, slide in enumerate(slides):
    key  = slide.name
    prev = selections.get(key)

    print(f"\n[{i+1}/{len(slides)}]  {slide.name}")

    try:
        _display_thumbnail(slide, i, len(slides))
    except Exception as e:
        print(f"   ⚠️  Could not display thumbnail: {e}")

    decision = _prompt_decision(slide, prev)
    selections[key] = decision
    session.save("slice_selections", selections)

    plt.close("all")
    emoji = "✅" if decision["decision"] == "include" else "❌"
    print(f"   {emoji}  Saved: {decision['decision'].upper()}")

# ── Summary ───────────────────────────────────────────────────────────────────
included = [k for k, v in selections.items() if v["decision"] == "include"]
excluded = [k for k, v in selections.items() if v["decision"] == "exclude"]

print("\n" + "=" * 60)
print(f"  ✅  Included: {len(included)}  |  ❌  Excluded: {len(excluded)}")
print("  Selections saved to session/slice_selections.json")
print("  → Proceed to nb02_preprocess.py")
