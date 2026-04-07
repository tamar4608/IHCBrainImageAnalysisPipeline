# nb03_atlas_registration.py
# ============================================================
# Notebook 3 — Atlas Registration
# Align each brain section to the Allen Mouse Brain Atlas.
# ============================================================
"""
Workflow (per slice):
  1. Auto-detect AP coordinate.
  2. Overlay section (green) on atlas plane (magenta) — grey = aligned.
  3. If wrong, manually scroll AP slider.
  4. Run fine registration (SimpleITK).
  5. Approve or Reject.

Region masks and transform files saved to data/processed/<slide>/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from utils import load_config, SessionStore

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import SimpleITK as sitk

# ── Config & session ──────────────────────────────────────────────────────────
cfg      = load_config()
session  = SessionStore(cfg["paths"]["session"])
PROC_DIR = Path(cfg["paths"]["processed"])
AP_RANGE = cfg["atlas"]["ap_range"]           # e.g. [400, 700]
ATLAS_NAME = cfg["atlas"]["name"]

# ── Load atlas (BrainGlobe) ───────────────────────────────────────────────────
try:
    from bg_atlasapi import BrainGlobeAtlas
    atlas = BrainGlobeAtlas(ATLAS_NAME)
    ATLAS_VOLUME    = atlas.reference          # (AP, DV, ML)
    ANNOTATION_VOL  = atlas.annotation
    RESOLUTION_UM   = atlas.resolution[0]     # µm per voxel
    print(f"Atlas loaded: {ATLAS_NAME}  resolution={RESOLUTION_UM} µm")
except Exception as e:
    print(f"⚠️  Could not load atlas: {e}")
    print("   Install BrainGlobe: pip install bg-atlasapi")
    print("   Then run: python -m bg_atlasapi.utils download_atlas allen_mouse_25um")
    ATLAS_VOLUME   = None
    ANNOTATION_VOL = None
    RESOLUTION_UM  = 25.0

# ── Included & preprocessed slides ───────────────────────────────────────────
selections = session.load("slice_selections", default={})
included   = [
    Path(v["slide"])
    for v in selections.values()
    if v["decision"] == "include"
]
proc_slides = [s for s in included if (PROC_DIR / s.stem / "DAPI.npy").exists()]

if not proc_slides:
    print("No preprocessed slides found. Run nb02 first.")
    sys.exit(0)

print(f"Slides to register: {len(proc_slides)}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_dapi(slide_path: Path) -> np.ndarray:
    """Load DAPI channel as float32 (H, W)."""
    return np.load(PROC_DIR / slide_path.stem / "DAPI.npy")


def auto_detect_ap(section: np.ndarray) -> int:
    """
    Very simple AP estimator: compare section projection to each atlas coronal plane.
    Falls back to midpoint of AP_RANGE if atlas is unavailable.
    """
    if ATLAS_VOLUME is None:
        return (AP_RANGE[0] + AP_RANGE[1]) // 2

    best_ap, best_cc = AP_RANGE[0], -1.0
    sec_flat = section.flatten()
    sec_norm = (sec_flat - sec_flat.mean()) / (sec_flat.std() + 1e-8)

    for ap in range(AP_RANGE[0], AP_RANGE[1], 10):
        plane = ATLAS_VOLUME[ap].astype(np.float32)
        plane_resized = _resize_to(plane, section.shape)
        pf = plane_resized.flatten()
        pf_norm = (pf - pf.mean()) / (pf.std() + 1e-8)
        cc = float(np.dot(sec_norm, pf_norm) / len(sec_norm))
        if cc > best_cc:
            best_cc, best_ap = cc, ap

    return best_ap


def _resize_to(img: np.ndarray, shape: tuple) -> np.ndarray:
    from PIL import Image as PILImage
    pil = PILImage.fromarray(img.astype(np.float32))
    pil = pil.resize((shape[1], shape[0]), PILImage.BILINEAR)
    return np.array(pil)


def make_overlay(section: np.ndarray, atlas_plane: np.ndarray) -> np.ndarray:
    """Magenta-green overlay. Grey pixels = good alignment."""
    s = np.clip(section, 0, 1)
    a = np.clip(_resize_to(atlas_plane.astype(np.float32) /
                            atlas_plane.max(), section.shape), 0, 1)
    r = a          # magenta = atlas
    g = s          # green   = section
    b = a
    return np.stack([r, g, b], axis=-1)


def run_registration(section: np.ndarray, atlas_plane: np.ndarray) -> sitk.Transform:
    """Run a 2-D affine registration (section → atlas) using SimpleITK."""
    fixed   = sitk.GetImageFromArray(atlas_plane.astype(np.float32))
    moving  = sitk.GetImageFromArray(
        _resize_to(section, atlas_plane.shape).astype(np.float32))

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMeanSquares()
    reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200,
                                       convergenceMinimumValue=1e-6,
                                       convergenceWindowSize=10)
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY))
    reg.SetInterpolator(sitk.sitkLinear)
    transform = reg.Execute(fixed, moving)
    return transform


# ── Main registration loop ────────────────────────────────────────────────────
reg_results: dict = session.load("registration_results", default={})

print("\n" + "=" * 55)
print("  Atlas Registration (close each window to continue)")
print("=" * 55)

for slide in proc_slides:
    key = slide.name
    if key in reg_results:
        prev = reg_results[key]
        print(f"\n[{slide.name}]  Previous: {prev['status']}  AP={prev.get('ap','?')}")
        redo = input("  Re-register? [y/N]: ").strip().lower()
        if redo != "y":
            continue

    section = _load_dapi(slide)
    print(f"\n[{slide.name}]  Auto-detecting AP …")
    ap = auto_detect_ap(section)
    print(f"  Suggested AP: {ap}")

    if ATLAS_VOLUME is not None:
        atlas_plane = ATLAS_VOLUME[ap].astype(np.float32)
        atlas_plane = atlas_plane / (atlas_plane.max() + 1e-8)

        # ── Interactive AP slider ────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(7, 7))
        plt.subplots_adjust(bottom=0.2)
        overlay_img = ax.imshow(make_overlay(section, atlas_plane))
        ax.set_title("Green=section  Magenta=atlas  Grey=aligned\n"
                     "Adjust AP slider, then close window")
        ax.axis("off")

        ax_ap  = plt.axes([0.15, 0.08, 0.65, 0.04])
        s_ap   = Slider(ax_ap, "AP", AP_RANGE[0], AP_RANGE[1],
                        valinit=ap, valstep=5)

        def _update_ap(val):
            nonlocal ap, atlas_plane
            ap = int(s_ap.val)
            atlas_plane = ATLAS_VOLUME[ap].astype(np.float32)
            atlas_plane /= (atlas_plane.max() + 1e-8)
            overlay_img.set_data(make_overlay(section, atlas_plane))
            fig.canvas.draw_idle()

        s_ap.on_changed(_update_ap)
        plt.show()
        plt.close("all")

        # ── Run fine registration ────────────────────────────────────────────
        print(f"  Running fine registration at AP={ap} …")
        try:
            transform = run_registration(section, atlas_plane)

            # Save transform
            tf_path = PROC_DIR / slide.stem / "transform.tfm"
            sitk.WriteTransform(transform, str(tf_path))

            # Save AP annotation slice
            if ANNOTATION_VOL is not None:
                ann_slice = ANNOTATION_VOL[ap]
                np.save(PROC_DIR / slide.stem / "annotation.npy", ann_slice)

            # Display result
            moved = sitk.GetArrayFromImage(
                sitk.Resample(sitk.GetImageFromArray(
                    _resize_to(section, atlas_plane.shape).astype(np.float32)),
                    sitk.GetImageFromArray(atlas_plane), transform,
                    sitk.sitkLinear, 0.0))

            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
            axes2[0].imshow(make_overlay(section, atlas_plane))
            axes2[0].set_title("Before registration")
            axes2[1].imshow(make_overlay(moved, atlas_plane))
            axes2[1].set_title("After registration")
            for a in axes2:
                a.axis("off")
            fig2.suptitle(slide.name)
            plt.tight_layout()
            plt.show(block=False)
            plt.pause(0.5)

            decision = input("  Approve or Reject? [A/R]: ").strip().upper()
            status = "approved" if decision == "A" else "rejected"
            plt.close("all")

        except Exception as e:
            print(f"  ⚠️  Registration failed: {e}")
            status = "failed"

    else:
        # Atlas unavailable — store AP guess only
        status = "no_atlas"
        print("  Atlas unavailable — storing AP estimate only.")

    reg_results[key] = {"ap": ap, "status": status}
    session.save("registration_results", reg_results)
    icon = {"approved": "✅", "rejected": "❌", "failed": "⚠️", "no_atlas": "—"}.get(status, "?")
    print(f"  {icon}  {status.upper()}")

approved = sum(1 for v in reg_results.values() if v["status"] == "approved")
print(f"\n  Approved: {approved} / {len(reg_results)}")
print("  → Proceed to nb04_cell_detection.py")
