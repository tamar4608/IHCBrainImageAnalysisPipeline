# IHC Brain Image Analysis Pipeline
### Orexin / cFOS Co-localization in Mouse Brain Sections

---

## Quick Start

```
conda activate ihc_pipeline
cd C:\Users\itayz\IHC_Pipeline
jupyter lab          # then open notebooks/ in browser
```

Or run each notebook as a plain Python script in order:

```
python notebooks/nb01_explore_select.py
python notebooks/nb02_preprocess.py
python notebooks/nb03_atlas_registration.py
python notebooks/nb04_cell_detection.py
python notebooks/nb05_coloc_report.py
```

---

## Pipeline Overview

| # | Script | What it does | Your decisions |
|---|--------|--------------|----------------|
| 1 | `nb01_explore_select.py` | Browse thumbnails of all sections | Include / Exclude each slice |
| 2 | `nb02_preprocess.py` | Split channels, subtract background, normalise | Tune sliders, approve batch |
| 3 | `nb03_atlas_registration.py` | Align to Allen Brain Atlas | Approve / Reject per slice |
| 4 | `nb04_cell_detection.py` | Cellpose cell detection per channel | Tune diameter & thresholds |
| 5 | `nb05_coloc_report.py` | Co-localization, export Excel | Set distance threshold |

---

## Directory Layout

```
IHC_Pipeline/
├── environment.yml            ← conda environment definition
├── config/
│   └── pipeline_config.yaml   ← all settings (edit here)
├── scripts/
│   └── utils.py               ← shared helpers (do not edit)
├── notebooks/
│   ├── nb01_explore_select.py
│   ├── nb02_preprocess.py
│   ├── nb03_atlas_registration.py
│   ├── nb04_cell_detection.py
│   └── nb05_coloc_report.py
├── session/                   ← auto-saved progress (JSON)
├── data/
│   ├── processed/             ← per-slide channel arrays & masks
│   └── results/               ← coloc_results_<timestamp>.xlsx
```

---

## Channel Mapping

| Channel | Fluorophore | Marker | Detects |
|---------|-------------|--------|---------|
| 0 (DAPI) | DAPI | Nuclear stain | All cell nuclei |
| 1 (GFP) | FITC/GFP | Orexin | Orexin-expressing neurons |
| 2 (Cy5) | Cy5 | cFOS | Activated neurons |

---

## One-Time Setup

```bash
conda env create -f environment.yml
conda activate ihc_pipeline
python -m ipykernel install --user --name ihc_pipeline --display-name "Python 3 (ihc_pipeline)"
conda env config vars set BRAINGLOBE_HOME=G:\brainglobe_cache -n ihc_pipeline

# Verify
python -c "import tifffile, cellpose, SimpleITK; print('All good!')"
```

---

## Important Science Notes

- **Hemispheres**: Label as Hemisphere A and B only — do NOT assign L/R.
- **Orexin neurons**: Prefer slices with visible GFP+ cells in LHA/PeF.
- **Flagging**: When in doubt, flag for PI review — do not discard data unilaterally.
- **Keep a decision log** of any excluded slices or changed parameters.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Disk full (`errno 28`) | `conda clean --all -y`; move processed data to G: drive |
| Wrong Jupyter kernel | Select `ihc_pipeline` from top-right dropdown |
| Atlas registration looks wrong | Scroll AP slider manually; reject if no good match |
| Too many / too few cells | Adjust diameter slider; increase intensity threshold for false positives |
