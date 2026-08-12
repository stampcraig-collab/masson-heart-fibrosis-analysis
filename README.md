# Mouse Heart Masson Trichrome Fibrosis Pipeline v4

Python-only desktop pipeline for quantifying Masson's Trichrome collagen/fibrosis in mouse heart tissue.

This version includes:

- `.svs` to `.tiff` batch conversion
- manual myocardium, vessel, and exclusion mask drawing
- zoom/pan while drawing masks
- faster interactive threshold tuning using a downsampled preview
- collagen/fibrosis quantification split into total, perivascular, perivascular ring, and interstitial compartments
- image-level, section-level, mouse-level, and group-level CSV outputs
- QC overlays and contact sheets

## Install

```bash
python -m pip install -r requirements.txt
```

## Full workflow

If starting with `.svs` files:

```bash
python scripts/00_batch_convert_svs_to_tiff.py --input-dir slides --output-dir images --level 1
```

Build metadata:

```bash
python scripts/01_build_metadata_from_images.py
```

Draw masks:

```bash
python scripts/02_draw_roi_masks.py --mask-type myocardium
python scripts/02_draw_roi_masks.py --mask-type vessel
python scripts/02_draw_roi_masks.py --mask-type exclude
```

Tune collagen threshold:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 1400
```

If slow, use:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 900
```

Run quantification and summaries:

```bash
python scripts/04_quantify_masson_fibrosis.py
python scripts/05_summarise_by_mouse.py
python scripts/06_plot_fibrosis_results.py
python scripts/07_make_qc_contact_sheet.py
```

## Mask drawing zoom controls

- Mouse wheel: zoom in/out
- `+` / `-`: zoom in/out
- `0`: reset zoom
- Right mouse drag or middle mouse drag: pan
- Arrow keys or `A/D/W/X`: pan

## Threshold tuning speed controls

The threshold tool now uses a resized preview. The final quantification still applies the saved HSV settings to the full-resolution images.

Good starting values:

- `--preview-max-dim 1400` for normal use
- `--preview-max-dim 900` if the GUI becomes slow
- `--preview-max-dim 1800` only if you need more visual detail

