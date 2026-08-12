# Instructions for Use: Mouse Heart Masson's Trichrome Fibrosis Pipeline v4

## 1. Purpose

This pipeline quantifies collagen/fibrosis in mouse heart Masson's Trichrome images using Python only. It supports total myocardial fibrosis and compartment-specific measurements:

- total collagen/fibrosis
- interstitial fibrosis
- perivascular collagen
- perivascular ring collagen

Version 4 adds two usability changes:

1. zoom and pan during manual mask creation
2. faster interactive threshold tuning using a downsampled preview

The final quantification still runs on the actual image files in `images/`.

## 2. Recommended project layout

```text
heart_masson_fibrosis_pipeline_v4/
├── slides/                         # optional: original .svs files
├── images/                         # converted TIFFs or ordinary image files
├── masks_manual/
│   ├── myocardium/                  # manually drawn myocardium masks
│   ├── vessel/                      # manually drawn vessel masks
│   └── exclude/                     # manually drawn exclusion masks
├── config/
│   └── default_config.json
├── metadata.csv
├── results/
└── scripts/
```

## 3. Install Python dependencies

Open a terminal in the pipeline folder and run:

```bash
python -m pip install -r requirements.txt
```

If `openslide-python` fails on Windows, keep the SVS conversion step separate by using QuPath or Bio-Formats to export TIFFs, then continue with the Python fibrosis pipeline.

## 4. Convert SVS slides to TIFF, if required

Place `.svs` files in:

```text
slides/
```

Run:

```bash
python scripts/00_batch_convert_svs_to_tiff.py --input-dir slides --output-dir images --level 1
```

Output files will be written to:

```text
images/
```

Use these rules:

```text
level 0 = full resolution, often very large and slow
level 1 = recommended starting point
level 2 = smaller and faster, useful if level 1 is too slow
```

For initial testing, run:

```bash
python scripts/00_batch_convert_svs_to_tiff.py --input-dir slides --output-dir images --level 2
```

For final analysis, use the lowest level that still clearly preserves collagen features relevant to your measurement.

## 5. Build metadata

After images exist in `images/`, run:

```bash
python scripts/01_build_metadata_from_images.py
```

This creates:

```text
metadata.csv
```

Edit `metadata.csv` in Excel or a text editor. Fill in at least:

```text
mouse_id
group
section
region
stain_batch
use_image
```

If you know the scale, fill in:

```text
pixel_size_um
```

This allows the perivascular buffer to be defined in micrometres rather than pixels.

## 6. Draw manual masks with zoom and pan

The pipeline uses three manual mask types.

### 6.1 Myocardium mask

This defines the analysed myocardium denominator.

```bash
python scripts/02_draw_roi_masks.py --mask-type myocardium
```

Draw around the myocardium you want to analyse. Exclude obvious chamber lumen, tissue outside the desired myocardial region, and large irrelevant structures.

### 6.2 Vessel mask

This defines the vessels used to construct the perivascular zones.

```bash
python scripts/02_draw_roi_masks.py --mask-type vessel
```

Manual vessel rule:

```text
Draw the vessel structure/lumen/wall, not the collagen ring itself.
```

Include valid intramyocardial vessels. Avoid ventricular lumen, tears, folds, cracks, valve tissue, and processing artefacts.

### 6.3 Exclusion mask

This removes tissue artefacts or regions that should not be analysed.

```bash
python scripts/02_draw_roi_masks.py --mask-type exclude
```

Use this for folds, tears, staining artefacts, glass/background accidentally included in the myocardium ROI, chamber lumen if included, valves if not part of the endpoint, and very large vessels if excluded by your protocol.

## 7. Mask drawing controls

### Polygon controls

```text
Left click             add polygon point
c                      close/fill current polygon into the mask
u                      undo last point
r                      reset current mask for this image
e                      toggle erase mode
s                      save current mask
n                      save and move to next image
p                      save and move to previous image
h                      print help text
q / Esc                quit
```

### Zoom and pan controls

```text
Mouse wheel            zoom in/out around the cursor
+ / =                  zoom in
- / _                  zoom out
0                      reset zoom to fit image
Right mouse drag       pan
Middle mouse drag      pan
Arrow keys / A/D/W/X   pan
```

The mask tool now renders only the visible crop while zoomed in. This is faster than resizing the full slide-sized TIFF every time the window refreshes.

## 8. Tune the collagen threshold

Run the interactive HSV threshold tool:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 1400
```

The preview window shows:

```text
original | collagen overlay | collagen mask
```

Yellow pixels in the overlay are pixels currently classified as collagen.

Press:

```text
w      write the current settings to config/default_config.json
q/Esc  quit
```

### 8.1 Threshold zoom controls

```text
Mouse wheel            zoom in/out around cursor
+ / =                  zoom in
- / _                  zoom out
0                      reset zoom
Right mouse drag       pan
Middle mouse drag      pan
Arrow keys / A/D/S/X   pan
```

In the threshold window, `w` is reserved for saving the HSV settings to the config file.

## 9. What to do if threshold tuning is slow or unresponsive

The threshold tool recalculates the collagen mask when slider values change. Large TIFFs can still be slow. Use these fixes in order.

### 9.1 Use a smaller preview

Try:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 900
```

or even:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 700
```

The final saved thresholds are still applied to the full image during quantification.

### 9.2 Convert SVS to a lower-resolution TIFF level

If level 1 TIFFs are too large, convert at level 2:

```bash
python scripts/00_batch_convert_svs_to_tiff.py --input-dir slides --output-dir images --level 2
```

Use this for testing. For final analysis, confirm that the chosen level still preserves collagen features adequately.

### 9.3 Use preview images for tuning only

You can tune on a representative smaller image or crop, save the HSV settings, then run the final quantification on all images.

Example:

```bash
python scripts/03_tune_masson_thresholds.py --interactive --image mouse01_level1.tiff --preview-max-dim 1000
```

### 9.4 Generate static threshold previews instead of interactive tuning

If the interactive GUI is unstable on your computer, generate static previews:

```bash
python scripts/03_tune_masson_thresholds.py --max-images 12 --preview-max-dim 1000
```

Then inspect:

```text
results/qc/threshold_previews/
```

Manually edit the HSV values in:

```text
config/default_config.json
```

and rerun the preview command until the threshold looks correct.

## 10. Run fibrosis quantification

After masks and thresholds are ready, run:

```bash
python scripts/04_quantify_masson_fibrosis.py
```

Main image-level output:

```text
results/image_level_compartment_fibrosis_results.csv
```

QC overlays:

```text
results/overlays/
```

Inspect the overlays before trusting the numbers.

## 11. Summarise by mouse and group

Run:

```bash
python scripts/05_summarise_by_mouse.py
```

Outputs:

```text
results/section_level_fibrosis_summary.csv
results/mouse_level_compartment_fibrosis_summary.csv
results/group_level_compartment_fibrosis_summary.csv
```

For statistical analysis, use the mouse-level table unless your experimental design says otherwise.

## 12. Plot and create QC contact sheet

```bash
python scripts/06_plot_fibrosis_results.py
python scripts/07_make_qc_contact_sheet.py
```

Outputs:

```text
results/plots/
results/qc/overlay_contact_sheet.png
```

## 13. Key measurements

The pipeline calculates:

```text
total_fibrosis_percent = total collagen area / analysis area × 100
perivascular_collagen_percent = collagen inside perivascular zone / perivascular zone area × 100
perivascular_ring_collagen_percent = collagen in perivascular ring / perivascular ring area × 100
interstitial_fibrosis_percent = collagen outside perivascular zone / interstitial area × 100
```

The perivascular zone is constructed from the manual vessel mask expanded by:

```text
perivascular_buffer_um
```

if `pixel_size_um` is known, otherwise by:

```text
perivascular_buffer_px_if_no_scale
```

These values are in:

```text
config/default_config.json
```

## 14. Recommended QC checklist

Before using results:

```text
1. Confirm myocardium masks include only the intended myocardium.
2. Confirm vessel masks mark vessels, not collagen rings.
3. Confirm exclusion masks remove folds, tears, artefacts, and irrelevant structures.
4. Confirm the collagen overlay detects blue/green collagen but not nuclei, shadows, or background.
5. Confirm perivascular collagen is classified near vessels and interstitial fibrosis is outside those zones.
6. Confirm each mouse is treated as the biological replicate.
```

## 15. Practical recommendation

For most datasets, start with:

```bash
python scripts/00_batch_convert_svs_to_tiff.py --input-dir slides --output-dir images --level 1
python scripts/01_build_metadata_from_images.py
python scripts/02_draw_roi_masks.py --mask-type myocardium --max-display 1400
python scripts/02_draw_roi_masks.py --mask-type vessel --max-display 1400
python scripts/02_draw_roi_masks.py --mask-type exclude --max-display 1400
python scripts/03_tune_masson_thresholds.py --interactive --preview-max-dim 1000
python scripts/04_quantify_masson_fibrosis.py
python scripts/05_summarise_by_mouse.py
python scripts/06_plot_fibrosis_results.py
python scripts/07_make_qc_contact_sheet.py
```

If anything becomes slow, lower the SVS conversion level or lower `--preview-max-dim` for threshold tuning.
