#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from common import (
    load_config, read_image_rgb, write_rgb, collagen_mask_hsv, auto_tissue_mask,
    load_manual_mask, resolve_image_path, image_lookup, get_bool,
    perivascular_buffer_px, dilate_mask, percent, area_um2, bool_to_u8, outline
)




def split_csv_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_selection_file(path: str | None) -> list[str]:
    if path is None:
        return []
    selection_path = Path(path)
    if not selection_path.exists():
        raise SystemExit(f"Selection file not found: {selection_path}")
    values: list[str] = []
    for line in selection_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        values.append(line)
    return values


def normalise_selector(value: str) -> set[str]:
    p = Path(str(value))
    tokens = {str(value), str(p), p.name, p.stem}
    return {str(t).strip() for t in tokens if str(t).strip()}


def row_matches_selection(
    row: pd.Series,
    image_path: Path,
    selected_images: list[str],
    selected_sample_ids: list[str],
) -> bool:
    if not selected_images and not selected_sample_ids:
        return True

    image_tokens = {str(image_path), image_path.name, image_path.stem}
    selected_image_tokens: set[str] = set()
    for value in selected_images:
        selected_image_tokens.update(normalise_selector(value))

    sample_id = str(row.get("sample_id", "")).strip()
    selected_sample_tokens = {str(x).strip() for x in selected_sample_ids if str(x).strip()}

    image_match = bool(selected_image_tokens & image_tokens)
    sample_match = sample_id in selected_sample_tokens

    return image_match or sample_match

def build_compartment_masks(rgb: np.ndarray, image_stem: str, row: pd.Series, cfg: dict) -> dict[str, np.ndarray | None | int | float | str]:
    h, w = rgb.shape[:2]
    mask_root = Path(cfg.get("manual_mask_dir", "masks_manual"))

    myocardium = load_manual_mask(mask_root, "myocardium", image_stem, (h, w))
    exclude = load_manual_mask(mask_root, "exclude", image_stem, (h, w))
    vessel = load_manual_mask(mask_root, "vessel", image_stem, (h, w))

    if myocardium is not None:
        analysis_mask = myocardium.copy()
        denominator_source = "manual_myocardium_mask"
    else:
        analysis_mask = auto_tissue_mask(rgb, cfg)
        denominator_source = "automatic_tissue_mask"

    if exclude is not None:
        analysis_mask = analysis_mask & ~exclude

    collagen = collagen_mask_hsv(rgb, cfg) & analysis_mask

    if vessel is not None and vessel.any():
        vessel_core = vessel & analysis_mask
        buffer_px, pixel_size_um = perivascular_buffer_px(row, cfg)
        perivascular_zone = dilate_mask(vessel_core, buffer_px) & analysis_mask
        perivascular_ring = perivascular_zone & ~vessel_core
        interstitial_zone = analysis_mask & ~perivascular_zone
        vessel_status = "manual_vessel_mask_present"
    else:
        buffer_px, pixel_size_um = perivascular_buffer_px(row, cfg)
        vessel_core = np.zeros_like(analysis_mask, dtype=bool)
        perivascular_zone = np.zeros_like(analysis_mask, dtype=bool)
        perivascular_ring = np.zeros_like(analysis_mask, dtype=bool)
        interstitial_zone = analysis_mask.copy()
        vessel_status = "no_vessel_mask_found"

    return {
        "analysis_mask": analysis_mask,
        "collagen_mask": collagen,
        "exclude_mask": exclude,
        "vessel_core_mask": vessel_core,
        "perivascular_zone_mask": perivascular_zone,
        "perivascular_ring_mask": perivascular_ring,
        "interstitial_zone_mask": interstitial_zone,
        "buffer_px": buffer_px,
        "pixel_size_um": pixel_size_um,
        "denominator_source": denominator_source,
        "vessel_status": vessel_status,
    }


def make_overlay(rgb: np.ndarray, masks: dict, cfg: dict) -> np.ndarray:
    analysis = masks["analysis_mask"]
    collagen = masks["collagen_mask"]
    peri = masks["perivascular_zone_mask"]
    inter = masks["interstitial_zone_mask"]
    vessel = masks["vessel_core_mask"]
    exclude = masks.get("exclude_mask")

    out = rgb.copy()
    dim = float(cfg.get("overlay", {}).get("background_dim_factor", 0.35))
    alpha = float(cfg.get("overlay", {}).get("alpha", 0.45))

    out[~analysis] = (out[~analysis] * dim).astype(np.uint8)
    color_layer = out.copy()

    # RGB colors: green interstitial collagen, yellow perivascular collagen, cyan vessel core, red exclusions.
    inter_collagen = collagen & inter
    peri_collagen = collagen & peri
    color_layer[inter_collagen] = [0, 255, 0]
    color_layer[peri_collagen] = [255, 255, 0]
    color_layer[vessel] = [0, 255, 255]
    if exclude is not None:
        color_layer[exclude] = [255, 0, 0]

    blended = cv2.addWeighted(out, 1 - alpha, color_layer, alpha, 0)

    if cfg.get("overlay", {}).get("draw_contours", True):
        # Draw contour lines in BGR-compatible after conversion below; this image is RGB currently.
        # cv2 draws numeric tuples directly in channel order, so tuples below are RGB.
        for mask, color, thickness in [
            (analysis, (0, 200, 255), 2),
            (peri, (0, 0, 255), 2),
            (vessel, (0, 255, 255), 1),
        ]:
            contours, _ = cv2.findContours(bool_to_u8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, color, thickness)
        if exclude is not None:
            contours, _ = cv2.findContours(bool_to_u8(exclude), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(blended, contours, -1, (255, 0, 0), 2)

    return blended


def mask_area(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


def quantify_one(row: pd.Series, image_path: Path, cfg: dict, output_dirs: dict[str, Path]) -> dict:
    rgb = read_image_rgb(image_path)
    masks = build_compartment_masks(rgb, image_path.stem, row, cfg)

    analysis = masks["analysis_mask"]
    collagen = masks["collagen_mask"]
    peri = masks["perivascular_zone_mask"]
    peri_ring = masks["perivascular_ring_mask"]
    inter = masks["interstitial_zone_mask"]
    vessel = masks["vessel_core_mask"]
    px_um = masks["pixel_size_um"]

    total_collagen = collagen & analysis
    peri_collagen = collagen & peri
    peri_ring_collagen = collagen & peri_ring
    inter_collagen = collagen & inter
    vessel_core_collagen = collagen & vessel

    analysis_area = mask_area(analysis)
    total_collagen_area = mask_area(total_collagen)
    peri_area = mask_area(peri)
    peri_ring_area = mask_area(peri_ring)
    inter_area = mask_area(inter)
    vessel_area = mask_area(vessel)
    peri_collagen_area = mask_area(peri_collagen)
    peri_ring_collagen_area = mask_area(peri_ring_collagen)
    inter_collagen_area = mask_area(inter_collagen)
    vessel_core_collagen_area = mask_area(vessel_core_collagen)

    overlay = make_overlay(rgb, masks, cfg)
    write_rgb(output_dirs["overlays"] / f"{image_path.stem}_compartment_overlay.png", overlay)

    cv2.imwrite(str(output_dirs["masks"] / f"{image_path.stem}_analysis_mask.png"), bool_to_u8(analysis))
    cv2.imwrite(str(output_dirs["masks"] / f"{image_path.stem}_collagen_mask.png"), bool_to_u8(collagen))
    cv2.imwrite(str(output_dirs["masks"] / f"{image_path.stem}_perivascular_zone_mask.png"), bool_to_u8(peri))
    cv2.imwrite(str(output_dirs["masks"] / f"{image_path.stem}_interstitial_zone_mask.png"), bool_to_u8(inter))

    result = {
        "image": image_path.name,
        "sample_id": row.get("sample_id", image_path.stem),
        "mouse_id": row.get("mouse_id", ""),
        "group": row.get("group", ""),
        "section": row.get("section", ""),
        "region": row.get("region", ""),
        "stain_batch": row.get("stain_batch", ""),
        "use_image": row.get("use_image", "TRUE"),
        "notes": row.get("notes", ""),
        "denominator_source": masks["denominator_source"],
        "vessel_status": masks["vessel_status"],
        "perivascular_buffer_px": masks["buffer_px"],
        "pixel_size_um": px_um if px_um is not None else np.nan,
        "analysis_area_px": analysis_area,
        "total_collagen_area_px": total_collagen_area,
        "total_fibrosis_percent": percent(total_collagen_area, analysis_area),
        "vessel_core_area_px": vessel_area,
        "vessel_core_collagen_area_px": vessel_core_collagen_area,
        "perivascular_zone_area_px": peri_area,
        "perivascular_collagen_area_px": peri_collagen_area,
        "perivascular_collagen_percent": percent(peri_collagen_area, peri_area),
        "perivascular_ring_area_px": peri_ring_area,
        "perivascular_ring_collagen_area_px": peri_ring_collagen_area,
        "perivascular_ring_collagen_percent": percent(peri_ring_collagen_area, peri_ring_area),
        "interstitial_zone_area_px": inter_area,
        "interstitial_collagen_area_px": inter_collagen_area,
        "interstitial_fibrosis_percent": percent(inter_collagen_area, inter_area),
        "analysis_area_um2": area_um2(analysis_area, px_um),
        "total_collagen_area_um2": area_um2(total_collagen_area, px_um),
        "perivascular_zone_area_um2": area_um2(peri_area, px_um),
        "perivascular_collagen_area_um2": area_um2(peri_collagen_area, px_um),
        "interstitial_zone_area_um2": area_um2(inter_area, px_um),
        "interstitial_collagen_area_um2": area_um2(inter_collagen_area, px_um),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantify total, perivascular, and interstitial fibrosis in Masson's Trichrome images.")
    parser.add_argument("--config", default="config/default_config.json")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--image", default=None, help="Quantify one image only. Accepts filename, stem, or path.")
    parser.add_argument("--images", default=None, help="Comma-separated list of images to quantify. Accepts filenames, stems, or paths.")
    parser.add_argument("--image-list", default=None, help="Text file with one image filename/stem/path per line.")
    parser.add_argument("--sample-id", default=None, help="Quantify one sample_id from metadata.csv.")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated list of sample_id values from metadata.csv.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_path = Path(args.metadata or cfg.get("metadata_csv", "metadata.csv"))
    image_dir = Path(args.image_dir or cfg.get("image_dir", "images"))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))

    selected_images = []
    selected_images.extend(split_csv_arg(args.image))
    selected_images.extend(split_csv_arg(args.images))
    selected_images.extend(read_selection_file(args.image_list))

    selected_sample_ids = []
    selected_sample_ids.extend(split_csv_arg(args.sample_id))
    selected_sample_ids.extend(split_csv_arg(args.sample_ids))

    if not metadata_path.exists():
        raise SystemExit(f"Metadata not found: {metadata_path}. Run 01_build_metadata_from_images.py first.")

    output_dirs = {
        "results": results_dir,
        "overlays": results_dir / "overlays",
        "masks": results_dir / "masks",
        "qc": results_dir / "qc",
    }
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_path)
    lookup = image_lookup(image_dir, cfg, recursive=args.recursive)
    rows = []

    selected_names_for_report = []
    unmatched_requested_images = list(selected_images)
    unmatched_requested_sample_ids = list(selected_sample_ids)

    for _, row in metadata.iterrows():
        if not get_bool(row, "use_image", True):
            continue

        image_path = resolve_image_path(row, image_dir, cfg, lookup)

        if not row_matches_selection(row, image_path, selected_images, selected_sample_ids):
            continue

        selected_names_for_report.append(image_path.name)

        # Update unmatched trackers
        for req in list(unmatched_requested_images):
            if normalise_selector(req) & {str(image_path), image_path.name, image_path.stem}:
                unmatched_requested_images.remove(req)
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id in unmatched_requested_sample_ids:
            unmatched_requested_sample_ids.remove(sample_id)

        print(f"Processing {image_path.name}")
        rows.append(quantify_one(row, image_path, cfg, output_dirs))

    if selected_images or selected_sample_ids:
        print(f"Selected images for quantification: {len(selected_names_for_report)}")
        for i, name in enumerate(selected_names_for_report, start=1):
            print(f"  {i}. {name}")

        if unmatched_requested_images:
            print("Warning: these requested image selectors did not match any quantified image:")
            for x in unmatched_requested_images:
                print(f"  - {x}")

        if unmatched_requested_sample_ids:
            print("Warning: these requested sample_id values did not match any quantified image:")
            for x in unmatched_requested_sample_ids:
                print(f"  - {x}")

    if not rows:
        raise SystemExit("No images were processed. Check metadata use_image column and any selection filters.")

    out = pd.DataFrame(rows)
    out_path = results_dir / "image_level_compartment_fibrosis_results.csv"
    out.to_csv(out_path, index=False)

    # Store a copy of config used for provenance.
    with open(results_dir / "analysis_config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")

    print(f"\nWrote {out_path}")
    print(f"Wrote overlays to {output_dirs['overlays']}")
    print("Overlay legend: green=interstitial collagen; yellow=perivascular collagen; cyan=vessel mask; red=excluded mask.")


if __name__ == "__main__":
    main()
