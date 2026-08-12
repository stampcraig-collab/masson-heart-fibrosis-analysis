#!/usr/bin/env python3
"""
08_overlay_manual_masks.py

Overlay manually drawn myocardium, vessel, and exclude masks on the original image.

This version can read the perivascular buffer from config/default_config.json.

Config keys checked, in priority order:
1. perivascular_buffer_px
2. perivascular_buffer_pixels
3. perivascular_buffer_um + microns_per_pixel

Command-line options can override the config.

Examples
--------
Use buffer from config/default_config.json:
    python scripts/08_overlay_manual_masks.py --show-vessel-border

Use a specific config file:
    python scripts/08_overlay_manual_masks.py --show-vessel-border --config config/default_config.json

Override the config buffer with 75 pixels:
    python scripts/08_overlay_manual_masks.py --show-vessel-border --vessel-border-px 75

Show myocardium as a bright yellow thicker border:
    python scripts/08_overlay_manual_masks.py --myocardium-contours-only --myocardium-contour-thickness 6

Save the generated vessel border mask:
    python scripts/08_overlay_manual_masks.py --show-vessel-border --save-vessel-border-mask

Preview vessel border as exclusion:
    python scripts/08_overlay_manual_masks.py --show-vessel-border --treat-vessel-border-as-exclude-preview
"""

from pathlib import Path
import argparse
import csv
import json

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]


def split_csv_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_selection_file(path: str | None) -> list[str]:
    if path is None:
        return []
    selection_path = Path(path)
    if not selection_path.exists():
        raise SystemExit(f"Image selection file not found: {selection_path}")
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
    return {t.strip() for t in tokens if str(t).strip()}


def load_metadata(metadata_path: Path) -> pd.DataFrame:
    if not metadata_path.exists():
        raise SystemExit(f"Metadata CSV not found: {metadata_path}")
    return pd.read_csv(metadata_path)


def filter_image_paths(image_paths: list[Path], metadata: pd.DataFrame | None, selected_images: list[str], selected_sample_ids: list[str]) -> list[Path]:
    if not selected_images and not selected_sample_ids:
        return image_paths

    selected_image_tokens: set[str] = set()
    for value in selected_images:
        selected_image_tokens.update(normalise_selector(value))

    selected_sample_tokens = {str(x).strip() for x in selected_sample_ids if str(x).strip()}

    # Build lookup from likely metadata columns to path names/stems
    sample_to_paths: dict[str, list[Path]] = {}
    if metadata is not None and not metadata.empty and selected_sample_tokens:
        for _, row in metadata.iterrows():
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                continue
            candidate_values = [
                row.get("image"), row.get("image_name"), row.get("filename"), row.get("file_name"),
                row.get("image_file"), row.get("image_filename"), row.get("image_path"), row.get("path")
            ]
            matched_paths = []
            for path in image_paths:
                path_tokens = {str(path), path.name, path.stem}
                for val in candidate_values:
                    if pd.isna(val):
                        continue
                    if normalise_selector(str(val)) & path_tokens:
                        matched_paths.append(path)
                        break
            # fallback: if only one path stem equals sample_id, allow direct sample_id=stem matching
            if not matched_paths:
                for path in image_paths:
                    if path.stem == sample_id or path.name == sample_id:
                        matched_paths.append(path)
            if matched_paths:
                sample_to_paths.setdefault(sample_id, [])
                for mp in matched_paths:
                    if mp not in sample_to_paths[sample_id]:
                        sample_to_paths[sample_id].append(mp)

    keep = []
    seen = set()
    for path in image_paths:
        path_tokens = {str(path), path.name, path.stem}
        image_match = bool(selected_image_tokens & path_tokens)
        sample_match = any(path in sample_to_paths.get(sid, []) for sid in selected_sample_tokens)
        if image_match or sample_match:
            key = str(path)
            if key not in seen:
                keep.append(path)
                seen.add(key)

    return keep


def load_config(config_path):
    config_path = Path(config_path)
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_perivascular_buffer_px(config, cli_value=None, default_px=50):
    """
    Determine the perivascular/vessel border radius in pixels.

    Priority:
    1. CLI --vessel-border-px, if supplied
    2. config["perivascular_buffer_px"]
    3. config["perivascular_buffer_pixels"]
    4. config["perivascular_buffer_um"] / config["microns_per_pixel"]
    5. default_px
    """
    if cli_value is not None:
        return int(round(float(cli_value))), "command line --vessel-border-px"

    for key in ("perivascular_buffer_px", "perivascular_buffer_pixels"):
        value = config.get(key)
        if value is not None and value != "":
            return int(round(float(value))), f"config/default_config.json:{key}"

    buffer_um = config.get("perivascular_buffer_um")
    microns_per_pixel = config.get("microns_per_pixel")

    if buffer_um is not None and buffer_um != "" and microns_per_pixel not in (None, "", 0):
        px = float(buffer_um) / float(microns_per_pixel)
        return int(round(px)), "config/default_config.json:perivascular_buffer_um / microns_per_pixel"

    return int(default_px), "script default"


def find_images(image_dir: Path):
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(f"*{ext}")))
        paths.extend(sorted(image_dir.glob(f"*{ext.upper()}")))
    return sorted(set(paths))


def read_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def read_mask(mask_path: Path, shape_hw):
    if not mask_path.exists():
        return np.zeros(shape_hw, dtype=np.uint8)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape_hw, dtype=np.uint8)

    if mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    return (mask > 0).astype(np.uint8)


def mask_to_contours(mask):
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def dilate_mask(mask, radius_px):
    if radius_px <= 0:
        return mask.copy()

    kernel_size = int(radius_px * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate((mask > 0).astype(np.uint8), kernel, iterations=1)


def create_vessel_border_mask(vessel_mask, myocardium_mask, exclude_mask, radius_px, ring_only=True):
    vessel_mask = (vessel_mask > 0).astype(np.uint8)
    myocardium_mask = (myocardium_mask > 0).astype(np.uint8)
    exclude_mask = (exclude_mask > 0).astype(np.uint8)

    dilated = dilate_mask(vessel_mask, radius_px)

    if ring_only:
        border = ((dilated > 0) & (vessel_mask == 0)).astype(np.uint8)
    else:
        border = (dilated > 0).astype(np.uint8)

    if myocardium_mask.sum() > 0:
        border = ((border > 0) & (myocardium_mask > 0)).astype(np.uint8)

    if exclude_mask.sum() > 0:
        border = ((border > 0) & (exclude_mask == 0)).astype(np.uint8)

    return border


def draw_mask(rgb, mask, color_rgb, alpha=0.35, contours_only=False, thickness=3):
    out = rgb.copy()
    if mask.sum() == 0:
        return out

    if contours_only:
        contours = mask_to_contours(mask)
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.drawContours(bgr, contours, -1, tuple(reversed(color_rgb)), thickness)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    pixels = mask > 0
    colour = np.array(color_rgb, dtype=np.float32)
    out[pixels] = ((1 - alpha) * out[pixels].astype(np.float32) + alpha * colour).astype(np.uint8)
    return out


def add_legend(rgb, items):
    out = rgb.copy()
    bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    present_items = [(label, color) for label, color, present in items if present]
    if not present_items:
        return rgb

    x0, y0 = 20, 25
    line_h = 32
    box = 20
    bg_w = 560
    bg_h = 20 + line_h * len(present_items)

    cv2.rectangle(bgr, (10, 10), (10 + bg_w, 10 + bg_h), (255, 255, 255), -1)
    cv2.rectangle(bgr, (10, 10), (10 + bg_w, 10 + bg_h), (0, 0, 0), 1)

    for i, (label, color_rgb) in enumerate(present_items):
        y = y0 + i * line_h
        cv2.rectangle(bgr, (x0, y - 15), (x0 + box, y + 5), tuple(reversed(color_rgb)), -1)
        cv2.putText(
            bgr,
            label,
            (x0 + 32, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def make_contact_sheet(image_paths, output_path, thumb_w=420, cols=3):
    if not image_paths:
        return

    thumbs = []
    for path in image_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue

        h, w = img.shape[:2]
        scale = thumb_w / max(w, 1)
        thumb_h = max(1, int(h * scale))
        thumb = cv2.resize(img, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        label_canvas = np.full((40, thumb_w, 3), 255, dtype=np.uint8)
        cv2.putText(
            label_canvas,
            path.name[:55],
            (8, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        thumbs.append(np.vstack([label_canvas, thumb]))

    if not thumbs:
        return

    max_h = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < max_h:
            pad = np.full((max_h - t.shape[0], t.shape[1], 3), 255, dtype=np.uint8)
            t = np.vstack([t, pad])
        padded.append(t)

    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.full_like(padded[0], 255))
        rows.append(np.hstack(row_imgs))

    sheet = np.vstack(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), sheet)


def main():
    parser = argparse.ArgumentParser(description="Overlay manual masks and optional vessel-derived perivascular border.")
    parser.add_argument("--image-dir", default="images", help="Folder containing original/converted images.")
    parser.add_argument("--mask-dir", default="masks_manual", help="Folder containing manual mask subfolders.")
    parser.add_argument("--output-dir", default="results/manual_mask_overlays", help="Output folder.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to default_config.json.")
    parser.add_argument("--metadata", default=None, help="Metadata CSV used for sample_id-based selection. Defaults to config metadata_csv.")
    parser.add_argument("--image", default=None, help="Overlay one image only. Accepts filename, stem, or path.")
    parser.add_argument("--images", default=None, help="Comma-separated list of images to overlay. Accepts filenames, stems, or paths.")
    parser.add_argument("--image-list", default=None, help="Text file with one image filename/stem/path per line.")
    parser.add_argument("--sample-id", default=None, help="Overlay one sample_id from metadata.csv.")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated list of sample_id values from metadata.csv.")

    parser.add_argument("--alpha", type=float, default=0.35, help="Filled overlay opacity.")
    parser.add_argument("--contours-only", action="store_true", help="Draw all masks as outlines only.")
    parser.add_argument("--myocardium-contours-only", action="store_true", help="Draw the myocardium mask as a border only (bright yellow).")
    parser.add_argument("--contour-thickness", type=int, default=3, help="Contour thickness in pixels.")
    parser.add_argument("--myocardium-contour-thickness", type=int, default=6, help="Contour thickness in pixels for the myocardium border when drawn as a border only.")

    parser.add_argument("--show-vessel-border", action="store_true",
                        help="Show vessel-derived perivascular border/buffer generated from vessel mask.")
    parser.add_argument("--vessel-border-px", type=float, default=None,
                        help="Override perivascular buffer radius in pixels. If omitted, reads config/default_config.json.")
    parser.add_argument("--vessel-border-ring-only", action="store_true",
                        help="Show only the ring outside the vessel, not vessel interior. Recommended.")
    parser.add_argument("--save-vessel-border-mask", action="store_true",
                        help="Save vessel-derived border masks to masks_manual/vessel_border/.")
    parser.add_argument("--treat-vessel-border-as-exclude-preview", action="store_true",
                        help="For preview only, combine vessel border with manual exclude mask and show it as green.")
    parser.add_argument("--no-legend", action="store_true", help="Do not draw legend.")
    args = parser.parse_args()

    config = load_config(args.config)
    vessel_border_px, vessel_border_source = get_perivascular_buffer_px(
        config=config,
        cli_value=args.vessel_border_px,
        default_px=50,
    )

    metadata_path = Path(args.metadata or config.get("metadata_csv", "metadata.csv"))
    selected_images = []
    selected_images.extend(split_csv_arg(args.image))
    selected_images.extend(split_csv_arg(args.images))
    selected_images.extend(read_selection_file(args.image_list))
    selected_sample_ids = []
    selected_sample_ids.extend(split_csv_arg(args.sample_id))
    selected_sample_ids.extend(split_csv_arg(args.sample_ids))
    metadata = None
    if selected_sample_ids:
        metadata = load_metadata(metadata_path)

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    vessel_border_dir = mask_dir / "vessel_border"
    if args.save_vessel_border_mask:
        vessel_border_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    if selected_images or selected_sample_ids:
        image_paths = filter_image_paths(image_paths, metadata, selected_images, selected_sample_ids)

    if not image_paths:
        raise SystemExit("No images matched the requested overlay selection.")

    print(f"Selected images for manual mask overlay: {len(image_paths)}")
    for i, path in enumerate(image_paths, start=1):
        print(f"  {i}. {path.name}")

    print(f"Using vessel/perivascular buffer radius: {vessel_border_px} px")
    print(f"Buffer source: {vessel_border_source}")

    overlay_paths = []
    summary_rows = []

    for image_path in image_paths:
        rgb = read_rgb(image_path)
        h, w = rgb.shape[:2]
        shape_hw = (h, w)
        stem = image_path.stem

        myocardium_mask = read_mask(mask_dir / "myocardium" / f"{stem}.png", shape_hw)
        vessel_mask = read_mask(mask_dir / "vessel" / f"{stem}.png", shape_hw)
        exclude_mask = read_mask(mask_dir / "exclude" / f"{stem}.png", shape_hw)

        vessel_border_mask = np.zeros(shape_hw, dtype=np.uint8)
        if args.show_vessel_border or args.save_vessel_border_mask or args.treat_vessel_border_as_exclude_preview:
            vessel_border_mask = create_vessel_border_mask(
                vessel_mask=vessel_mask,
                myocardium_mask=myocardium_mask,
                exclude_mask=exclude_mask,
                radius_px=vessel_border_px,
                ring_only=args.vessel_border_ring_only,
            )

        if args.save_vessel_border_mask:
            out_border = vessel_border_dir / f"{stem}.png"
            cv2.imwrite(str(out_border), (vessel_border_mask > 0).astype(np.uint8) * 255)

        preview_exclude_mask = exclude_mask.copy()
        if args.treat_vessel_border_as_exclude_preview:
            preview_exclude_mask = ((preview_exclude_mask > 0) | (vessel_border_mask > 0)).astype(np.uint8)

        out = rgb.copy()

        # RGB colour key:
        # yellow myocardium border, cyan vessel, green exclude, magenta vessel border.
        myocardium_border_only = args.myocardium_contours_only or args.contours_only
        myocardium_thickness = args.myocardium_contour_thickness if myocardium_border_only else args.contour_thickness
        out = draw_mask(out, myocardium_mask, (255, 255, 0), args.alpha, myocardium_border_only, myocardium_thickness)
        out = draw_mask(out, vessel_mask, (0, 255, 255), args.alpha, args.contours_only, args.contour_thickness)
        out = draw_mask(out, preview_exclude_mask, (0, 255, 0), args.alpha, args.contours_only, args.contour_thickness)

        if args.show_vessel_border and not args.treat_vessel_border_as_exclude_preview:
            out = draw_mask(out, vessel_border_mask, (255, 0, 255), args.alpha, args.contours_only, args.contour_thickness)

        if not args.no_legend:
            out = add_legend(out, [
                ("myocardium border", (255, 255, 0), myocardium_mask.sum() > 0),
                ("vessel mask", (0, 255, 255), vessel_mask.sum() > 0),
                ("exclude mask", (0, 255, 0), exclude_mask.sum() > 0),
                (f"vessel border/buffer {vessel_border_px}px", (255, 0, 255),
                 args.show_vessel_border and vessel_border_mask.sum() > 0 and not args.treat_vessel_border_as_exclude_preview),
                (f"exclude + vessel buffer {vessel_border_px}px", (0, 255, 0),
                 args.treat_vessel_border_as_exclude_preview and vessel_border_mask.sum() > 0),
            ])

        out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out_path = output_dir / f"{stem}_manual_mask_overlay.png"
        cv2.imwrite(str(out_path), out_bgr)
        overlay_paths.append(out_path)

        summary_rows.append({
            "image": image_path.name,
            "myocardium_mask_px": int(myocardium_mask.sum()),
            "vessel_mask_px": int(vessel_mask.sum()),
            "exclude_mask_px": int(exclude_mask.sum()),
            "vessel_border_mask_px": int(vessel_border_mask.sum()),
            "vessel_border_radius_px": int(vessel_border_px),
            "vessel_border_source": vessel_border_source,
            "vessel_border_ring_only": bool(args.vessel_border_ring_only),
            "saved_vessel_border_mask": bool(args.save_vessel_border_mask),
            "exclude_preview_includes_vessel_border": bool(args.treat_vessel_border_as_exclude_preview),
            "overlay_file": str(out_path),
        })

    summary_csv = output_dir / "manual_mask_overlay_summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    contact_sheet = output_dir / "manual_mask_overlay_contact_sheet.png"
    make_contact_sheet(overlay_paths, contact_sheet)

    print(f"Processed {len(image_paths)} image(s).")
    print(f"Overlays saved to: {output_dir}")
    print(f"Summary CSV: {summary_csv}")
    print(f"Contact sheet: {contact_sheet}")

    if args.save_vessel_border_mask:
        print(f"Vessel-derived border masks saved to: {vessel_border_dir}")


if __name__ == "__main__":
    main()
