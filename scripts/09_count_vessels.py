#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import argparse
import csv

import cv2
import numpy as np
import pandas as pd


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]


def split_csv_arg(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_selection_file(path: str | None) -> list[str]:
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Selection file not found: {p}")
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def normalise_selector(value: str) -> set[str]:
    p = Path(str(value))
    return {x for x in {str(value), str(p), p.name, p.stem} if x}


def find_images(image_dir: Path) -> list[Path]:
    paths = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(f"*{ext}")))
        paths.extend(sorted(image_dir.glob(f"*{ext.upper()}")))
    return sorted(set(paths))


def filter_images(image_paths: list[Path], selected_images: list[str]) -> list[Path]:
    if not selected_images:
        return image_paths

    selected_tokens: set[str] = set()
    for item in selected_images:
        selected_tokens.update(normalise_selector(item))

    kept = []
    for path in image_paths:
        tokens = {str(path), path.name, path.stem}
        if selected_tokens & tokens:
            kept.append(path)

    return kept


def read_mask(mask_path: Path) -> np.ndarray | None:
    if not mask_path.exists():
        return None
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    return mask > 0


def count_vessels(mask: np.ndarray, min_area: int) -> dict:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)

    label_image = np.zeros_like(labels, dtype=np.uint16)
    areas = []
    centroids = []
    new_label = 1

    for old_label in range(1, stats.shape[0]):
        area = int(stats[old_label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        pixels = labels == old_label
        ys, xs = np.where(pixels)
        if len(xs) == 0:
            continue

        label_image[pixels] = new_label
        areas.append(area)
        centroids.append((float(xs.mean()), float(ys.mean())))
        new_label += 1

    return {
        "vessel_count": len(areas),
        "total_vessel_mask_area_px": int(np.count_nonzero(label_image)),
        "component_areas": areas,
        "component_centroids": centroids,
        "label_image": label_image,
    }


def make_labelled_overlay(image_path: Path, mask: np.ndarray, count_result: dict, output_path: Path):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        h, w = mask.shape
        bgr = np.full((h, w, 3), 255, dtype=np.uint8)

    if bgr.shape[:2] != mask.shape:
        bgr = cv2.resize(bgr, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)

    overlay = bgr.copy()
    vessel_pixels = count_result["label_image"] > 0
    overlay[vessel_pixels] = (0, 255, 255)
    out = cv2.addWeighted(bgr, 0.7, overlay, 0.3, 0)

    for i, (cx, cy) in enumerate(count_result["component_centroids"], start=1):
        label_mask = (count_result["label_image"] == i).astype(np.uint8) * 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, (0, 255, 255), 3)
        cv2.putText(out, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 4, cv2.LINE_AA)
        cv2.putText(out, str(i), (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2, cv2.LINE_AA)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), out)


def main():
    parser = argparse.ArgumentParser(description="Count vessels from manually drawn vessel masks.")
    parser.add_argument("--image-dir", default="images", help="Folder containing source images.")
    parser.add_argument("--mask-dir", default="masks_manual/vessel", help="Folder containing vessel masks.")
    parser.add_argument("--output-dir", default="results/vessel_counts", help="Output folder.")
    parser.add_argument("--min-vessel-area-px", type=int, default=50, help="Minimum connected-component area to count as a vessel.")
    parser.add_argument("--image", default=None, help="Count one image only. Accepts filename, stem, or path.")
    parser.add_argument("--images", default=None, help="Comma-separated list of images to count.")
    parser.add_argument("--image-list", default=None, help="Text file containing one image filename/stem/path per line.")
    parser.add_argument("--make-overlays", action="store_true", help="Create labelled vessel overlays.")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_images = []
    selected_images.extend(split_csv_arg(args.image))
    selected_images.extend(split_csv_arg(args.images))
    selected_images.extend(read_selection_file(args.image_list))

    image_paths = filter_images(find_images(image_dir), selected_images)

    if not image_paths:
        raise SystemExit("No images found or selected.")

    rows = []

    for image_path in image_paths:
        stem = image_path.stem
        mask_path = mask_dir / f"{stem}.png"
        mask = read_mask(mask_path)

        if mask is None:
            row = {
                "image": image_path.name,
                "image_stem": stem,
                "vessel_mask_found": False,
                "vessel_count": 0,
                "total_vessel_mask_area_px": 0,
                "mean_vessel_area_px": 0,
                "median_vessel_area_px": 0,
                "min_vessel_area_px": 0,
                "max_vessel_area_px": 0,
                "min_vessel_area_threshold_px": args.min_vessel_area_px,
                "mask_path": str(mask_path),
            }
            print(f"{image_path.name}: no vessel mask found")
            rows.append(row)
            continue

        result = count_vessels(mask, args.min_vessel_area_px)
        areas = result["component_areas"]

        row = {
            "image": image_path.name,
            "image_stem": stem,
            "vessel_mask_found": True,
            "vessel_count": int(result["vessel_count"]),
            "total_vessel_mask_area_px": int(result["total_vessel_mask_area_px"]),
            "mean_vessel_area_px": float(np.mean(areas)) if areas else 0,
            "median_vessel_area_px": float(np.median(areas)) if areas else 0,
            "min_vessel_area_px": int(min(areas)) if areas else 0,
            "max_vessel_area_px": int(max(areas)) if areas else 0,
            "min_vessel_area_threshold_px": args.min_vessel_area_px,
            "mask_path": str(mask_path),
        }
        rows.append(row)

        print(f"{image_path.name}: {row['vessel_count']} vessel(s)")

        if args.make_overlays:
            overlay_path = output_dir / "labelled_overlays" / f"{stem}_vessel_count_overlay.png"
            make_labelled_overlay(image_path, mask, result, overlay_path)

    by_image = pd.DataFrame(rows)
    by_image_path = output_dir / "vessel_counts_by_image.csv"
    by_image.to_csv(by_image_path, index=False)

    summary = {
        "n_images": int(len(by_image)),
        "n_images_with_vessel_masks": int(by_image["vessel_mask_found"].sum()) if len(by_image) else 0,
        "total_vessels_counted": int(by_image["vessel_count"].sum()) if len(by_image) else 0,
        "mean_vessels_per_image": float(by_image["vessel_count"].mean()) if len(by_image) else 0,
        "median_vessels_per_image": float(by_image["vessel_count"].median()) if len(by_image) else 0,
        "min_vessel_area_threshold_px": int(args.min_vessel_area_px),
    }

    summary_path = output_dir / "vessel_count_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print()
    print(f"Wrote: {by_image_path}")
    print(f"Wrote: {summary_path}")
    if args.make_overlays:
        print(f"Wrote labelled overlays to: {output_dir / 'labelled_overlays'}")


if __name__ == "__main__":
    main()
