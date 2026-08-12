#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

from common import load_config, list_images


def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata.csv from images in the images folder.")
    parser.add_argument("--config", default="config/default_config.json", help="Path to config JSON.")
    parser.add_argument("--image-dir", default=None, help="Folder containing Masson's Trichrome images.")
    parser.add_argument("--output", default="metadata.csv", help="Output metadata CSV path.")
    parser.add_argument("--recursive", action="store_true", help="Search image folder recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing metadata CSV.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    image_dir = Path(args.image_dir or cfg.get("image_dir", "images"))
    output = Path(args.output)

    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} already exists. Use --overwrite if you want to replace it.")

    image_paths = list_images(image_dir, cfg, recursive=args.recursive)
    if not image_paths:
        raise SystemExit(f"No images found in {image_dir}. Put image files there first.")

    rows = []
    for path in image_paths:
        rows.append({
            "image": path.name,
            "sample_id": path.stem,
            "mouse_id": "",
            "group": "",
            "section": "",
            "region": "LV myocardium",
            "stain_batch": "",
            "pixel_size_um": "",
            "use_image": "TRUE",
            "notes": ""
        })

    df = pd.DataFrame(rows)
    df.to_csv(output, index=False)
    print(f"Wrote {output} with {len(df)} images.")
    print("Edit mouse_id, group, section, region, stain_batch and pixel_size_um before final analysis.")


if __name__ == "__main__":
    main()
