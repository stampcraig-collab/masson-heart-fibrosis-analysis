#!/usr/bin/env python

from pathlib import Path
import argparse
from PIL import Image, ImageDraw

# Allow very large histology overlay images.
# This is safe here because these are your own generated images.
Image.MAX_IMAGE_PIXELS = None


def find_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def make_thumbnail(path: Path, thumb_size: int):
    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)

        canvas = Image.new("RGB", (thumb_size, thumb_size + 40), "white")
        x = (thumb_size - img.width) // 2
        y = 0
        canvas.paste(img, (x, y))

        draw = ImageDraw.Draw(canvas)
        label = path.name
        if len(label) > 45:
            label = label[:42] + "..."
        draw.text((5, thumb_size + 8), label, fill="black")

        return canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="results/overlays")
    parser.add_argument("--output", default="results/qc/overlay_contact_sheet.jpg")
    parser.add_argument("--thumb-size", type=int, default=350)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--max-images", type=int, default=0, help="0 = use all images")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = find_images(input_dir)

    if args.max_images and args.max_images > 0:
        image_paths = image_paths[:args.max_images]

    if not image_paths:
        raise SystemExit(f"No overlay images found in {input_dir}")

    thumbs = []
    for i, path in enumerate(image_paths, start=1):
        print(f"Making thumbnail {i}/{len(image_paths)}: {path.name}")
        try:
            thumbs.append(make_thumbnail(path, args.thumb_size))
        except Exception as e:
            print(f"WARNING: skipping {path.name}: {e}")

    if not thumbs:
        raise SystemExit("No thumbnails could be created.")

    columns = max(1, args.columns)
    rows = (len(thumbs) + columns - 1) // columns

    sheet_w = columns * args.thumb_size
    sheet_h = rows * (args.thumb_size + 40)

    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")

    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = col * args.thumb_size
        y = row * (args.thumb_size + 40)
        sheet.paste(thumb, (x, y))

    sheet.save(output_path, quality=90)
    print(f"Saved contact sheet: {output_path}")


if __name__ == "__main__":
    main()
