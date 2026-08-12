from pathlib import Path
import argparse
import numpy as np
import openslide
import tifffile


def print_slide_levels(slide, slide_name):
    print(f"Available pyramid levels for {slide_name}:")
    for i in range(slide.level_count):
        width, height = slide.level_dimensions[i]
        downsample = slide.level_downsamples[i]
        print(f"  Level {i}: {width} x {height}, downsample={downsample:.2f}")


def convert_one_svs(input_path, output_dir, level, overwrite=False):
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_level{level}.tiff"

    if output_path.exists() and not overwrite:
        print(f"Skipping {input_path.name}: output already exists ({output_path.name}). Use --overwrite to replace it.")
        return

    slide = openslide.OpenSlide(str(input_path))

    print("\n" + "=" * 80)
    print(f"Converting: {input_path.name}")
    print_slide_levels(slide, input_path.name)

    if level >= slide.level_count:
        print(f"WARNING: Requested level {level}, but {input_path.name} only has {slide.level_count} level(s). Skipping.")
        return

    width, height = slide.level_dimensions[level]
    print(f"Reading level {level}: {width} x {height}")

    image = slide.read_region((0, 0), level, (width, height)).convert("RGB")
    arr = np.asarray(image)

    tifffile.imwrite(
        output_path,
        arr,
        bigtiff=True,
        compression="deflate"
    )

    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert .svs whole-slide images into TIFF files at a selected pyramid level."
    )
    parser.add_argument("--input-dir", default="slides", help="Folder containing .svs files. Default: slides")
    parser.add_argument("--output-dir", default="images", help="Folder for converted TIFFs. Default: images")
    parser.add_argument("--level", type=int, default=1, help="SVS pyramid level to export. Default: 1")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing TIFF outputs")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    svs_files = sorted(input_dir.glob("*.svs")) + sorted(input_dir.glob("*.SVS"))

    if not svs_files:
        raise FileNotFoundError(f"No .svs files found in {input_dir.resolve()}")

    print(f"Found {len(svs_files)} .svs file(s) in {input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Requested level: {args.level}")

    for svs in svs_files:
        convert_one_svs(svs, args.output_dir, args.level, overwrite=args.overwrite)

    print("\nDone. Converted TIFFs are ready for the fibrosis pipeline.")


if __name__ == "__main__":
    main()
