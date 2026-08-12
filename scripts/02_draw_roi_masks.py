#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from common import load_config, read_image_rgb, resolve_image_path, image_lookup, bool_to_u8


MASK_COLORS_RGB = {
    "myocardium": (0, 200, 255),
    "exclude": (255, 0, 0),
    "vessel": (0, 255, 255),
}

HELP_TEXT = """
Manual ROI drawing controls
---------------------------
Left click             add polygon point
c                      close/fill current polygon into the mask
u                      undo last point
r                      reset current mask for this image
e                      toggle erase mode; fill polygon with background instead of foreground
s                      save current mask
n                      save and move to next image
p                      save and move to previous image
h                      print this help text
q / Esc                quit

Zoom and pan controls
---------------------
Mouse wheel            zoom in/out around the cursor
+ / =                  zoom in
- / _                  zoom out
0                      reset zoom to fit image
Right mouse drag       pan
Middle mouse drag      pan
Arrow keys / WASD      pan

Important vessel rule
---------------------
Draw the vessel structure/lumen/wall, not the collagen ring itself.
Multiple vessels can be added to one vessel mask by drawing several polygons before saving.
"""


class MaskDrawer:
    def __init__(self, rgb: np.ndarray, initial_mask: np.ndarray, mask_type: str, max_display: int = 1200):
        self.rgb = rgb
        self.mask = initial_mask.copy()
        self.mask_type = mask_type
        self.points: list[tuple[int, int]] = []
        self.erase_mode = False
        self.max_display = max_display
        self.window_name = f"Draw {mask_type} mask"
        self.h, self.w = rgb.shape[:2]

        # Initial canvas shows the whole image while respecting max_display.
        fit_scale = min(float(max_display) / float(max(self.w, 1)), float(max_display) / float(max(self.h, 1)), 1.0)
        self.fit_scale = max(fit_scale, 1e-6)
        self.zoom_factor = 1.0
        self.scale = self.fit_scale * self.zoom_factor
        self.canvas_w = max(320, int(round(self.w * self.fit_scale)))
        self.canvas_h = max(240, int(round(self.h * self.fit_scale)))
        self.canvas_w = min(self.canvas_w, max_display)
        self.canvas_h = min(self.canvas_h, max_display)
        self.center_x = self.w / 2.0
        self.center_y = self.h / 2.0

        self.dragging = False
        self.last_mouse_xy: tuple[int, int] | None = None

    def _display_scale(self) -> float:
        return self.fit_scale * self.zoom_factor

    def _view_bounds(self) -> tuple[int, int, int, int, float]:
        scale = self._display_scale()
        view_w = max(1, int(round(self.canvas_w / scale)))
        view_h = max(1, int(round(self.canvas_h / scale)))
        view_w = min(view_w, self.w)
        view_h = min(view_h, self.h)

        x0 = int(round(self.center_x - view_w / 2))
        y0 = int(round(self.center_y - view_h / 2))
        x0 = max(0, min(self.w - view_w, x0))
        y0 = max(0, min(self.h - view_h, y0))
        x1 = x0 + view_w
        y1 = y0 + view_h
        self.center_x = x0 + view_w / 2
        self.center_y = y0 + view_h / 2
        return x0, y0, x1, y1, scale

    def display_to_image(self, x: int, y: int) -> tuple[int, int]:
        x0, y0, _, _, scale = self._view_bounds()
        ix = int(round(x0 + x / scale))
        iy = int(round(y0 + y / scale))
        ix = max(0, min(self.w - 1, ix))
        iy = max(0, min(self.h - 1, iy))
        return ix, iy

    def image_to_display(self, ix: int, iy: int, bounds: tuple[int, int, int, int, float] | None = None) -> tuple[int, int]:
        if bounds is None:
            bounds = self._view_bounds()
        x0, y0, _, _, scale = bounds
        return int(round((ix - x0) * scale)), int(round((iy - y0) * scale))

    def pan_display_pixels(self, dx: float, dy: float) -> None:
        scale = self._display_scale()
        self.center_x -= dx / scale
        self.center_y -= dy / scale
        self._view_bounds()

    def zoom(self, factor: float, anchor_display_xy: tuple[int, int] | None = None) -> None:
        if factor <= 0:
            return
        if anchor_display_xy is None:
            anchor_display_xy = (self.canvas_w // 2, self.canvas_h // 2)
        anchor_img_before = self.display_to_image(*anchor_display_xy)
        old_zoom = self.zoom_factor
        self.zoom_factor = float(np.clip(self.zoom_factor * factor, 1.0, 40.0))
        if abs(self.zoom_factor - old_zoom) < 1e-9:
            return
        scale = self._display_scale()
        ax, ay = anchor_display_xy
        self.center_x = anchor_img_before[0] + (self.canvas_w / 2.0 - ax) / scale
        self.center_y = anchor_img_before[1] + (self.canvas_h / 2.0 - ay) / scale
        self._view_bounds()

    def reset_zoom(self) -> None:
        self.zoom_factor = 1.0
        self.center_x = self.w / 2.0
        self.center_y = self.h / 2.0
        self._view_bounds()

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(self.display_to_image(x, y))
        elif event in {cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN}:
            self.dragging = True
            self.last_mouse_xy = (x, y)
        elif event in {cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP}:
            self.dragging = False
            self.last_mouse_xy = None
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.last_mouse_xy is not None:
            lx, ly = self.last_mouse_xy
            self.pan_display_pixels(x - lx, y - ly)
            self.last_mouse_xy = (x, y)
        elif event == cv2.EVENT_MOUSEWHEEL:
            # OpenCV reports wheel direction in the high-order word of flags.
            if flags > 0:
                self.zoom(1.25, (x, y))
            else:
                self.zoom(1 / 1.25, (x, y))

    def fill_polygon(self) -> None:
        if len(self.points) < 3:
            print("Need at least 3 points to fill a polygon.")
            return
        pts = np.array(self.points, dtype=np.int32).reshape((-1, 1, 2))
        value = 0 if self.erase_mode else 255
        mask_u8 = bool_to_u8(self.mask)
        cv2.fillPoly(mask_u8, [pts], value)
        self.mask = mask_u8 > 0
        mode = "erased" if self.erase_mode else "added"
        print(f"Polygon {mode}.")
        self.points = []

    def render(self) -> np.ndarray:
        bounds = self._view_bounds()
        x0, y0, x1, y1, scale = bounds
        base = self.rgb[y0:y1, x0:x1].copy()
        crop_mask = self.mask[y0:y1, x0:x1]
        color = np.array(MASK_COLORS_RGB.get(self.mask_type, (0, 255, 0)), dtype=np.uint8)

        if crop_mask.any():
            overlay = base.copy()
            overlay[crop_mask] = color
            base = cv2.addWeighted(base, 0.65, overlay, 0.35, 0)
            contours, _ = cv2.findContours(bool_to_u8(crop_mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(base, contours, -1, tuple(int(c) for c in color.tolist()), max(1, int(round(2 / scale))))

        # Resize only the current crop, not the entire image. This is much faster for large TIFFs.
        canvas = cv2.resize(base, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        disp_pts = [self.image_to_display(ix, iy, bounds) for ix, iy in self.points]
        for i, pt in enumerate(disp_pts):
            cv2.circle(canvas, pt, 4, (255, 255, 255), -1)
            cv2.circle(canvas, pt, 5, (0, 0, 0), 1)
            if i > 0:
                cv2.line(canvas, disp_pts[i - 1], pt, (255, 255, 255), 2)
        if len(disp_pts) > 2:
            cv2.line(canvas, disp_pts[-1], disp_pts[0], (255, 255, 255), 1)

        text = (
            f"{self.mask_type} | zoom={self.zoom_factor:.1f}x | erase={self.erase_mode} | "
            f"points={len(self.points)} | wheel/+/- zoom | right-drag pan | c fill | s save | n next"
        )
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def load_existing_mask(mask_path: Path, shape_hw: tuple[int, int]) -> np.ndarray:
    if not mask_path.exists():
        return np.zeros(shape_hw, dtype=bool)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return np.zeros(shape_hw, dtype=bool)
    h, w = shape_hw
    if mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask > 0


def save_mask(mask_path: Path, mask: np.ndarray) -> None:
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(mask_path), bool_to_u8(mask))
    print(f"Saved {mask_path}")


def split_csv_arg(value: str | None) -> list[str]:
    """Split a comma-separated command-line argument into clean tokens."""
    if value is None:
        return []
    return [x.strip() for x in str(value).split(",") if x.strip()]


def read_selection_file(path: str | None) -> list[str]:
    """Read one image name/stem/path per line from a text file."""
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
    """
    Return equivalent selector tokens for matching an image.

    Allows matching by:
    - exact filename, e.g. M01_level1.tiff
    - stem, e.g. M01_level1
    - path string, e.g. images/M01_level1.tiff
    """
    p = Path(str(value))
    tokens = {
        str(value),
        str(p),
        p.name,
        p.stem,
    }
    return {t.strip() for t in tokens if str(t).strip()}


def filter_image_paths(
    image_paths: list[Path],
    metadata: pd.DataFrame,
    selected_images: list[str],
    selected_sample_ids: list[str],
) -> list[Path]:
    """
    Filter image paths using image filename/stem/path and/or sample_id.

    If no selectors are supplied, returns all paths.
    """
    if not selected_images and not selected_sample_ids:
        return image_paths

    selected_image_tokens: set[str] = set()
    for value in selected_images:
        selected_image_tokens.update(normalise_selector(value))

    selected_sample_tokens = {str(x).strip() for x in selected_sample_ids if str(x).strip()}

    keep_paths: list[Path] = []
    unmatched_image_tokens = set(selected_image_tokens)
    unmatched_sample_tokens = set(selected_sample_tokens)

    for idx, path in enumerate(image_paths):
        row = metadata.iloc[idx] if idx < len(metadata) else {}
        sample_id = str(row.get("sample_id", path.stem)).strip() if hasattr(row, "get") else path.stem

        image_tokens = {
            str(path),
            path.name,
            path.stem,
            str(Path(path.name)),
        }

        image_match = bool(selected_image_tokens & image_tokens)
        sample_match = sample_id in selected_sample_tokens

        if image_match or sample_match:
            keep_paths.append(path)
            unmatched_image_tokens -= image_tokens
            unmatched_sample_tokens.discard(sample_id)

    if selected_image_tokens and unmatched_image_tokens:
        # Only report compact unmatched terms; full token expansion can be noisy.
        requested = set(selected_images)
        matched_simple = {p.name for p in keep_paths} | {p.stem for p in keep_paths} | {str(p) for p in keep_paths}
        still_unmatched = sorted(x for x in requested if x not in matched_simple and Path(x).name not in matched_simple and Path(x).stem not in matched_simple)
        if still_unmatched:
            print("Warning: these requested image selectors did not match any image:")
            for x in still_unmatched:
                print(f"  - {x}")

    if selected_sample_tokens and unmatched_sample_tokens:
        print("Warning: these requested sample_id values did not match metadata:")
        for x in sorted(unmatched_sample_tokens):
            print(f"  - {x}")

    return keep_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw manual myocardium, exclude, or vessel masks using a Python/OpenCV GUI with zoom/pan.")
    parser.add_argument("--config", default="config/default_config.json")
    parser.add_argument("--metadata", default=None, help="Metadata CSV. Defaults to config metadata_csv.")
    parser.add_argument("--image-dir", default=None, help="Image directory. Defaults to config image_dir.")
    parser.add_argument("--mask-type", required=True, choices=["myocardium", "exclude", "vessel"])
    parser.add_argument("--max-display", type=int, default=1200, help="Maximum initial display dimension in pixels.")
    parser.add_argument("--image", default=None, help="Annotate one image only. Accepts filename, stem, or path.")
    parser.add_argument("--images", default=None, help="Comma-separated list of images to annotate. Accepts filenames, stems, or paths.")
    parser.add_argument("--image-list", default=None, help="Text file containing one image filename/stem/path per line.")
    parser.add_argument("--sample-id", default=None, help="Annotate one sample_id from metadata.csv.")
    parser.add_argument("--sample-ids", default=None, help="Comma-separated list of sample_id values from metadata.csv.")
    parser.add_argument("--start-at", default=None, help="Start the masking GUI at this image filename/stem/path after filtering.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    metadata_path = Path(args.metadata or cfg.get("metadata_csv", "metadata.csv"))
    image_dir = Path(args.image_dir or cfg.get("image_dir", "images"))
    mask_root = Path(cfg.get("manual_mask_dir", "masks_manual"))
    mask_dir = mask_root / args.mask_type
    mask_dir.mkdir(parents=True, exist_ok=True)

    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
    else:
        from common import list_images
        paths = list_images(image_dir, cfg)
        metadata = pd.DataFrame({"image": [p.name for p in paths], "sample_id": [p.stem for p in paths], "use_image": ["TRUE"] * len(paths)})

    lookup = image_lookup(image_dir, cfg)
    image_paths = []
    for _, row in metadata.iterrows():
        use = str(row.get("use_image", "TRUE")).strip().lower() not in {"false", "0", "no", "n"}
        if not use:
            continue
        image_paths.append(resolve_image_path(row, image_dir, cfg, lookup))

    selected_images = []
    selected_images.extend(split_csv_arg(args.image))
    selected_images.extend(split_csv_arg(args.images))
    selected_images.extend(read_selection_file(args.image_list))

    selected_sample_ids = []
    selected_sample_ids.extend(split_csv_arg(args.sample_id))
    selected_sample_ids.extend(split_csv_arg(args.sample_ids))

    if selected_images or selected_sample_ids:
        image_paths = filter_image_paths(
            image_paths=image_paths,
            metadata=metadata.loc[
                metadata.get("use_image", "TRUE").astype(str).str.strip().str.lower().map(lambda x: x not in {"false", "0", "no", "n"})
            ].reset_index(drop=True) if "use_image" in metadata.columns else metadata.reset_index(drop=True),
            selected_images=selected_images,
            selected_sample_ids=selected_sample_ids,
        )

    if not image_paths:
        raise SystemExit("No images available to annotate after filtering.")

    print(f"Images selected for {args.mask_type} masking: {len(image_paths)}")
    for i, path in enumerate(image_paths, start=1):
        print(f"  {i}. {path.name}")

    print(HELP_TEXT)
    idx = 0

    if args.start_at:
        start_tokens = normalise_selector(args.start_at)
        for i, path in enumerate(image_paths):
            if start_tokens & {str(path), path.name, path.stem}:
                idx = i
                break
        else:
            print(f"Warning: --start-at did not match selected images: {args.start_at}")
    while 0 <= idx < len(image_paths):
        path = image_paths[idx]
        print(f"\nImage {idx + 1}/{len(image_paths)}: {path.name}")
        rgb = read_image_rgb(path)
        mask_path = mask_dir / f"{path.stem}.png"
        initial_mask = load_existing_mask(mask_path, rgb.shape[:2])
        drawer = MaskDrawer(rgb, initial_mask, args.mask_type, args.max_display)

        cv2.namedWindow(drawer.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(drawer.window_name, drawer.canvas_w, drawer.canvas_h)
        cv2.setMouseCallback(drawer.window_name, drawer.mouse_callback)

        move = 0
        while True:
            cv2.imshow(drawer.window_name, drawer.render())
            key = cv2.waitKey(30) & 0xFF
            if key == 255:
                continue
            if key in [ord('q'), 27]:
                cv2.destroyAllWindows()
                return
            if key == ord('h'):
                print(HELP_TEXT)
            elif key == ord('c'):
                drawer.fill_polygon()
            elif key == ord('u'):
                if drawer.points:
                    drawer.points.pop()
            elif key == ord('r'):
                drawer.mask[:] = False
                drawer.points = []
                print("Mask reset for this image. Press s to save the reset mask.")
            elif key == ord('e'):
                drawer.erase_mode = not drawer.erase_mode
                print(f"Erase mode: {drawer.erase_mode}")
            elif key in [ord('+'), ord('=')]:
                drawer.zoom(1.25)
            elif key in [ord('-'), ord('_')]:
                drawer.zoom(1 / 1.25)
            elif key == ord('0'):
                drawer.reset_zoom()
            elif key in [ord('a'), 81]:
                drawer.pan_display_pixels(80, 0)
            elif key in [ord('d'), 83]:
                drawer.pan_display_pixels(-80, 0)
            elif key in [ord('w'), 82]:
                drawer.pan_display_pixels(0, 80)
            elif key in [ord('s')]:
                save_mask(mask_path, drawer.mask)
            elif key in [ord('x'), 84]:  # x or down arrow for pan down, because s is save
                drawer.pan_display_pixels(0, -80)
            elif key == ord('n'):
                save_mask(mask_path, drawer.mask)
                move = 1
                break
            elif key == ord('p'):
                save_mask(mask_path, drawer.mask)
                move = -1
                break

        cv2.destroyWindow(drawer.window_name)
        idx += move

    cv2.destroyAllWindows()
    print("Finished all images.")


if __name__ == "__main__":
    main()
