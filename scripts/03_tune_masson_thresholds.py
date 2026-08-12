#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

from common import (
    load_config, save_config, read_image_rgb, write_rgb, collagen_mask_hsv,
    auto_tissue_mask, load_manual_mask, resolve_image_path, image_lookup, bool_to_u8
)


HELP_TEXT = """
Interactive threshold controls
------------------------------
Slider controls         tune HSV collagen threshold
w                       write current settings to the config JSON
q / Esc                 quit
h                       print this help text

Zoom and pan controls
---------------------
Mouse wheel             zoom in/out around cursor
+ / =                   zoom in
- / _                   zoom out
0                       reset zoom to fit preview
Right mouse drag        pan
Middle mouse drag       pan
Arrow keys / A/D/S/X    pan; w is reserved for writing settings to config

Speed controls
--------------
The interactive tool uses a downsampled preview by default. This keeps the GUI responsive.
Increase --preview-max-dim for more detail, or decrease it if the window is still slow.
"""


def resize_to_max_dim(rgb: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    h, w = rgb.shape[:2]
    biggest = max(h, w)
    if max_dim <= 0 or biggest <= max_dim:
        return rgb, 1.0
    scale = float(max_dim) / float(biggest)
    out = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return out, scale


def resize_mask(mask: np.ndarray | None, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if mask is None:
        return None
    h, w = shape_hw
    if mask.shape[:2] == (h, w):
        return mask
    return cv2.resize(bool_to_u8(mask), (w, h), interpolation=cv2.INTER_NEAREST) > 0


def make_preview(rgb: np.ndarray, collagen: np.ndarray, analysis_mask: np.ndarray | None = None) -> np.ndarray:
    original = rgb.copy()
    overlay = rgb.copy()
    if analysis_mask is not None:
        overlay[~analysis_mask] = (overlay[~analysis_mask] * 0.35).astype(np.uint8)
    overlay[collagen] = np.array([255, 255, 0], dtype=np.uint8)
    blended = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
    collagen_rgb = np.zeros_like(rgb)
    collagen_rgb[collagen] = [255, 255, 255]
    preview = np.concatenate([original, blended, collagen_rgb], axis=1)
    return preview


class ZoomPanViewer:
    def __init__(self, max_display: int = 1800):
        self.max_display = max_display
        self.image: np.ndarray | None = None
        self.window_name = "Interactive Masson threshold tuning"
        self.fit_scale = 1.0
        self.zoom_factor = 1.0
        self.canvas_w = 1000
        self.canvas_h = 700
        self.center_x = 0.0
        self.center_y = 0.0
        self.dragging = False
        self.last_mouse_xy: tuple[int, int] | None = None

    def set_image(self, rgb: np.ndarray) -> None:
        old_center = (self.center_x, self.center_y)
        old_zoom = self.zoom_factor
        self.image = rgb
        h, w = rgb.shape[:2]
        self.fit_scale = min(float(self.max_display) / float(max(w, 1)), float(self.max_display) / float(max(h, 1)), 1.0)
        self.fit_scale = max(self.fit_scale, 1e-6)
        self.canvas_w = min(max(480, int(round(w * self.fit_scale))), self.max_display)
        self.canvas_h = min(max(320, int(round(h * self.fit_scale))), self.max_display)
        if old_center == (0.0, 0.0):
            self.center_x = w / 2.0
            self.center_y = h / 2.0
            self.zoom_factor = 1.0
        else:
            self.center_x = max(0, min(w - 1, old_center[0]))
            self.center_y = max(0, min(h - 1, old_center[1]))
            self.zoom_factor = float(np.clip(old_zoom, 1.0, 40.0))
        self._view_bounds()

    def _display_scale(self) -> float:
        return self.fit_scale * self.zoom_factor

    def _view_bounds(self) -> tuple[int, int, int, int, float]:
        if self.image is None:
            return 0, 0, 1, 1, 1.0
        h, w = self.image.shape[:2]
        scale = self._display_scale()
        view_w = max(1, int(round(self.canvas_w / scale)))
        view_h = max(1, int(round(self.canvas_h / scale)))
        view_w = min(view_w, w)
        view_h = min(view_h, h)
        x0 = int(round(self.center_x - view_w / 2))
        y0 = int(round(self.center_y - view_h / 2))
        x0 = max(0, min(w - view_w, x0))
        y0 = max(0, min(h - view_h, y0))
        x1 = x0 + view_w
        y1 = y0 + view_h
        self.center_x = x0 + view_w / 2
        self.center_y = y0 + view_h / 2
        return x0, y0, x1, y1, scale

    def display_to_image(self, x: int, y: int) -> tuple[int, int]:
        if self.image is None:
            return 0, 0
        x0, y0, _, _, scale = self._view_bounds()
        h, w = self.image.shape[:2]
        ix = int(round(x0 + x / scale))
        iy = int(round(y0 + y / scale))
        return max(0, min(w - 1, ix)), max(0, min(h - 1, iy))

    def pan_display_pixels(self, dx: float, dy: float) -> None:
        scale = self._display_scale()
        self.center_x -= dx / scale
        self.center_y -= dy / scale
        self._view_bounds()

    def zoom(self, factor: float, anchor_display_xy: tuple[int, int] | None = None) -> None:
        if self.image is None or factor <= 0:
            return
        if anchor_display_xy is None:
            anchor_display_xy = (self.canvas_w // 2, self.canvas_h // 2)
        anchor_before = self.display_to_image(*anchor_display_xy)
        old_zoom = self.zoom_factor
        self.zoom_factor = float(np.clip(self.zoom_factor * factor, 1.0, 40.0))
        if abs(self.zoom_factor - old_zoom) < 1e-9:
            return
        scale = self._display_scale()
        ax, ay = anchor_display_xy
        self.center_x = anchor_before[0] + (self.canvas_w / 2.0 - ax) / scale
        self.center_y = anchor_before[1] + (self.canvas_h / 2.0 - ay) / scale
        self._view_bounds()

    def reset_zoom(self) -> None:
        if self.image is None:
            return
        h, w = self.image.shape[:2]
        self.zoom_factor = 1.0
        self.center_x = w / 2.0
        self.center_y = h / 2.0
        self._view_bounds()

    def mouse_callback(self, event, x, y, flags, param):
        if event in {cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN}:
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
            if flags > 0:
                self.zoom(1.25, (x, y))
            else:
                self.zoom(1 / 1.25, (x, y))

    def render(self) -> np.ndarray:
        if self.image is None:
            return np.zeros((self.canvas_h, self.canvas_w, 3), dtype=np.uint8)
        x0, y0, x1, y1, scale = self._view_bounds()
        crop = self.image[y0:y1, x0:x1]
        canvas = cv2.resize(crop, (self.canvas_w, self.canvas_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        text = f"zoom={self.zoom_factor:.1f}x | wheel/+/- zoom | right-drag pan | w write config | q quit"
        cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 36), (0, 0, 0), -1)
        cv2.putText(canvas, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)


def build_analysis_mask_for_preview(full_rgb: np.ndarray, preview_rgb: np.ndarray, cfg: dict, image_stem: str) -> np.ndarray:
    h, w = full_rgb.shape[:2]
    ph, pw = preview_rgb.shape[:2]
    mask_root = Path(cfg.get("manual_mask_dir", "masks_manual"))
    myocardium_full = load_manual_mask(mask_root, "myocardium", image_stem, (h, w))
    exclude_full = load_manual_mask(mask_root, "exclude", image_stem, (h, w))

    myocardium = resize_mask(myocardium_full, (ph, pw))
    exclude = resize_mask(exclude_full, (ph, pw))
    analysis_mask = myocardium if myocardium is not None else auto_tissue_mask(preview_rgb, cfg)
    if exclude is not None:
        analysis_mask = analysis_mask & ~exclude
    return analysis_mask




def _auto_candidate_mask(rgb: np.ndarray, analysis_mask: np.ndarray, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return likely blue/green collagen candidate pixels for automatic HSV estimation."""
    auto = cfg.get("auto_threshold", {})
    hue_low = int(auto.get("broad_hue_low", 45))
    hue_high = int(auto.get("broad_hue_high", 150))
    sat_min = int(auto.get("min_saturation", 25))
    value_min = int(auto.get("min_value", 20))
    blue_green_minus_red = int(auto.get("blue_green_minus_red_min", 8))

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    if hue_low <= hue_high:
        hue_mask = (h >= hue_low) & (h <= hue_high)
    else:
        hue_mask = (h >= hue_low) | (h <= hue_high)

    # Masson's collagen is blue/green. This reduces red myocardium and pink stain carry-over.
    colour_dominance = ((b - r) >= blue_green_minus_red) | ((g - r) >= blue_green_minus_red)
    candidate = analysis_mask & hue_mask & (s >= sat_min) & (v >= value_min) & colour_dominance
    return candidate, hsv


def _candidate_preview(rgb: np.ndarray, candidate: np.ndarray, analysis_mask: np.ndarray) -> np.ndarray:
    overlay = rgb.copy()
    overlay[~analysis_mask] = (overlay[~analysis_mask] * 0.35).astype(np.uint8)
    overlay[candidate] = np.array([255, 255, 0], dtype=np.uint8)
    blended = cv2.addWeighted(rgb, 0.65, overlay, 0.35, 0)
    candidate_rgb = np.zeros_like(rgb)
    candidate_rgb[candidate] = [255, 255, 255]
    return np.concatenate([rgb, blended, candidate_rgb], axis=1)


def auto_tune_thresholds(
    cfg: dict,
    metadata_path: Path,
    image_dir: Path,
    output_dir: Path,
    max_images: int,
    preview_max_dim: int,
    config_path: Path,
    write_config: bool,
) -> None:
    """Estimate collagen HSV thresholds automatically from likely blue/green pixels."""
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    candidate_dir = output_dir / "candidate_previews"
    final_preview_dir = output_dir / "threshold_previews"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    final_preview_dir.mkdir(parents=True, exist_ok=True)

    auto = cfg.setdefault("auto_threshold", {})
    q_low = float(auto.get("quantile_low", 2.0))
    q_high = float(auto.get("quantile_high", 98.0))
    sat_q = float(auto.get("saturation_low_quantile", 5.0))
    value_q = float(auto.get("value_low_quantile", 2.0))
    min_candidate_pixels = int(auto.get("min_candidate_pixels_total", 500))

    metadata = pd.read_csv(metadata_path)
    lookup = image_lookup(image_dir, cfg)

    all_h = []
    all_s = []
    all_v = []
    rows = []
    count = 0

    for _, row in metadata.iterrows():
        if count >= max_images:
            break
        use = str(row.get("use_image", "TRUE")).strip().lower() not in {"false", "0", "no", "n"}
        if not use:
            continue

        image_path = resolve_image_path(row, image_dir, cfg, lookup)
        rgb_full = read_image_rgb(image_path)
        rgb, scale = resize_to_max_dim(rgb_full, preview_max_dim)
        analysis_mask = build_analysis_mask_for_preview(rgb_full, rgb, cfg, image_path.stem)
        candidate, hsv = _auto_candidate_mask(rgb, analysis_mask, cfg)

        candidate_count = int(np.count_nonzero(candidate))
        analysis_count = int(np.count_nonzero(analysis_mask))
        candidate_fraction = candidate_count / analysis_count if analysis_count else 0.0

        if candidate_count > 0:
            h, s, v = cv2.split(hsv)
            all_h.append(h[candidate])
            all_s.append(s[candidate])
            all_v.append(v[candidate])

        preview = _candidate_preview(rgb, candidate, analysis_mask)
        max_w = 3000
        if preview.shape[1] > max_w:
            shrink = max_w / preview.shape[1]
            preview = cv2.resize(preview, (max_w, int(preview.shape[0] * shrink)), interpolation=cv2.INTER_AREA)
        write_rgb(candidate_dir / f"{image_path.stem}_auto_candidate_preview.png", preview)

        rows.append({
            "image": image_path.name,
            "preview_scale": scale,
            "analysis_pixels_preview": analysis_count,
            "candidate_pixels_preview": candidate_count,
            "candidate_fraction_preview": candidate_fraction,
        })
        print(f"Auto-threshold candidate preview: {image_path.name} | candidates={candidate_count} | fraction={candidate_fraction:.4f}")
        count += 1

    if not rows:
        raise SystemExit("No images were available for automatic thresholding. Check metadata use_image column.")

    summary = pd.DataFrame(rows)
    summary_path = output_dir / "auto_threshold_candidate_summary.csv"
    summary.to_csv(summary_path, index=False)

    if not all_h:
        raise SystemExit("Automatic thresholding found no candidate blue/green pixels. Use interactive tuning instead.")

    h_values = np.concatenate(all_h)
    s_values = np.concatenate(all_s)
    v_values = np.concatenate(all_v)
    total_candidates = int(h_values.size)

    if total_candidates < min_candidate_pixels:
        print(f"Warning: only {total_candidates} candidate pixels found; automatic thresholds may be unstable.")
        print("Consider using --max-images with more representative images or use --interactive.")

    broad_low = int(auto.get("broad_hue_low", 45))
    broad_high = int(auto.get("broad_hue_high", 150))
    hue_low = int(np.floor(np.percentile(h_values, q_low)))
    hue_high = int(np.ceil(np.percentile(h_values, q_high)))
    hue_low = max(0, max(broad_low, hue_low))
    hue_high = min(179, min(broad_high, hue_high))

    # Guard against pathologically narrow/invalid values.
    if hue_high <= hue_low:
        hue_low = broad_low
        hue_high = broad_high

    saturation_low = int(np.floor(np.percentile(s_values, sat_q)))
    value_low = int(np.floor(np.percentile(v_values, value_q)))
    saturation_low = max(0, min(255, saturation_low))
    value_low = max(0, min(255, value_low))

    cfg["collagen_hsv_ranges"] = [{
        "name": "auto_blue_green_collagen",
        "hue_low": hue_low,
        "hue_high": hue_high,
        "saturation_low": saturation_low,
        "saturation_high": 255,
        "value_low": value_low,
        "value_high": 255,
    }]

    auto_result = {
        "total_candidate_pixels": total_candidates,
        "hue_low": hue_low,
        "hue_high": hue_high,
        "saturation_low": saturation_low,
        "value_low": value_low,
        "quantile_low": q_low,
        "quantile_high": q_high,
        "saturation_low_quantile": sat_q,
        "value_low_quantile": value_q,
        "preview_max_dim": preview_max_dim,
        "max_images": max_images,
    }
    with open(output_dir / "auto_threshold_result.json", "w", encoding="utf-8") as f:
        import json as _json
        _json.dump(auto_result, f, indent=2)
        f.write("\n")

    suggested_config = output_dir / "suggested_config_auto_threshold.json"
    save_config(cfg, suggested_config)

    if write_config:
        save_config(cfg, config_path)
        print(f"Wrote automatic HSV thresholds to config: {config_path}")
    else:
        print(f"Did not overwrite config. Suggested config written to: {suggested_config}")
        print("Use --write-config to update config/default_config.json automatically.")

    print("Automatic HSV threshold result:")
    print(f"  hue_low={hue_low}")
    print(f"  hue_high={hue_high}")
    print(f"  saturation_low={saturation_low}")
    print(f"  value_low={value_low}")
    print(f"  candidate_pixels={total_candidates}")
    print(f"Wrote candidate summary: {summary_path}")
    print(f"Wrote candidate previews to: {candidate_dir}")

    generate_previews(cfg, metadata_path, image_dir, final_preview_dir, max_images=max_images, preview_max_dim=preview_max_dim)
    print(f"Wrote final threshold previews to: {final_preview_dir}")

def generate_previews(cfg: dict, metadata_path: Path, image_dir: Path, output_dir: Path, max_images: int, preview_max_dim: int) -> None:
    if not metadata_path.exists():
        raise SystemExit(f"Metadata file not found: {metadata_path}")
    metadata = pd.read_csv(metadata_path)
    lookup = image_lookup(image_dir, cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for _, row in metadata.iterrows():
        if count >= max_images:
            break
        use = str(row.get("use_image", "TRUE")).strip().lower() not in {"false", "0", "no", "n"}
        if not use:
            continue
        image_path = resolve_image_path(row, image_dir, cfg, lookup)
        rgb_full = read_image_rgb(image_path)
        rgb, scale = resize_to_max_dim(rgb_full, preview_max_dim)
        analysis_mask = build_analysis_mask_for_preview(rgb_full, rgb, cfg, image_path.stem)
        collagen = collagen_mask_hsv(rgb, cfg) & analysis_mask
        preview = make_preview(rgb, collagen, analysis_mask)
        max_w = 3000
        if preview.shape[1] > max_w:
            shrink = max_w / preview.shape[1]
            preview = cv2.resize(preview, (max_w, int(preview.shape[0] * shrink)), interpolation=cv2.INTER_AREA)
        write_rgb(output_dir / f"{image_path.stem}_threshold_preview.png", preview)
        print(f"Wrote preview for {image_path.name} using scale {scale:.3f}")
        count += 1
    print(f"Wrote {count} threshold previews to {output_dir}")
    print("Preview layout: original | collagen overlay | collagen mask")


def interactive_tune(cfg: dict, image_path: Path, config_path: Path, preview_max_dim: int, max_display: int) -> None:
    rgb_full = read_image_rgb(image_path)
    rgb, scale = resize_to_max_dim(rgb_full, preview_max_dim)
    analysis_mask = build_analysis_mask_for_preview(rgb_full, rgb, cfg, image_path.stem)

    if scale < 1.0:
        print(f"Interactive thresholding is using a downsampled preview: scale={scale:.3f}, size={rgb.shape[1]} x {rgb.shape[0]}")
        print("The written HSV settings are still applied to full-resolution images during final quantification.")
    else:
        print(f"Interactive thresholding is using full image size: {rgb.shape[1]} x {rgb.shape[0]}")

    if not cfg.get("collagen_hsv_ranges"):
        cfg["collagen_hsv_ranges"] = [{"name": "blue_green_collagen"}]
    r = cfg["collagen_hsv_ranges"][0]

    viewer = ZoomPanViewer(max_display=max_display)
    cv2.namedWindow(viewer.window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(viewer.window_name, viewer.mouse_callback)

    def noop(_):
        pass

    cv2.createTrackbar("hue_low", viewer.window_name, int(r.get("hue_low", 55)), 179, noop)
    cv2.createTrackbar("hue_high", viewer.window_name, int(r.get("hue_high", 135)), 179, noop)
    cv2.createTrackbar("sat_low", viewer.window_name, int(r.get("saturation_low", 30)), 255, noop)
    cv2.createTrackbar("value_low", viewer.window_name, int(r.get("value_low", 20)), 255, noop)
    cv2.createTrackbar("min_area", viewer.window_name, int(cfg.get("collagen_cleanup", {}).get("min_object_area_px", 40)), 1000, noop)

    print(HELP_TEXT)
    last_values = None
    dirty = True
    while True:
        values = (
            cv2.getTrackbarPos("hue_low", viewer.window_name),
            cv2.getTrackbarPos("hue_high", viewer.window_name),
            cv2.getTrackbarPos("sat_low", viewer.window_name),
            cv2.getTrackbarPos("value_low", viewer.window_name),
            cv2.getTrackbarPos("min_area", viewer.window_name),
        )
        if values != last_values:
            r["hue_low"], r["hue_high"], r["saturation_low"], r["value_low"], min_area = values
            cfg.setdefault("collagen_cleanup", {})["min_object_area_px"] = min_area
            collagen = collagen_mask_hsv(rgb, cfg) & analysis_mask
            preview = make_preview(rgb, collagen, analysis_mask)
            viewer.set_image(preview)
            cv2.resizeWindow(viewer.window_name, viewer.canvas_w, viewer.canvas_h)
            last_values = values
            dirty = False

        cv2.imshow(viewer.window_name, viewer.render())
        key = cv2.waitKey(30) & 0xFF
        if key == 255:
            continue
        if key in [ord('q'), 27]:
            break
        if key == ord('h'):
            print(HELP_TEXT)
        elif key == ord('w'):
            save_config(cfg, config_path)
            print(f"Saved updated HSV settings to {config_path}")
        elif key in [ord('+'), ord('=')]:
            viewer.zoom(1.25)
        elif key in [ord('-'), ord('_')]:
            viewer.zoom(1 / 1.25)
        elif key == ord('0'):
            viewer.reset_zoom()
        elif key in [ord('a'), 81]:
            viewer.pan_display_pixels(80, 0)
        elif key in [ord('d'), 83]:
            viewer.pan_display_pixels(-80, 0)
        elif key in [ord('w')]:
            # Already handled above as write-config; kept separate to make intent explicit.
            pass
        elif key in [ord('x'), 84]:
            viewer.pan_display_pixels(0, -80)
        elif key in [ord('s'), 82]:
            # s / up arrow pans up. Use w to write config.
            viewer.pan_display_pixels(0, 80)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune or preview Masson's Trichrome collagen colour thresholds.")
    parser.add_argument("--config", default="config/default_config.json")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--output-dir", default="results/qc/threshold_previews")
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--auto", action="store_true", help="Automatically estimate HSV collagen thresholds from representative images.")
    parser.add_argument("--write-config", action="store_true", help="With --auto, write suggested HSV thresholds into config/default_config.json.")
    parser.add_argument("--auto-output-dir", default="results/qc/auto_threshold", help="Output directory for automatic threshold reports/previews.")
    parser.add_argument("--interactive", action="store_true", help="Open an interactive HSV tuning window for a single image.")
    parser.add_argument("--image", default=None, help="Image filename/path for interactive tuning. Defaults to first metadata image.")
    parser.add_argument("--preview-max-dim", type=int, default=None, help="Maximum dimension of the interactive/preview image used for tuning. Lower is faster.")
    parser.add_argument("--max-display", type=int, default=1800, help="Maximum display dimension for the OpenCV window.")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_config(config_path)
    metadata_path = Path(args.metadata or cfg.get("metadata_csv", "metadata.csv"))
    image_dir = Path(args.image_dir or cfg.get("image_dir", "images"))
    preview_max_dim = int(args.preview_max_dim or cfg.get("interactive_preview_max_dim", 1400))

    if args.auto:
        auto_tune_thresholds(
            cfg=cfg,
            metadata_path=metadata_path,
            image_dir=image_dir,
            output_dir=Path(args.auto_output_dir),
            max_images=args.max_images,
            preview_max_dim=preview_max_dim,
            config_path=config_path,
            write_config=args.write_config,
        )
    elif args.interactive:
        if args.image:
            image_path = Path(args.image)
            if not image_path.exists():
                image_path = image_dir / args.image
        else:
            metadata = pd.read_csv(metadata_path)
            lookup = image_lookup(image_dir, cfg)
            image_path = resolve_image_path(metadata.iloc[0], image_dir, cfg, lookup)
        interactive_tune(cfg, image_path, config_path, preview_max_dim=preview_max_dim, max_display=args.max_display)
    else:
        generate_previews(cfg, metadata_path, image_dir, Path(args.output_dir), args.max_images, preview_max_dim=preview_max_dim)


if __name__ == "__main__":
    main()
