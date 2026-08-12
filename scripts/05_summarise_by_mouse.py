#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


METRICS = [
    "total_fibrosis_percent",
    "interstitial_fibrosis_percent",
    "perivascular_collagen_percent",
    "perivascular_ring_collagen_percent",
    "analysis_area_px",
    "total_collagen_area_px",
    "interstitial_collagen_area_px",
    "perivascular_collagen_area_px",
]


def summarise(df: pd.DataFrame, group_cols: list[str], count_name: str) -> pd.DataFrame:
    existing_groups = [c for c in group_cols if c in df.columns]
    existing_metrics = [m for m in METRICS if m in df.columns]
    if not existing_groups:
        df = df.copy()
        df["all"] = "all"
        existing_groups = ["all"]
    count_col = "image" if "image" in df.columns else ("n_images" if "n_images" in df.columns else existing_metrics[0])
    summary = df.groupby(existing_groups, dropna=False).agg(
        **{count_name: (count_col, "count")},
        **{f"{m}_mean": (m, "mean") for m in existing_metrics},
        **{f"{m}_median": (m, "median") for m in existing_metrics},
        **{f"{m}_sd": (m, "std") for m in existing_metrics},
    ).reset_index()
    return summary


def group_level_from_mouse(mouse_df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in mouse_df.columns if c.endswith("_percent_mean")]
    rows = []
    group_cols = [c for c in ["group", "region"] if c in mouse_df.columns]
    if not group_cols:
        group_cols = ["group"] if "group" in mouse_df.columns else []
    for keys, sub in mouse_df.groupby(group_cols, dropna=False) if group_cols else [((), mouse_df)]:
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["n_mice"] = sub["mouse_id"].nunique() if "mouse_id" in sub.columns else len(sub)
        for col in metric_cols:
            vals = pd.to_numeric(sub[col], errors="coerce").dropna()
            base[f"{col}_group_mean"] = vals.mean() if len(vals) else np.nan
            base[f"{col}_group_sd"] = vals.std(ddof=1) if len(vals) > 1 else np.nan
            base[f"{col}_group_sem"] = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else np.nan
        rows.append(base)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise fibrosis results by section, mouse, and group.")
    parser.add_argument("--input", default="results/image_level_compartment_fibrosis_results.csv")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    input_path = Path(args.input)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise SystemExit(f"Input results not found: {input_path}. Run 04_quantify_masson_fibrosis.py first.")

    df = pd.read_csv(input_path)
    if "mouse_id" not in df.columns or df["mouse_id"].isna().all():
        print("WARNING: mouse_id is missing/empty. Mouse-level biological replicate summaries will be less useful.")

    section_cols = ["mouse_id", "group", "region", "section", "stain_batch"]
    mouse_cols = ["mouse_id", "group", "region", "stain_batch"]

    section = summarise(df, section_cols, "n_images")
    mouse_input = section.rename(columns={f"{m}_mean": m for m in METRICS if f"{m}_mean" in section.columns})
    mouse = summarise(mouse_input, mouse_cols, "n_sections")
    group = group_level_from_mouse(mouse)

    section_path = results_dir / "section_level_compartment_fibrosis_summary.csv"
    mouse_path = results_dir / "mouse_level_compartment_fibrosis_summary.csv"
    group_path = results_dir / "group_level_compartment_fibrosis_summary.csv"
    section.to_csv(section_path, index=False)
    mouse.to_csv(mouse_path, index=False)
    group.to_csv(group_path, index=False)

    print(f"Wrote {section_path}")
    print(f"Wrote {mouse_path}")
    print(f"Wrote {group_path}")
    print("Use the mouse-level file for statistical comparisons unless your experimental design says otherwise.")


if __name__ == "__main__":
    main()
