#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


PLOT_METRICS = {
    "total_fibrosis_percent_mean": "Total fibrosis (%)",
    "interstitial_fibrosis_percent_mean": "Interstitial fibrosis (%)",
    "perivascular_collagen_percent_mean": "Perivascular collagen (%)",
    "perivascular_ring_collagen_percent_mean": "Perivascular ring collagen (%)"
}


def plot_metric(df: pd.DataFrame, metric_col: str, y_label: str, out_path: Path) -> None:
    plot_df = df[["group", metric_col]].copy()
    plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric_col])
    if plot_df.empty:
        print(f"Skipping {metric_col}: no numeric values")
        return

    groups = list(plot_df["group"].fillna("Unknown").astype(str).unique())
    values = [plot_df.loc[plot_df["group"].fillna("Unknown").astype(str) == g, metric_col].values for g in groups]

    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(groups)), 5))
    
    try:
        ax.boxplot(values, tick_labels=groups, showmeans=True)
    except TypeError:
        ax.boxplot(values, labels=groups, showmeans=True)
    rng = np.random.default_rng(123)
    for i, vals in enumerate(values, start=1):
        jitter = rng.normal(0, 0.04, size=len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, zorder=3)
    ax.set_ylabel(y_label)
    ax.set_xlabel("Group")
    ax.set_title(y_label + " by mouse")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot mouse-level fibrosis summaries.")
    parser.add_argument("--input", default="results/mouse_level_compartment_fibrosis_summary.csv")
    parser.add_argument("--output-dir", default="results/plots")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}. Run 05_summarise_by_mouse.py first.")

    df = pd.read_csv(input_path)
    if "group" not in df.columns:
        df["group"] = "Unknown"
    for col, label in PLOT_METRICS.items():
        if col in df.columns:
            filename = col.replace("_mean", "") + "_by_group.png"
            plot_metric(df, col, label, output_dir / filename)


if __name__ == "__main__":
    main()
