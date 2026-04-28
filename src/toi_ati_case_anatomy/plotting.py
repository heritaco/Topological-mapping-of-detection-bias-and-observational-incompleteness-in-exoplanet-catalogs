from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_top_regions(top_regions: pd.DataFrame, path: Path, *, value_col: str = "TOI") -> None:
    if top_regions.empty or value_col not in top_regions.columns:
        return
    df = top_regions.sort_values(value_col, ascending=True)
    labels = df.get("node_id", pd.Series(range(len(df)))).astype(str)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.28 * len(df))))
    ax.barh(labels, df[value_col])
    ax.set_xlabel(value_col)
    ax.set_ylabel("Node ID")
    ax.set_title(f"Top regiones por {value_col}")
    _save(fig, path)


def plot_top_anchors(top_anchors: pd.DataFrame, path: Path, *, value_col: str = "ATI") -> None:
    if top_anchors.empty or value_col not in top_anchors.columns:
        return
    df = top_anchors.sort_values(value_col, ascending=True)
    labels = (
        df.get("anchor_pl_name", pd.Series(range(len(df)))).astype(str)
        + " / "
        + df.get("node_id", pd.Series([""] * len(df))).astype(str)
    )
    fig, ax = plt.subplots(figsize=(9, max(4, 0.30 * len(df))))
    ax.barh(labels, df[value_col])
    ax.set_xlabel(value_col)
    ax.set_ylabel("Anchor / node")
    ax.set_title(f"Top planetas ancla por {value_col}")
    _save(fig, path)


def plot_toi_decomposition(top_regions: pd.DataFrame, path: Path) -> None:
    cols = ["shadow_score", "one_minus_I_R3", "C_phys", "S_net"]
    available = [c for c in cols if c in top_regions.columns]
    if top_regions.empty or not available or "node_id" not in top_regions.columns:
        return
    df = top_regions.set_index("node_id")[available]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    bottom = None
    # Stacked bars are for visual decomposition only; TOI is multiplicative, not additive.
    for col in available:
        values = df[col]
        ax.barh(df.index.astype(str), values, left=bottom, label=col)
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("factor value; stacked only for visual audit")
    ax.set_title("Descomposición visual de factores TOI")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_ati_decomposition(top_anchors: pd.DataFrame, path: Path) -> None:
    cols = ["TOI", "positive_delta_rel_neighbors_best", "one_minus_anchor_I_R3", "anchor_representativeness"]
    available = [c for c in cols if c in top_anchors.columns]
    if top_anchors.empty or not available:
        return
    label = top_anchors.get("anchor_pl_name", pd.Series(range(len(top_anchors)))).astype(str) + " / " + top_anchors.get("node_id", pd.Series([""]*len(top_anchors))).astype(str)
    df = top_anchors.copy()
    df["label"] = label
    df = df.set_index("label")[available]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(df))))
    bottom = None
    for col in available:
        values = df[col]
        ax.barh(df.index.astype(str), values, left=bottom, label=col)
        bottom = values if bottom is None else bottom + values
    ax.set_xlabel("factor value; stacked only for visual audit")
    ax.set_title("Descomposición visual de factores ATI")
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_deficit_by_radius(deficit_summary: pd.DataFrame, path: Path) -> None:
    if deficit_summary.empty:
        return
    radius_cols = [c for c in deficit_summary.columns if c not in {"node_id", "anchor_pl_name", "delta_rel_neighbors_mean", "delta_rel_neighbors_median", "delta_rel_neighbors_best", "best_radius"}]
    radius_cols = [c for c in radius_cols if pd.api.types.is_numeric_dtype(deficit_summary[c])]
    if not radius_cols:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([deficit_summary[c].dropna() for c in radius_cols], labels=radius_cols)
    ax.axhline(0, linewidth=1)
    ax.set_ylabel("delta_rel_neighbors")
    ax.set_title("Déficit relativo por radio")
    _save(fig, path)
