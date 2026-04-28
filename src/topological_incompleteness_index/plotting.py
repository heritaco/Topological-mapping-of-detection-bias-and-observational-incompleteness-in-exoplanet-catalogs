from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..visual_style import apply_axis_style, configure_matplotlib


configure_matplotlib(plt.matplotlib)


def _save(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.03, 0.55, message, transform=ax.transAxes, va="center")
    _save(fig, path)


def plot_top_regions(regions: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    if regions.empty:
        placeholder_figure(path, "Top TOI regions", "No hay regiones disponibles.")
        return
    data = regions.sort_values("TOI", ascending=False).head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9.2, max(4.6, 0.34 * len(data))))
    ax.barh(data["node_id"].astype(str), pd.to_numeric(data["TOI"], errors="coerce"), color="0.45", edgecolor="black", linewidth=0.4)
    apply_axis_style(ax, "Top regiones por TOI", "TOI", "Node ID")
    _save(fig, path)


def plot_top_anchors(anchors: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    if anchors.empty:
        placeholder_figure(path, "Top ATI anchors", "No hay planetas ancla disponibles.")
        return
    data = anchors.sort_values("ATI", ascending=False).head(top_n).iloc[::-1]
    labels = data["anchor_pl_name"].astype(str) + " / " + data["node_id"].astype(str)
    fig, ax = plt.subplots(figsize=(9.8, max(4.6, 0.34 * len(data))))
    ax.barh(labels, pd.to_numeric(data["ATI"], errors="coerce"), color="0.35", edgecolor="black", linewidth=0.4)
    apply_axis_style(ax, "Top planetas ancla por ATI", "ATI", "Anchor / node")
    _save(fig, path)


def plot_toi_vs_shadow(regions: pd.DataFrame, path: Path) -> None:
    if regions.empty:
        placeholder_figure(path, "TOI vs shadow score", "No hay regiones disponibles.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    sc = ax.scatter(
        pd.to_numeric(regions["shadow_score"], errors="coerce"),
        pd.to_numeric(regions["TOI"], errors="coerce"),
        c=pd.to_numeric(regions["I_R3"], errors="coerce"),
        s=42 + 8 * np.sqrt(pd.to_numeric(regions["n_members"], errors="coerce").fillna(1.0)),
        edgecolors="black",
        linewidths=0.4,
    )
    top = regions.sort_values("TOI", ascending=False).head(8)
    for _, row in top.iterrows():
        ax.annotate(str(row["node_id"]), (float(row["shadow_score"]), float(row["TOI"])), xytext=(5, 4), textcoords="offset points", fontsize=8)
    apply_axis_style(ax, "TOI vs shadow score", "shadow_score", "TOI")
    fig.colorbar(sc, ax=ax, shrink=0.82, label="I_R3")
    _save(fig, path)


def plot_toi_vs_physical_distance(regions: pd.DataFrame, path: Path) -> None:
    if regions.empty:
        placeholder_figure(path, "TOI vs physical distance", "No hay regiones disponibles.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    methods = sorted(regions["top_method"].astype(str).fillna("Unknown").unique().tolist()) if "top_method" in regions.columns else ["Unknown"]
    cmap = plt.get_cmap("tab10")
    for idx, method in enumerate(methods):
        subset = regions[regions["top_method"].astype(str) == method].copy() if "top_method" in regions.columns else regions.copy()
        ax.scatter(
            pd.to_numeric(subset["physical_distance_v_to_N1"], errors="coerce"),
            pd.to_numeric(subset["TOI"], errors="coerce"),
            s=42,
            color=cmap(idx % 10),
            edgecolors="black",
            linewidths=0.4,
            label=method,
        )
    apply_axis_style(ax, "TOI vs distancia fisica al vecindario", "physical_distance_v_to_N1", "TOI")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, path)


def plot_ati_vs_delta(anchors: pd.DataFrame, path: Path) -> None:
    if anchors.empty:
        placeholder_figure(path, "ATI vs delta_rel_neighbors_best", "No hay anclas disponibles.")
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    methods = sorted(anchors["discoverymethod"].astype(str).fillna("Unknown").unique().tolist())
    cmap = plt.get_cmap("tab10")
    for idx, method in enumerate(methods):
        subset = anchors[anchors["discoverymethod"].astype(str) == method].copy()
        ax.scatter(
            pd.to_numeric(subset["delta_rel_neighbors_best"], errors="coerce"),
            pd.to_numeric(subset["ATI"], errors="coerce"),
            s=48,
            color=cmap(idx % 10),
            edgecolors="black",
            linewidths=0.4,
            label=method,
        )
    apply_axis_style(ax, "ATI vs deficit relativo local", "delta_rel_neighbors_best", "ATI")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    _save(fig, path)


def plot_rank_matrix(regions: pd.DataFrame, anchors: pd.DataFrame, path: Path, top_n: int = 10) -> None:
    if regions.empty or anchors.empty:
        placeholder_figure(path, "TOI/ATI rank matrix", "No hay regiones y anclas suficientes.")
        return
    merged = anchors.merge(regions[["node_id", "I_R3", "physical_distance_v_to_N1"]], on="node_id", how="left")
    data = merged.sort_values(["ATI", "TOI"], ascending=False).head(top_n).copy()
    columns = ["TOI", "ATI", "I_R3", "delta_rel_neighbors_best", "physical_distance_v_to_N1"]
    matrix = data[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.4, max(4.4, 0.42 * len(data))))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels((data["anchor_pl_name"].astype(str) + " / " + data["node_id"].astype(str)).tolist())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Matriz visual TOI/ATI")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save(fig, path)


def plot_deficit_distribution_by_radius(deficits: pd.DataFrame, path: Path) -> None:
    if deficits.empty:
        placeholder_figure(path, "Deficit by radius", "No hay deficits por radio.")
        return
    labels = ["r_node_median", "r_node_q75", "r_kNN"]
    values = [pd.to_numeric(deficits.loc[deficits["radius_type"] == label, "delta_rel_neighbors"], errors="coerce").dropna().to_numpy(dtype=float) for label in labels]
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.boxplot(values, labels=labels, patch_artist=True, boxprops={"facecolor": "0.75", "edgecolor": "black"}, medianprops={"color": "black"})
    apply_axis_style(ax, "Distribucion de delta_rel_neighbors por radio", "radius_type", "delta_rel_neighbors")
    _save(fig, path)


def plot_top_anchor_rv_proxy(anchors: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    if anchors.empty or "rv_proxy" not in anchors.columns:
        placeholder_figure(path, "Top anchor RV proxy", "No hay proxy RV disponible.")
        return
    top = pd.to_numeric(anchors.sort_values("ATI", ascending=False).head(top_n)["rv_proxy"], errors="coerce").dropna()
    all_values = pd.to_numeric(anchors["rv_proxy"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.hist(all_values, bins=18, alpha=0.55, label="all anchors", color="0.70", edgecolor="black", linewidth=0.4)
    ax.hist(top, bins=12, alpha=0.75, label="top ATI anchors", color="0.35", edgecolor="black", linewidth=0.4)
    apply_axis_style(ax, "Proxy RV en anclas top vs total", "rv_proxy", "Count")
    ax.legend(frameon=False)
    _save(fig, path)


def plot_method_distribution_high_toi(regions: pd.DataFrame, path: Path) -> None:
    subset = regions[regions["region_class"].astype(str) == "high_toi_region"].copy() if not regions.empty else pd.DataFrame()
    if subset.empty:
        placeholder_figure(path, "Method distribution high TOI", "No hay regiones high_toi_region.")
        return
    counts = subset["top_method"].astype(str).value_counts().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ax.bar(counts.index.tolist(), counts.to_numpy(dtype=float), color="0.5", edgecolor="black", linewidth=0.4)
    ax.tick_params(axis="x", rotation=30)
    apply_axis_style(ax, "Metodo dominante entre regiones high TOI", "Dominant method", "Count")
    _save(fig, path)
