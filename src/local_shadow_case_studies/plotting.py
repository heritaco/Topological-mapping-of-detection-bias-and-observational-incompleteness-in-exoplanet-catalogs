from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


METHOD_COLORS = {
    "Radial Velocity": "tab:red",
    "Transit": "tab:blue",
    "Microlensing": "tab:green",
    "Imaging": "tab:orange",
    "Astrometry": "tab:brown",
    "Unknown": "0.55",
}
METHOD_MARKERS = {
    "Radial Velocity": "o",
    "Transit": "s",
    "Microlensing": "^",
    "Imaging": "D",
    "Astrometry": "P",
    "Unknown": "o",
}


def _save_fig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.6, message, transform=ax.transAxes, va="center")
    _save_fig(fig, path)


def plot_case_ego_network(path: Path, node_id: str, graph: nx.Graph, node_metrics: pd.DataFrame, n1_nodes: list[str], n2_nodes: list[str], shadow_score: float | None, seed: int) -> None:
    ego_nodes = {node_id, *n1_nodes, *n2_nodes}
    subgraph = graph.subgraph([node for node in ego_nodes if graph.has_node(node)]).copy()
    if subgraph.number_of_nodes() == 0:
        placeholder_figure(path, f"Ego network {node_id}", "No hay grafo local disponible.")
        return
    positions = nx.spring_layout(subgraph, seed=seed)
    meta = node_metrics.set_index(node_metrics["node_id"].astype(str))
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    nx.draw_networkx_edges(subgraph, pos=positions, ax=ax, edge_color="0.82", width=0.8)
    for current in subgraph.nodes():
        row = meta.loc[str(current)] if str(current) in meta.index else None
        method = str(row["top_method"]) if row is not None and "top_method" in row.index else "Unknown"
        n_members = float(row["n_members"]) if row is not None and "n_members" in row.index else 1.0
        size = 120 + 26 * np.sqrt(max(n_members, 1.0))
        linewidth = 2.0 if str(current) == node_id else 0.8
        edgecolor = "black" if str(current) == node_id else "0.35"
        nx.draw_networkx_nodes(
            subgraph,
            pos=positions,
            nodelist=[current],
            node_size=size,
            node_color=METHOD_COLORS.get(method, "0.55"),
            edgecolors=edgecolor,
            linewidths=linewidth,
            ax=ax,
        )
    nx.draw_networkx_labels(subgraph, pos=positions, ax=ax, font_size=8)
    label = f"shadow_score={shadow_score:.3f}" if shadow_score is not None and np.isfinite(shadow_score) else "shadow_score=NA"
    ax.set_title(f"Ego-network local de {node_id}\n{label}")
    ax.axis("off")
    _save_fig(fig, path)


def plot_case_r3_projections(path: Path, node_id: str, regions: pd.DataFrame, anchor_name: str | None) -> None:
    if regions.empty or not regions["r3_valid"].any():
        placeholder_figure(path, f"R3 projections {node_id}", "No hay suficientes miembros con R3 valido.")
        return
    projections = [
        ("r3_log_mass", "r3_log_period", "log10(pl_bmasse)", "log10(pl_orbper)"),
        ("r3_log_mass", "r3_log_semimajor", "log10(pl_bmasse)", "log10(pl_orbsmax)"),
        ("r3_log_period", "r3_log_semimajor", "log10(pl_orbper)", "log10(pl_orbsmax)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    region_styles = {"node": 0.95, "N1": 0.65, "N2": 0.35}
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, projections):
        for region_type, alpha in region_styles.items():
            subset = regions[(regions["region_type"] == region_type) & (regions["r3_valid"])].copy()
            if subset.empty:
                continue
            for method, group in subset.groupby(subset["discoverymethod"].astype("string").fillna("Unknown")):
                ax.scatter(
                    pd.to_numeric(group[xcol], errors="coerce"),
                    pd.to_numeric(group[ycol], errors="coerce"),
                    color=METHOD_COLORS.get(str(method), "0.55"),
                    marker=METHOD_MARKERS.get(str(method), "o"),
                    alpha=alpha,
                    s=42,
                    linewidths=0.4,
                    edgecolors="black",
                    label=f"{region_type} / {method}",
                )
        anchor = regions[(regions["is_anchor"]) & (regions["r3_valid"])].copy()
        if not anchor.empty:
            ax.scatter(anchor[xcol], anchor[ycol], marker="*", s=180, color="gold", edgecolors="black", linewidths=0.9, label=anchor_name or "anchor")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    handles, labels = axes[0].get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        unique[label] = handle
    fig.legend(unique.values(), unique.keys(), loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=False, fontsize=8)
    fig.suptitle(f"Proyecciones R3 del caso {node_id}", y=1.06)
    _save_fig(fig, path)


def plot_case_method_composition(path: Path, node_id: str, composition: pd.DataFrame) -> None:
    if composition.empty:
        placeholder_figure(path, f"Method composition {node_id}", "No hay composicion por metodo disponible.")
        return
    pivot = composition.pivot_table(index="method", columns="region_type", values="fraction", fill_value=0.0)
    regions = [region for region in ["node", "N1", "N2"] if region in pivot.columns]
    x = np.arange(len(pivot.index))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(7.5, 0.65 * len(pivot.index)), 5.2))
    for idx, region in enumerate(regions):
        ax.bar(x + (idx - (len(regions) - 1) / 2.0) * width, pivot[region].to_numpy(dtype=float), width=width, label=region, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=35, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title(f"Metodo del nodo vs vecindario: {node_id}")
    ax.legend(frameon=False)
    _save_fig(fig, path)


def plot_case_imputation_audit(path: Path, node_id: str, summary: dict[str, object]) -> None:
    variables = ["pl_bmasse", "pl_orbper", "pl_orbsmax"]
    observed = [float(summary.get(f"observed_fraction_{var}", 0.0) or 0.0) for var in variables]
    physical = [float(summary.get(f"physically_derived_fraction_{var}", 0.0) or 0.0) for var in variables]
    imputed = [float(summary.get(f"imputed_fraction_{var}", 0.0) or 0.0) for var in variables]
    x = np.arange(len(variables))
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.bar(x, observed, label="observed", color="0.45", edgecolor="black", linewidth=0.4)
    ax.bar(x, physical, bottom=observed, label="physically_derived", color="0.72", edgecolor="black", linewidth=0.4)
    ax.bar(x, imputed, bottom=np.array(observed) + np.array(physical), label="imputed", color="0.9", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.set_ylabel("Fraction")
    ax.set_title(f"Auditoria R3 de imputacion: {node_id}")
    ax.legend(frameon=False)
    _save_fig(fig, path)


def plot_case_anchor_deficit(path: Path, node_id: str, deficit: pd.DataFrame) -> None:
    if deficit.empty:
        placeholder_figure(path, f"Anchor deficit {node_id}", "No hay resultados de deficit local.")
        return
    data = deficit.copy()
    x = np.arange(len(data))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.bar(x - width, pd.to_numeric(data["N_obs"], errors="coerce").fillna(0.0), width=width, label="N_obs", edgecolor="black", linewidth=0.4)
    ax.bar(x, pd.to_numeric(data["N_exp_neighbors"], errors="coerce").fillna(0.0), width=width, label="N_exp_neighbors", edgecolor="black", linewidth=0.4)
    ax.bar(x + width, pd.to_numeric(data["N_exp_analog"], errors="coerce").fillna(0.0), width=width, label="N_exp_analog", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(data["radius_type"].astype(str).tolist())
    ax.set_ylabel("Count")
    ax.set_title(f"Deficit local de vecinos: {node_id}")
    ax.legend(frameon=False)
    _save_fig(fig, path)


def plot_case_rv_proxy_distribution(path: Path, node_id: str, regions: pd.DataFrame, anchor_name: str | None) -> None:
    if regions.empty or "rv_proxy" not in regions.columns:
        placeholder_figure(path, f"RV proxy {node_id}", "No hay proxy RV disponible.")
        return
    labels = ["node", "N1", "N2"]
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for idx, label in enumerate(labels):
        subset = pd.to_numeric(regions.loc[regions["region_type"] == label, "rv_proxy"], errors="coerce").dropna()
        if subset.empty:
            continue
        jitter = np.linspace(-0.08, 0.08, len(subset))
        ax.scatter(np.full(len(subset), idx) + jitter, subset, color="0.35", s=28, alpha=0.8)
    anchor = regions[regions["is_anchor"]].copy()
    if not anchor.empty and pd.notna(anchor["rv_proxy"]).any():
        ax.scatter([0], [pd.to_numeric(anchor["rv_proxy"], errors="coerce").iloc[0]], marker="*", s=200, color="gold", edgecolors="black", linewidths=0.8, label=anchor_name or "anchor")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("RV detectability proxy")
    ax.set_title(f"Distribucion de proxy RV: {node_id}")
    if not anchor.empty:
        ax.legend(frameon=False)
    _save_fig(fig, path)


def plot_three_case_shadow_vs_physical_distance(path: Path, comparison: pd.DataFrame) -> None:
    if comparison.empty:
        placeholder_figure(path, "Three case shadow vs distance", "No hay casos para comparar.")
        return
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    x = pd.to_numeric(comparison["physical_distance_v_to_N1"], errors="coerce")
    y = pd.to_numeric(comparison["shadow_score"], errors="coerce")
    ax.scatter(x, y, s=80, color="tab:red", edgecolors="black", linewidths=0.6)
    for _, row in comparison.iterrows():
        ax.annotate(str(row["node_id"]), (float(row["physical_distance_v_to_N1"]), float(row["shadow_score"])), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("physical_distance_v_to_N1")
    ax.set_ylabel("shadow_score")
    ax.set_title("Three-case shadow vs physical distance")
    _save_fig(fig, path)


def plot_three_case_deficit_comparison(path: Path, comparison: pd.DataFrame) -> None:
    if comparison.empty:
        placeholder_figure(path, "Three case deficit comparison", "No hay casos para comparar.")
        return
    x = np.arange(len(comparison))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ax.bar(x - width / 2.0, pd.to_numeric(comparison["delta_rel_neighbors_best"], errors="coerce").fillna(0.0), width=width, label="neighbors", edgecolor="black", linewidth=0.4)
    ax.bar(x + width / 2.0, pd.to_numeric(comparison["delta_rel_analog_best"], errors="coerce").fillna(0.0), width=width, label="analog", edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["node_id"].astype(str).tolist(), rotation=15)
    ax.set_ylabel("Relative deficit")
    ax.set_title("Relative local deficit by case")
    ax.legend(frameon=False)
    _save_fig(fig, path)


def plot_three_case_confidence_matrix(path: Path, comparison: pd.DataFrame) -> None:
    if comparison.empty:
        placeholder_figure(path, "Three case confidence matrix", "No hay casos para comparar.")
        return
    columns = ["shadow_score", "I_R3", "method_l1_boundary_N1", "physical_distance_v_to_N1", "delta_rel_neighbors_best"]
    matrix = comparison[columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(matrix, cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(columns)))
    ax.set_xticklabels(columns, rotation=25, ha="right")
    ax.set_yticks(range(len(comparison)))
    ax.set_yticklabels(comparison["node_id"].astype(str).tolist())
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title("Confidence matrix for three local cases")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_fig(fig, path)

