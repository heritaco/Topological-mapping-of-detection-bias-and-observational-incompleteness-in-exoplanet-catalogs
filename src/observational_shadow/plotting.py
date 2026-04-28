from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


def _save_fig(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def placeholder_figure(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.03, 0.55, message, transform=ax.transAxes, va="center")
    _save_fig(fig, path)


def graph_positions(node_metrics: pd.DataFrame, edge_table: pd.DataFrame, seed: int) -> dict[str, tuple[float, float]]:
    if {"lens_1_mean", "lens_2_mean"}.issubset(node_metrics.columns):
        coords = node_metrics.set_index("node_id")[["lens_1_mean", "lens_2_mean"]].apply(pd.to_numeric, errors="coerce")
        if coords.notna().all(axis=1).all() and coords.nunique().min() > 1:
            return {str(idx): (float(row["lens_1_mean"]), float(row["lens_2_mean"])) for idx, row in coords.iterrows()}
    graph = nx.Graph()
    graph.add_nodes_from(node_metrics["node_id"].astype(str).tolist())
    if not edge_table.empty:
        for row in edge_table.itertuples(index=False):
            graph.add_edge(str(getattr(row, "source")), str(getattr(row, "target")))
    if graph.number_of_nodes() == 0:
        return {}
    return {str(key): tuple(value) for key, value in nx.spring_layout(graph, seed=seed).items()}


def _draw_edges(ax: Any, positions: dict[str, tuple[float, float]], edge_table: pd.DataFrame) -> None:
    if edge_table.empty:
        return
    for row in edge_table.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        if source in positions and target in positions:
            ax.plot(
                [positions[source][0], positions[target][0]],
                [positions[source][1], positions[target][1]],
                color="0.80",
                linewidth=0.7,
                zorder=1,
            )


def _node_sizes(frame: pd.DataFrame) -> np.ndarray:
    sizes = pd.to_numeric(frame["n_members"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return 35.0 + 18.0 * np.sqrt(np.maximum(sizes, 1.0))


def plot_graph_by_shadow_score(node_metrics: pd.DataFrame, edge_table: pd.DataFrame, path: Path, seed: int) -> None:
    if node_metrics.empty:
        placeholder_figure(path, "Grafo Mapper por shadow_score", "No hay nodos disponibles.")
        return
    positions = graph_positions(node_metrics, edge_table, seed)
    subset = node_metrics[node_metrics["node_id"].astype(str).isin(positions)].copy()
    if subset.empty:
        placeholder_figure(path, "Grafo Mapper por shadow_score", "No hay posiciones de nodos disponibles.")
        return
    coords = np.array([positions[node_id] for node_id in subset["node_id"].astype(str)], dtype=float)
    values = pd.to_numeric(subset["shadow_score"], errors="coerce").to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _draw_edges(ax, positions, edge_table)
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=_node_sizes(subset), c=values, cmap="viridis", edgecolors="black", linewidths=0.3, zorder=2)
    ax.set_title("Mapper orbital coloreado por indice de sombra observacional")
    ax.set_axis_off()
    cbar = fig.colorbar(sc, ax=ax, shrink=0.78)
    cbar.set_label("shadow_score")
    _save_fig(fig, path)


def plot_graph_by_shadow_class(node_metrics: pd.DataFrame, edge_table: pd.DataFrame, path: Path, seed: int) -> None:
    if node_metrics.empty:
        placeholder_figure(path, "Grafo Mapper por clase de sombra", "No hay nodos disponibles.")
        return
    positions = graph_positions(node_metrics, edge_table, seed)
    colors = {
        "high_confidence_shadow": "tab:red",
        "small_sample_shadow": "tab:orange",
        "imputation_sensitive_shadow": "tab:purple",
        "mixed_or_low_shadow": "0.62",
        "no_neighbor_information": "tab:blue",
    }
    fig, ax = plt.subplots(figsize=(11, 8.5))
    _draw_edges(ax, positions, edge_table)
    for klass, color in colors.items():
        subset = node_metrics[node_metrics["shadow_class"].astype(str) == klass].copy()
        subset = subset[subset["node_id"].astype(str).isin(positions)]
        if subset.empty:
            continue
        coords = np.array([positions[node_id] for node_id in subset["node_id"].astype(str)], dtype=float)
        ax.scatter(coords[:, 0], coords[:, 1], s=_node_sizes(subset), color=color, label=klass, edgecolors="black", linewidths=0.3, zorder=2)
    ax.set_title("Mapper orbital por clase heuristica de sombra observacional")
    ax.set_axis_off()
    ax.legend(title="Clase", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _save_fig(fig, path)


def plot_scatter(node_metrics: pd.DataFrame, x_column: str, y_column: str, title: str, xlabel: str, ylabel: str, path: Path, color_by: str | None = None) -> None:
    if node_metrics.empty or x_column not in node_metrics.columns or y_column not in node_metrics.columns:
        placeholder_figure(path, title, "No hay datos suficientes para esta figura.")
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    x = pd.to_numeric(node_metrics[x_column], errors="coerce")
    y = pd.to_numeric(node_metrics[y_column], errors="coerce")
    sizes = 18.0 + 4.0 * np.sqrt(pd.to_numeric(node_metrics["n_members"], errors="coerce").fillna(1.0))
    if color_by and color_by in node_metrics.columns:
        categories = sorted(node_metrics[color_by].astype(str).unique().tolist())
        cmap = plt.get_cmap("tab20")
        for idx, category in enumerate(categories):
            mask = node_metrics[color_by].astype(str) == category
            ax.scatter(x[mask], y[mask], s=sizes[mask], alpha=0.8, edgecolors="black", linewidths=0.3, label=category, color=cmap(idx % 20))
        ax.legend(title=color_by, loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    else:
        ax.scatter(x, y, s=sizes, alpha=0.8, edgecolors="black", linewidths=0.3, color="tab:blue")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_fig(fig, path)


def plot_top_candidates(candidates: pd.DataFrame, path: Path, top_n: int = 15) -> None:
    if candidates.empty:
        placeholder_figure(path, "Principales comunidades candidatas", "No hay candidatos disponibles.")
        return
    data = candidates.head(top_n).iloc[::-1].copy()
    labels = data.apply(lambda row: f"{row['node_id']} ({row['top_method']}, n={int(row['n_members'])})", axis=1)
    fig, ax = plt.subplots(figsize=(9.5, max(4.8, 0.35 * len(data))))
    ax.barh(labels, pd.to_numeric(data["shadow_score"], errors="coerce"), color="0.45", edgecolor="black", linewidth=0.4)
    ax.set_title("Top comunidades por sombra observacional")
    ax.set_xlabel("shadow_score")
    _save_fig(fig, path)


def plot_shadow_by_method(node_metrics: pd.DataFrame, path: Path) -> None:
    if node_metrics.empty:
        placeholder_figure(path, "Sombra observacional por metodo dominante", "No hay datos disponibles.")
        return
    methods = sorted(node_metrics["top_method"].astype(str).unique().tolist())
    values = [pd.to_numeric(node_metrics.loc[node_metrics["top_method"].astype(str) == method, "shadow_score"], errors="coerce").dropna().to_numpy() for method in methods]
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(methods)), 5.4))
    ax.boxplot(values, labels=methods, patch_artist=True, boxprops={"facecolor": "0.75", "edgecolor": "black"}, medianprops={"color": "black"})
    ax.set_title("Distribucion de shadow_score por metodo dominante")
    ax.set_xlabel("Metodo dominante")
    ax.set_ylabel("shadow_score")
    ax.tick_params(axis="x", rotation=45)
    _save_fig(fig, path)


def plot_component_summary(component_summary: pd.DataFrame, path: Path, primary_config_id: str) -> None:
    data = component_summary[component_summary["config_id"].astype(str) == primary_config_id].copy()
    if data.empty:
        placeholder_figure(path, "Resumen de sombra por componente", "No hay componentes disponibles.")
        return
    data = data.sort_values("max_shadow_score", ascending=False).head(25).iloc[::-1]
    labels = data["component_id"].astype(str)
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, 0.30 * len(data))))
    ax.barh(labels, pd.to_numeric(data["max_shadow_score"], errors="coerce"), color="0.55", edgecolor="black", linewidth=0.4)
    ax.set_title("Maximo shadow_score por componente orbital")
    ax.set_xlabel("max shadow_score")
    ax.set_ylabel("Componente")
    _save_fig(fig, path)


def plot_config_comparison(config_summary: pd.DataFrame, path: Path) -> None:
    if config_summary.empty:
        placeholder_figure(path, "Comparacion entre configuraciones", "No hay configuraciones disponibles.")
        return
    labels = config_summary["config_id"].astype(str).tolist()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    columns = [
        ("mean_shadow_score", "Media"),
        ("median_shadow_score", "Mediana"),
        ("max_shadow_score", "Maximo"),
        ("n_high_confidence_shadow", "Nodos alta confianza"),
    ]
    for ax, (column, title) in zip(axes.flatten(), columns):
        ax.bar(x, pd.to_numeric(config_summary[column], errors="coerce"), color="0.6", edgecolor="black", linewidth=0.4)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    fig.suptitle("Comparacion de sombra observacional entre configuraciones", y=0.98)
    _save_fig(fig, path)

