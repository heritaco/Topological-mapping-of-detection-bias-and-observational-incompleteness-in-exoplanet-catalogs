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
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.02, 0.6, message, transform=ax.transAxes)
    _save_fig(fig, path)


def method_color_map(methods: list[str]) -> dict[str, Any]:
    cmap = plt.get_cmap("tab20")
    return {method: cmap(idx % 20) for idx, method in enumerate(methods)}


def graph_positions(node_metrics: pd.DataFrame, edge_table: pd.DataFrame, seed: int) -> dict[str, tuple[float, float]]:
    if {"lens_1_mean", "lens_2_mean"}.issubset(node_metrics.columns):
        coords = node_metrics.set_index("node_id")[["lens_1_mean", "lens_2_mean"]].apply(pd.to_numeric, errors="coerce")
        if coords.notna().all(axis=1).all() and coords.nunique().min() > 1:
            return {node_id: (float(row["lens_1_mean"]), float(row["lens_2_mean"])) for node_id, row in coords.reset_index().set_index("node_id").iterrows()}
    graph = nx.Graph()
    graph.add_nodes_from(node_metrics["node_id"].astype(str).tolist())
    for row in edge_table.itertuples(index=False):
        graph.add_edge(str(getattr(row, "source")), str(getattr(row, "target")))
    if graph.number_of_nodes() == 0:
        return {}
    return nx.spring_layout(graph, seed=seed)


def _draw_base_graph(ax: Any, positions: dict[str, tuple[float, float]], edge_table: pd.DataFrame) -> None:
    for row in edge_table.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        if source not in positions or target not in positions:
            continue
        xs = [positions[source][0], positions[target][0]]
        ys = [positions[source][1], positions[target][1]]
        ax.plot(xs, ys, color="0.8", linewidth=0.8, zorder=1)
    ax.set_axis_off()


def _node_sizes(node_metrics: pd.DataFrame) -> np.ndarray:
    values = pd.to_numeric(node_metrics["n_members"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    return 35.0 + 18.0 * np.sqrt(np.maximum(values, 1.0))


def plot_graph_by_dominant_method(
    node_metrics: pd.DataFrame,
    edge_table: pd.DataFrame,
    path: Path,
    seed: int,
) -> None:
    if node_metrics.empty:
        placeholder_figure(path, "Dominant discovery method", "No hay nodos disponibles para esta configuracion.")
        return
    positions = graph_positions(node_metrics, edge_table, seed=seed)
    methods = sorted(node_metrics["top_method"].astype(str).dropna().unique().tolist())
    color_lookup = method_color_map(methods)
    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_base_graph(ax, positions, edge_table)
    sizes = _node_sizes(node_metrics)
    for method in methods:
        subset = node_metrics[node_metrics["top_method"].astype(str) == method]
        coords = np.array([positions[node_id] for node_id in subset["node_id"].astype(str) if node_id in positions], dtype=float)
        if coords.size == 0:
            continue
        subset_sizes = sizes[subset.index.to_numpy()]
        ax.scatter(coords[:, 0], coords[:, 1], s=subset_sizes, color=color_lookup[method], label=method, edgecolors="black", linewidths=0.3, zorder=2)
    ax.set_title("Mapper orbital coloreado por metodo de descubrimiento dominante")
    ax.legend(title="Metodo dominante", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _save_fig(fig, path)


def plot_graph_by_continuous_metric(
    node_metrics: pd.DataFrame,
    edge_table: pd.DataFrame,
    value_column: str,
    title: str,
    path: Path,
    seed: int,
    cmap: str = "viridis",
) -> None:
    if node_metrics.empty:
        placeholder_figure(path, title, "No hay nodos disponibles para esta configuracion.")
        return
    positions = graph_positions(node_metrics, edge_table, seed=seed)
    fig, ax = plt.subplots(figsize=(12, 9))
    _draw_base_graph(ax, positions, edge_table)
    subset = node_metrics[node_metrics["node_id"].astype(str).isin(positions.keys())].copy()
    coords = np.array([positions[node_id] for node_id in subset["node_id"].astype(str)], dtype=float)
    values = pd.to_numeric(subset[value_column], errors="coerce").to_numpy(dtype=float)
    sc = ax.scatter(coords[:, 0], coords[:, 1], s=_node_sizes(subset), c=values, cmap=cmap, edgecolors="black", linewidths=0.3, zorder=2)
    ax.set_title(title)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label(value_column)
    _save_fig(fig, path)


def plot_node_method_heatmap(
    fraction_matrix: pd.DataFrame,
    node_metrics: pd.DataFrame,
    path: Path,
    top_n_nodes: int,
) -> str:
    if fraction_matrix.empty:
        placeholder_figure(path, "Node x method heatmap", "No hay matriz nodo-metodo disponible.")
        return "empty"
    methods = [column for column in fraction_matrix.columns if column not in {"config_id", "node_id"}]
    merged = fraction_matrix.merge(node_metrics[["node_id", "n_members"]], on="node_id", how="left")
    selected = merged.sort_values("n_members", ascending=False).head(top_n_nodes).copy()
    matrix = selected[methods].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(8, len(methods) * 1.2), max(6, len(selected) * 0.22)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(f"Top {len(selected)} nodos por tamano: composicion por metodo")
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(selected)))
    ax.set_yticklabels(selected["node_id"].astype(str).tolist())
    fig.colorbar(im, ax=ax, shrink=0.8, label="Fraccion por nodo")
    _save_fig(fig, path)
    return f"top_{len(selected)}_nodes_by_n_members"


def plot_enrichment_heatmap(
    enrichment_df: pd.DataFrame,
    node_metrics: pd.DataFrame,
    path: Path,
    top_n_nodes: int,
    top_n_methods: int,
) -> None:
    if enrichment_df.empty:
        placeholder_figure(path, "Node-method enrichment", "No hay resultados de enriquecimiento disponibles.")
        return
    ranked_nodes = (
        enrichment_df.sort_values(["fdr_q_value", "z_score"], ascending=[True, False])
        .drop_duplicates(subset=["node_id"])
        .head(top_n_nodes)["node_id"]
        .astype(str)
        .tolist()
    )
    subset = enrichment_df[enrichment_df["node_id"].astype(str).isin(ranked_nodes)].copy()
    if subset.empty:
        placeholder_figure(path, "Node-method enrichment", "No hay nodos enriquecidos para visualizar.")
        return
    ranked_methods = (
        subset.groupby("method")["z_score"].max().sort_values(ascending=False).head(top_n_methods).index.astype(str).tolist()
    )
    pivot = (
        subset[subset["method"].astype(str).isin(ranked_methods)]
        .pivot(index="node_id", columns="method", values="z_score")
        .reindex(index=ranked_nodes, columns=ranked_methods)
    )
    fig, ax = plt.subplots(figsize=(max(8, len(ranked_methods) * 1.25), max(6, len(ranked_nodes) * 0.22)))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="coolwarm")
    ax.set_title("Z-scores de sobrerrepresentacion por nodo y metodo")
    ax.set_xticks(np.arange(len(ranked_methods)))
    ax.set_xticklabels(ranked_methods, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(ranked_nodes)))
    ax.set_yticklabels(ranked_nodes)
    fig.colorbar(im, ax=ax, shrink=0.8, label="z-score")
    _save_fig(fig, path)


def plot_null_histogram(
    null_distribution: pd.DataFrame,
    metric: str,
    observed_value: float,
    title: str,
    path: Path,
) -> None:
    if null_distribution.empty or metric not in null_distribution.columns:
        placeholder_figure(path, title, "No hay distribucion nula almacenada para esta metrica.")
        return
    values = pd.to_numeric(null_distribution[metric], errors="coerce").dropna().to_numpy(dtype=float)
    if values.size == 0 or not np.isfinite(observed_value):
        placeholder_figure(path, title, "La metrica observada o la distribucion nula no es valida.")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=30, color="0.7", edgecolor="black")
    ax.axvline(observed_value, color="tab:red", linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Frecuencia")
    _save_fig(fig, path)


def plot_scatter(
    node_metrics: pd.DataFrame,
    x_column: str,
    y_column: str,
    title: str,
    path: Path,
) -> None:
    if node_metrics.empty:
        placeholder_figure(path, title, "No hay nodos disponibles para esta figura.")
        return
    x = pd.to_numeric(node_metrics[x_column], errors="coerce")
    y = pd.to_numeric(node_metrics[y_column], errors="coerce")
    size = np.maximum(pd.to_numeric(node_metrics["n_members"], errors="coerce").fillna(1.0).to_numpy(dtype=float), 1.0)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(x, y, s=18.0 + 4.0 * np.sqrt(size), alpha=0.8, edgecolors="black", linewidths=0.3)
    ax.set_title(title)
    ax.set_xlabel(x_column)
    ax.set_ylabel(y_column)
    _save_fig(fig, path)


def plot_component_method_composition(
    component_composition: pd.DataFrame,
    path: Path,
) -> None:
    if component_composition.empty:
        placeholder_figure(path, "Component composition", "No hay composicion por componente disponible.")
        return
    pivot = component_composition.pivot(index="component_id", columns="method", values="fraction").fillna(0.0)
    methods = pivot.columns.astype(str).tolist()
    colors = method_color_map(methods)
    fig, ax = plt.subplots(figsize=(10, max(5, len(pivot) * 0.35)))
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for method in methods:
        values = pivot[method].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=method, color=colors[method], edgecolor="white", linewidth=0.3)
        bottom += values
    ax.set_title("Composicion por metodo de descubrimiento en cada componente")
    ax.set_ylabel("Fraccion")
    ax.set_xticks(x)
    ax.set_xticklabels([str(value) for value in pivot.index.tolist()], rotation=45, ha="right")
    ax.legend(title="Metodo", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    _save_fig(fig, path)


def plot_config_comparison(summary_global: pd.DataFrame, path: Path) -> None:
    if summary_global.empty:
        placeholder_figure(path, "Configuration comparison", "No hay configuraciones disponibles para comparar.")
        return
    metrics = [
        ("weighted_mean_purity", "Pureza media"),
        ("weighted_mean_entropy", "Entropia media"),
        ("node_method_nmi", "NMI observada"),
        ("global_nmi_z_score", "z-score NMI"),
        ("mean_node_imputation_fraction", "Imputacion media"),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    axes_flat = axes.flatten()
    labels = summary_global["config_id"].astype(str).tolist()
    x = np.arange(len(labels))
    for idx, (column, label) in enumerate(metrics):
        ax = axes_flat[idx]
        values = pd.to_numeric(summary_global[column], errors="coerce").to_numpy(dtype=float)
        ax.bar(x, values, color="0.6", edgecolor="black", linewidth=0.4)
        ax.set_title(label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right")
    axes_flat[-1].axis("off")
    fig.suptitle("Comparacion de metricas de sesgo observacional por configuracion", y=0.98)
    _save_fig(fig, path)
