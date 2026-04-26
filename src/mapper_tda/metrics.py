from __future__ import annotations

import json
from itertools import combinations
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


GRAPH_DISTANCE_COLUMNS = [
    "n_nodes",
    "n_edges",
    "beta_0",
    "beta_1",
    "graph_density",
    "average_degree",
    "average_clustering",
    "mean_node_size",
    "mean_node_imputation_fraction",
    "mean_node_physically_derived_fraction",
]

GLOBAL_FEATURES = ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"]


def _python_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _json_list(values: list[Any]) -> str:
    return json.dumps([_python_scalar(value) for value in values], ensure_ascii=False)


def _label_entropy(labels: pd.Series) -> float:
    counts = labels.dropna().astype("string").value_counts(normalize=True)
    if counts.empty:
        return 0.0
    return float(-(counts * np.log2(counts)).sum())


def _dominant_label(labels: pd.Series) -> tuple[str | None, float | None]:
    counts = labels.dropna().astype("string").value_counts(normalize=True)
    if counts.empty:
        return None, None
    return str(counts.index[0]), float(counts.iloc[0])


def mapper_graph_to_networkx(graph: dict) -> nx.Graph:
    nx_graph = nx.Graph()
    nodes = graph.get("nodes", {})
    for node_id, members in nodes.items():
        nx_graph.add_node(node_id, size=len(members), sample_indices=list(members))
    for source, targets in graph.get("links", {}).items():
        for target in targets:
            if source != target:
                nx_graph.add_edge(source, target)
    return nx_graph


def compute_graph_metrics(nx_graph: nx.Graph, graph: dict) -> dict[str, Any]:
    node_sizes = [len(members) for members in graph.get("nodes", {}).values()]
    n_nodes = int(nx_graph.number_of_nodes())
    n_edges = int(nx_graph.number_of_edges())
    beta_0 = int(nx.number_connected_components(nx_graph)) if n_nodes else 0
    beta_1 = int(n_edges - n_nodes + beta_0)
    if beta_1 < 0:
        raise ValueError(f"beta_1 invalido: {beta_1} (E={n_edges}, V={n_nodes}, C={beta_0})")

    isolates = list(nx.isolates(nx_graph)) if n_nodes else []
    components = list(nx.connected_components(nx_graph)) if n_nodes else []
    largest_component = max(components, key=len) if components else set()

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "graph_density": float(nx.density(nx_graph)) if n_nodes else 0.0,
        "average_degree": float(np.mean([degree for _, degree in nx_graph.degree()])) if n_nodes else 0.0,
        "average_clustering": float(nx.average_clustering(nx_graph)) if n_nodes else 0.0,
        "transitivity": float(nx.transitivity(nx_graph)) if n_nodes else 0.0,
        "n_isolates": int(len(isolates)),
        "largest_component_size": int(len(largest_component)),
        "largest_component_fraction": float(len(largest_component) / n_nodes) if n_nodes else 0.0,
        "mean_node_size": float(np.mean(node_sizes)) if node_sizes else 0.0,
        "median_node_size": float(np.median(node_sizes)) if node_sizes else 0.0,
        "min_node_size": int(np.min(node_sizes)) if node_sizes else 0,
        "max_node_size": int(np.max(node_sizes)) if node_sizes else 0,
    }


def _sample_ids_for_members(graph: dict, members: list[int]) -> list[Any]:
    lookup = graph.get("sample_id_lookup")
    if lookup:
        return [_python_scalar(lookup[index]) for index in members]
    return [_python_scalar(index) for index in members]


def _fraction_from_flag(frame: pd.DataFrame, column: str) -> float | None:
    if column not in frame.columns:
        return None
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).mean())


def _source_fraction(frame: pd.DataFrame, feature: str, source_kind: str) -> float | None:
    col = f"{feature}_was_{source_kind}"
    return _fraction_from_flag(frame, col)


def _dominant_source(frame: pd.DataFrame, feature: str) -> str | None:
    source_col = f"{feature}_source"
    if source_col not in frame.columns:
        return None
    dominant, _ = _dominant_label(frame[source_col].astype("string"))
    return dominant


def _row_level_any_fraction(frame: pd.DataFrame, suffix: str) -> tuple[int, float]:
    cols = [column for column in frame.columns if column.endswith(suffix)]
    if not cols:
        return 0, 0.0
    flags = frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(bool)
    any_count = int(flags.any(axis=1).sum())
    return any_count, float(any_count / len(frame)) if len(frame) else 0.0


def _global_traceability(frame: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    observed_values: list[float] = []
    derived_values: list[float] = []
    imputed_values: list[float] = []

    for feature in features:
        observed = _source_fraction(frame, feature, "observed")
        derived = _source_fraction(frame, feature, "physically_derived")
        imputed = _source_fraction(frame, feature, "imputed")
        if observed is not None:
            observed_values.append(observed)
        if derived is not None:
            derived_values.append(derived)
        if imputed is not None:
            imputed_values.append(imputed)

    n_any_imputed, frac_any_imputed = _row_level_any_fraction(frame, "_was_imputed")
    n_any_derived, frac_any_derived = _row_level_any_fraction(frame, "_was_physically_derived")
    row_imputation = frame.get("imputation_fraction_row", pd.Series(dtype=float))

    return {
        "mean_imputation_fraction": float(pd.to_numeric(row_imputation, errors="coerce").mean())
        if not row_imputation.empty
        else float(np.mean(imputed_values))
        if imputed_values
        else 0.0,
        "median_imputation_fraction": float(pd.to_numeric(row_imputation, errors="coerce").median())
        if not row_imputation.empty
        else float(np.median(imputed_values))
        if imputed_values
        else 0.0,
        "observed_fraction": float(np.mean(observed_values)) if observed_values else 0.0,
        "physically_derived_fraction": float(np.mean(derived_values)) if derived_values else 0.0,
        "imputed_fraction": float(np.mean(imputed_values)) if imputed_values else 0.0,
        "n_any_imputed": n_any_imputed,
        "frac_any_imputed": frac_any_imputed,
        "n_any_physically_derived": n_any_derived,
        "frac_any_physically_derived": frac_any_derived,
    }


def _compute_row_imputation_fraction(frame: pd.DataFrame, features: list[str]) -> pd.Series:
    cols = [f"{feature}_was_imputed" for feature in features if f"{feature}_was_imputed" in frame.columns]
    if not cols:
        return pd.Series(np.zeros(len(frame)), index=frame.index, dtype=float)
    return frame.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0).mean(axis=1)


def build_node_table(
    graph: dict,
    nx_graph: nx.Graph,
    lens: np.ndarray,
    physical_df: pd.DataFrame,
    used_features: list[str],
    config_id: str,
) -> pd.DataFrame:
    frame = physical_df.copy()
    frame["imputation_fraction_row"] = _compute_row_imputation_fraction(frame, used_features)
    components = list(nx.connected_components(nx_graph))
    component_lookup: dict[str, int] = {}
    largest_component = max(components, key=len) if components else set()
    for component_id, members in enumerate(components):
        for node_id in members:
            component_lookup[str(node_id)] = component_id

    clustering_lookup = nx.clustering(nx_graph) if nx_graph.number_of_nodes() else {}
    rows: list[dict[str, Any]] = []
    for node_id, members in graph.get("nodes", {}).items():
        node_frame = frame.iloc[list(members)].copy()
        lens_values = lens[list(members)] if len(members) else np.zeros((0, 2), dtype=float)
        row: dict[str, Any] = {
            "config_id": config_id,
            "node_id": node_id,
            "n_members": int(len(members)),
            "member_indices": _json_list(_sample_ids_for_members(graph, list(members))),
            "example_pl_names": _json_list(node_frame.get("pl_name", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()[:5]),
            "example_hostnames": _json_list(node_frame.get("hostname", pd.Series(dtype="string")).dropna().astype(str).unique().tolist()[:5]),
            "degree": int(nx_graph.degree(node_id)) if node_id in nx_graph else 0,
            "clustering_coefficient": float(clustering_lookup.get(node_id, 0.0)),
            "component_id": int(component_lookup.get(str(node_id), -1)),
            "is_in_largest_component": bool(node_id in largest_component),
            "lens_1_mean": float(np.mean(lens_values[:, 0])) if len(lens_values) else 0.0,
            "lens_2_mean": float(np.mean(lens_values[:, 1])) if len(lens_values) else 0.0,
            "lens_1_std": float(np.std(lens_values[:, 0])) if len(lens_values) else 0.0,
            "lens_2_std": float(np.std(lens_values[:, 1])) if len(lens_values) else 0.0,
        }

        for feature in used_features:
            if feature not in node_frame.columns:
                continue
            values = pd.to_numeric(node_frame[feature], errors="coerce")
            if values.notna().any():
                row[f"mean_{feature}"] = float(values.mean())
                row[f"median_{feature}"] = float(values.median())
                row[f"q25_{feature}"] = float(values.quantile(0.25))
                row[f"q75_{feature}"] = float(values.quantile(0.75))
            row[f"frac_{feature}_observed"] = _source_fraction(node_frame, feature, "observed")
            row[f"frac_{feature}_physically_derived"] = _source_fraction(node_frame, feature, "physically_derived")
            row[f"frac_{feature}_imputed"] = _source_fraction(node_frame, feature, "imputed")
            row[f"dominant_{feature}_source"] = _dominant_source(node_frame, feature)

        for feature in GLOBAL_FEATURES:
            if feature in node_frame.columns:
                values = pd.to_numeric(node_frame[feature], errors="coerce")
                if values.notna().any():
                    row[f"mean_{feature}"] = float(values.mean())

        row.update(_global_traceability(node_frame, used_features))

        if "discoverymethod" in node_frame.columns:
            dominant, _ = _dominant_label(node_frame["discoverymethod"].astype("string"))
            row["discoverymethod_top"] = dominant
            row["discoverymethod_entropy"] = _label_entropy(node_frame["discoverymethod"].astype("string"))
        if "disc_year" in node_frame.columns:
            disc_year = pd.to_numeric(node_frame["disc_year"], errors="coerce")
            if disc_year.notna().any():
                row["disc_year_median"] = float(disc_year.median())
                row["disc_year_min"] = int(disc_year.min())
                row["disc_year_max"] = int(disc_year.max())
        if "radius_class" in node_frame.columns:
            dominant, _ = _dominant_label(node_frame["radius_class"].astype("string"))
            row["radius_class_top"] = dominant
        if "planet_regime" in node_frame.columns:
            dominant, _ = _dominant_label(node_frame["planet_regime"].astype("string"))
            row["planet_regime_top"] = dominant

        rows.append(row)
    return pd.DataFrame(rows)


def build_edge_table(
    graph: dict,
    physical_df: pd.DataFrame,
    used_features: list[str],
    config_id: str,
) -> pd.DataFrame:
    frame = physical_df.copy()
    frame["imputation_fraction_row"] = _compute_row_imputation_fraction(frame, used_features)
    nodes = graph.get("nodes", {})
    seen: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []

    for source, targets in graph.get("links", {}).items():
        source_members = set(nodes.get(source, []))
        for target in targets:
            edge = tuple(sorted((source, target)))
            if source == target or edge in seen:
                continue
            seen.add(edge)
            target_members = set(nodes.get(target, []))
            shared_positions = sorted(source_members & target_members)
            shared_frame = frame.iloc[shared_positions].copy() if shared_positions else frame.iloc[[]].copy()
            source_size = len(source_members)
            target_size = len(target_members)
            union_size = len(source_members | target_members)
            rows.append(
                {
                    "config_id": config_id,
                    "source": edge[0],
                    "target": edge[1],
                    "n_shared_members": int(len(shared_positions)),
                    "jaccard_members": float(len(shared_positions) / union_size) if union_size else 0.0,
                    "source_size": int(source_size),
                    "target_size": int(target_size),
                    "mean_imputation_fraction_shared": float(shared_frame["imputation_fraction_row"].mean())
                    if len(shared_frame)
                    else 0.0,
                    "physically_derived_fraction_shared": _global_traceability(shared_frame, used_features).get(
                        "physically_derived_fraction", 0.0
                    )
                    if len(shared_frame)
                    else 0.0,
                    "imputed_fraction_shared": _global_traceability(shared_frame, used_features).get("imputed_fraction", 0.0)
                    if len(shared_frame)
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def _standardize_metric_columns(metrics_df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    standardized = pd.DataFrame(index=metrics_df.index)
    for column in columns:
        values = pd.to_numeric(metrics_df[column], errors="coerce")
        mean = values.mean(skipna=True)
        std = values.std(skipna=True, ddof=0)
        if pd.isna(std) or std == 0:
            standardized[column] = values.where(values.isna(), 0.0)
        else:
            standardized[column] = (values - mean) / std
    return standardized


def compare_mapper_graphs(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df is None or metrics_df.empty or len(metrics_df) < 2:
        return pd.DataFrame()
    frame = metrics_df.copy()
    metric_columns = [column for column in GRAPH_DISTANCE_COLUMNS if column in frame.columns]
    if not metric_columns:
        return pd.DataFrame()
    standardized = _standardize_metric_columns(frame, metric_columns)
    rows: list[dict[str, Any]] = []
    for left_index, right_index in combinations(frame.index.tolist(), 2):
        common_columns = [
            column
            for column in standardized.columns
            if pd.notna(standardized.loc[left_index, column]) and pd.notna(standardized.loc[right_index, column])
        ]
        if common_columns:
            delta = standardized.loc[left_index, common_columns] - standardized.loc[right_index, common_columns]
            distance = float(np.linalg.norm(delta.to_numpy(dtype=float)))
        else:
            distance = None
        left = frame.loc[left_index]
        right = frame.loc[right_index]
        rows.append(
            {
                "graph_a": left["config_id"],
                "graph_b": right["config_id"],
                "space_a": left.get("space"),
                "lens_a": left.get("lens"),
                "space_b": right.get("space"),
                "lens_b": right.get("lens"),
                "metric_zscore_l2_distance": distance,
                "common_metric_count": int(len(common_columns)),
                "metrics_used": ", ".join(common_columns),
            }
        )
    return pd.DataFrame(rows)
