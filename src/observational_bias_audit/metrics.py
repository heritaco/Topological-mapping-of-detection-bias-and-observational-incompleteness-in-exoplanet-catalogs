from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score


def purity_from_counts(counts: np.ndarray | list[float]) -> float:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    return float(values.max() / total)


def shannon_entropy_from_counts(counts: np.ndarray | list[float]) -> float:
    values = np.asarray(counts, dtype=float)
    total = float(values.sum())
    if total <= 0:
        return 0.0
    probs = values[values > 0] / total
    return float(-(probs * np.log(probs)).sum())


def normalized_entropy_from_counts(counts: np.ndarray | list[float], universe_k: int) -> float:
    if universe_k <= 1:
        return float("nan")
    denom = np.log(float(universe_k))
    if denom <= 0:
        return float("nan")
    return float(shannon_entropy_from_counts(counts) / denom)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    result = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.dropna()
    if valid.empty:
        return result
    order = valid.sort_values().index.tolist()
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)
    q = np.empty(m, dtype=float)
    running = 1.0
    for pos in range(m - 1, -1, -1):
        rank = pos + 1
        running = min(running, ranked[pos] * m / rank)
        q[pos] = running
    q = np.clip(q, 0.0, 1.0)
    result.loc[order] = q
    return result


def nominal_assortativity_from_codes(edge_pairs: np.ndarray, node_labels: np.ndarray, n_categories: int) -> float:
    if edge_pairs.size == 0 or n_categories <= 1:
        return float("nan")
    source_labels = node_labels[edge_pairs[:, 0]]
    target_labels = node_labels[edge_pairs[:, 1]]
    if source_labels.size == 0:
        return float("nan")
    directed_pairs = np.vstack(
        [
            np.column_stack([source_labels, target_labels]),
            np.column_stack([target_labels, source_labels]),
        ]
    )
    matrix = np.zeros((n_categories, n_categories), dtype=float)
    np.add.at(matrix, (directed_pairs[:, 0], directed_pairs[:, 1]), 1.0)
    total = matrix.sum()
    if total <= 0:
        return float("nan")
    e_matrix = matrix / total
    a = e_matrix.sum(axis=1)
    b = e_matrix.sum(axis=0)
    expected = float(np.sum(a * b))
    denom = 1.0 - expected
    if denom <= 0:
        return float("nan")
    return float((np.trace(e_matrix) - expected) / denom)


def component_beta1(edge_table: pd.DataFrame, node_ids: list[str]) -> float:
    if edge_table.empty:
        return float("nan")
    node_set = set(node_ids)
    internal = edge_table[
        edge_table["source"].astype(str).isin(node_set)
        & edge_table["target"].astype(str).isin(node_set)
    ].copy()
    v = len(node_set)
    if v == 0:
        return 0.0
    value = int(len(internal) - v + 1)
    return float(max(value, 0))


def build_node_method_matrices(
    membership_with_metadata: pd.DataFrame,
    node_order: list[str] | None = None,
    method_order: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    if membership_with_metadata.empty:
        empty = pd.DataFrame()
        return empty, empty, []
    working = membership_with_metadata.copy()
    working["discoverymethod"] = working["discoverymethod"].astype("string").fillna("Unknown")
    count_df = pd.crosstab(working["node_id"].astype(str), working["discoverymethod"].astype(str))
    if node_order is not None:
        count_df = count_df.reindex(node_order, fill_value=0)
    if method_order is not None:
        count_df = count_df.reindex(columns=method_order, fill_value=0)
    else:
        count_df = count_df.reindex(sorted(count_df.columns.tolist()), axis=1)
    total = count_df.sum(axis=1).replace(0, np.nan)
    fraction_df = count_df.div(total, axis=0).fillna(0.0)
    return count_df, fraction_df, [str(value) for value in count_df.columns.tolist()]


def build_node_metrics(
    config_id: str,
    membership_with_metadata: pd.DataFrame,
    node_table: pd.DataFrame,
    edge_table: pd.DataFrame,
    peripheral_degree_threshold: int,
    peripheral_component_max_nodes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    node_order = node_table["node_id"].astype(str).tolist() if not node_table.empty else None
    count_df, fraction_df, method_order = build_node_method_matrices(
        membership_with_metadata=membership_with_metadata,
        node_order=node_order,
        method_order=None,
    )
    if count_df.empty:
        return pd.DataFrame(), count_df, fraction_df, {"method_universe": [], "peripheral_rule": ""}

    base = node_table.copy()
    base["node_id"] = base["node_id"].astype(str)
    base = base.set_index("node_id", drop=False)
    base = base.loc[count_df.index]

    universe_k = len(method_order)
    component_sizes = base.get("component_id", pd.Series(dtype="Int64")).astype("string").value_counts()
    rows: list[dict[str, Any]] = []
    excluded_nodes: list[str] = []
    for node_id in count_df.index:
        counts = count_df.loc[node_id].to_numpy(dtype=float)
        n_members = int(counts.sum())
        if n_members <= 0:
            excluded_nodes.append(str(node_id))
            continue
        fractions = fraction_df.loc[node_id]
        top_method = str(fractions.idxmax()) if not fractions.empty else "Unknown"
        top_fraction = float(fractions.max()) if not fractions.empty else 0.0
        row = {
            "config_id": config_id,
            "node_id": str(node_id),
            "n_members": n_members,
            "top_method": top_method,
            "top_method_fraction": top_fraction,
            "method_entropy": shannon_entropy_from_counts(counts),
            "method_entropy_norm": normalized_entropy_from_counts(counts, universe_k=universe_k),
            "mean_imputation_fraction": pd.to_numeric(pd.Series([base.loc[node_id].get("mean_imputation_fraction")]), errors="coerce").iloc[0],
            "mean_physically_derived_fraction": pd.to_numeric(pd.Series([base.loc[node_id].get("physically_derived_fraction")]), errors="coerce").iloc[0],
            "degree": pd.to_numeric(pd.Series([base.loc[node_id].get("degree")]), errors="coerce").iloc[0],
            "component_id": pd.to_numeric(pd.Series([base.loc[node_id].get("component_id")]), errors="coerce").iloc[0],
            "lens_1_mean": pd.to_numeric(pd.Series([base.loc[node_id].get("lens_1_mean")]), errors="coerce").iloc[0],
            "lens_2_mean": pd.to_numeric(pd.Series([base.loc[node_id].get("lens_2_mean")]), errors="coerce").iloc[0],
        }
        rows.append(row)

    node_metrics = pd.DataFrame(rows)
    if node_metrics.empty:
        return node_metrics, count_df, fraction_df, {"method_universe": method_order, "excluded_zero_member_nodes": excluded_nodes, "peripheral_rule": ""}

    node_metrics["degree"] = pd.to_numeric(node_metrics["degree"], errors="coerce")
    node_metrics["component_id"] = pd.to_numeric(node_metrics["component_id"], errors="coerce")
    component_id_as_text = node_metrics["component_id"].astype("Int64").astype("string")
    node_metrics["component_n_nodes"] = component_id_as_text.map(component_sizes).astype(float)
    node_metrics["is_peripheral"] = (
        node_metrics["degree"].fillna(0) <= int(peripheral_degree_threshold)
    ) | (
        node_metrics["component_n_nodes"].fillna(float("inf")) <= int(peripheral_component_max_nodes)
    )

    count_out = count_df.reset_index().rename(columns={"node_id": "node_id"})
    count_out.insert(0, "config_id", config_id)
    fraction_out = fraction_df.reset_index().rename(columns={"node_id": "node_id"})
    fraction_out.insert(0, "config_id", config_id)
    metadata = {
        "method_universe": method_order,
        "excluded_zero_member_nodes": excluded_nodes,
        "peripheral_rule": (
            f"is_peripheral = (degree <= {peripheral_degree_threshold}) "
            f"or (component_n_nodes <= {peripheral_component_max_nodes})"
        ),
        "has_edges": not edge_table.empty,
    }
    return node_metrics.sort_values(["component_id", "node_id"]).reset_index(drop=True), count_out, fraction_out, metadata


def build_component_method_summary(
    config_id: str,
    membership_with_metadata: pd.DataFrame,
    node_metrics: pd.DataFrame,
    edge_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if membership_with_metadata.empty or node_metrics.empty:
        return pd.DataFrame(), pd.DataFrame()
    annotated = membership_with_metadata.merge(
        node_metrics[["node_id", "component_id", "is_peripheral"]],
        on="node_id",
        how="inner",
    )
    if annotated.empty:
        return pd.DataFrame(), pd.DataFrame()
    annotated["member_key"] = annotated["row_index"].astype(str)
    composition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for component_id, component_memberships in annotated.groupby("component_id", dropna=False):
        component_nodes = node_metrics[node_metrics["component_id"] == component_id].copy()
        unique_members = component_memberships.drop_duplicates(subset=["member_key"]).copy()
        method_counts = unique_members["discoverymethod"].astype("string").value_counts().sort_values(ascending=False)
        counts = method_counts.to_numpy(dtype=float)
        dominant_method = str(method_counts.index[0]) if len(method_counts) else "Unknown"
        dominant_fraction = purity_from_counts(counts)
        entropy = shannon_entropy_from_counts(counts)
        entropy_norm = normalized_entropy_from_counts(counts, universe_k=len(method_counts))
        for method_name, method_count in method_counts.items():
            composition_rows.append(
                {
                    "config_id": config_id,
                    "component_id": component_id,
                    "method": str(method_name),
                    "count": int(method_count),
                    "fraction": float(method_count / method_counts.sum()) if method_counts.sum() else 0.0,
                }
            )
        summary_rows.append(
            {
                "config_id": config_id,
                "component_id": component_id,
                "n_nodes": int(len(component_nodes)),
                "n_members_unique": int(len(unique_members)),
                "top_method": dominant_method,
                "top_method_fraction": dominant_fraction,
                "method_entropy": entropy,
                "method_entropy_norm": entropy_norm,
                "mean_imputation_fraction": float(pd.to_numeric(component_nodes["mean_imputation_fraction"], errors="coerce").mean()),
                "mean_physically_derived_fraction": float(pd.to_numeric(component_nodes["mean_physically_derived_fraction"], errors="coerce").mean()),
                "fraction_peripheral_nodes": float(pd.to_numeric(component_nodes["is_peripheral"], errors="coerce").mean()),
                "beta_1_internal": component_beta1(edge_table, component_nodes["node_id"].astype(str).tolist()),
                "methods_present": ", ".join(str(value) for value in method_counts.index[:5].tolist()),
            }
        )
    return (
        pd.DataFrame(summary_rows).sort_values(["component_id"]).reset_index(drop=True),
        pd.DataFrame(composition_rows).sort_values(["component_id", "count"], ascending=[True, False]).reset_index(drop=True),
    )


def build_global_bias_row(
    config_id: str,
    node_metrics: pd.DataFrame,
    membership_with_metadata: pd.DataFrame,
    count_matrix: pd.DataFrame,
    edge_table: pd.DataFrame,
) -> dict[str, Any]:
    if node_metrics.empty or count_matrix.empty:
        return {
            "config_id": config_id,
            "weighted_mean_purity": np.nan,
            "weighted_mean_entropy": np.nan,
            "weighted_mean_entropy_norm": np.nan,
            "node_method_nmi": np.nan,
            "dominant_method_assortativity": np.nan,
            "mean_node_imputation_fraction": np.nan,
            "mean_physically_derived_fraction": np.nan,
            "n_nodes": 0,
            "n_edges": int(len(edge_table)),
            "n_methods": 0,
        }
    wide = count_matrix.set_index("node_id")
    value_matrix = wide.drop(columns=["config_id"], errors="ignore").to_numpy(dtype=float)
    sizes = value_matrix.sum(axis=1)
    purities = np.array([purity_from_counts(row) for row in value_matrix], dtype=float)
    entropies = np.array([shannon_entropy_from_counts(row) for row in value_matrix], dtype=float)
    universe_k = wide.drop(columns=["config_id"], errors="ignore").shape[1]
    entropies_norm = np.array([normalized_entropy_from_counts(row, universe_k=universe_k) for row in value_matrix], dtype=float)
    weighted_mean_purity = float(np.average(purities, weights=sizes)) if sizes.sum() > 0 else np.nan
    weighted_mean_entropy = float(np.average(entropies, weights=sizes)) if sizes.sum() > 0 else np.nan
    valid_norm = np.isfinite(entropies_norm)
    weighted_mean_entropy_norm = float(np.average(entropies_norm[valid_norm], weights=sizes[valid_norm])) if valid_norm.any() else np.nan

    nmi = np.nan
    methods = membership_with_metadata["discoverymethod"].astype("string").fillna("Unknown")
    if membership_with_metadata["node_id"].nunique() > 1 and methods.nunique() > 1:
        nmi = float(
            normalized_mutual_info_score(
                membership_with_metadata["node_id"].astype(str),
                methods.astype(str),
            )
        )

    assortativity = np.nan
    if not edge_table.empty and node_metrics["top_method"].nunique(dropna=True) > 1:
        node_lookup = {node_id: idx for idx, node_id in enumerate(node_metrics["node_id"].astype(str).tolist())}
        pairs = []
        for row in edge_table.itertuples(index=False):
            source = str(getattr(row, "source"))
            target = str(getattr(row, "target"))
            if source in node_lookup and target in node_lookup:
                pairs.append((node_lookup[source], node_lookup[target]))
        if pairs:
            method_categories = sorted(node_metrics["top_method"].astype(str).dropna().unique().tolist())
            method_lookup = {method: idx for idx, method in enumerate(method_categories)}
            node_labels = node_metrics["top_method"].astype(str).map(method_lookup).to_numpy(dtype=int)
            assortativity = nominal_assortativity_from_codes(np.asarray(pairs, dtype=int), node_labels=node_labels, n_categories=len(method_lookup))

    return {
        "config_id": config_id,
        "weighted_mean_purity": weighted_mean_purity,
        "weighted_mean_entropy": weighted_mean_entropy,
        "weighted_mean_entropy_norm": weighted_mean_entropy_norm,
        "node_method_nmi": nmi,
        "dominant_method_assortativity": assortativity,
        "mean_node_imputation_fraction": float(pd.to_numeric(node_metrics["mean_imputation_fraction"], errors="coerce").mean()),
        "mean_physically_derived_fraction": float(pd.to_numeric(node_metrics["mean_physically_derived_fraction"], errors="coerce").mean()),
        "n_nodes": int(len(node_metrics)),
        "n_edges": int(len(edge_table)),
        "n_methods": int(methods.nunique()),
    }


def build_central_vs_peripheral_summary(node_metrics: pd.DataFrame) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    config_id = str(node_metrics["config_id"].iloc[0]) if "config_id" in node_metrics.columns and not node_metrics.empty else "unknown"
    rows: list[dict[str, Any]] = []
    for is_peripheral, group in node_metrics.groupby("is_peripheral"):
        if group.empty:
            continue
        dominant_mode = group["top_method"].mode(dropna=True)
        rows.append(
            {
                "config_id": config_id,
                "region": "peripheral" if bool(is_peripheral) else "central",
                "n_nodes": int(len(group)),
                "mean_purity": float(pd.to_numeric(group["top_method_fraction"], errors="coerce").mean()),
                "mean_entropy": float(pd.to_numeric(group["method_entropy"], errors="coerce").mean()),
                "mean_imputation_fraction": float(pd.to_numeric(group["mean_imputation_fraction"], errors="coerce").mean()),
                "mean_node_size": float(pd.to_numeric(group["n_members"], errors="coerce").mean()),
                "most_frequent_dominant_method": str(dominant_mode.iloc[0]) if not dominant_mode.empty else "Unknown",
            }
        )
    return pd.DataFrame(rows)
