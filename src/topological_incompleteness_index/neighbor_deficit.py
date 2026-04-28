from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .network_metrics import NodeNeighborhood, node_neighborhood
from .r3_geometry import centroid, medoid_row, z_matrix


@dataclass
class NeighborDeficitParameters:
    epsilon: float
    knn_min: int
    knn_max: int
    analog_min_nodes: int
    analog_shadow_quantile_max: float
    analog_physical_distance_quantile: float
    reference_min_planets: int
    min_node_members: int
    min_r3_valid_fraction: float


def delta_rel(n_exp: float | None, n_obs: float, epsilon: float) -> float | None:
    if n_exp is None or not np.isfinite(n_exp):
        return None
    return float((n_exp - n_obs) / (n_exp + epsilon))


def delta_rel_best(values: list[float | None]) -> float:
    valid = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(max([0.0, *valid])) if valid else 0.0


def classify_deficit(value: float) -> str:
    if value <= 0.10:
        return "no_deficit"
    if value <= 0.30:
        return "weak_deficit"
    if value <= 0.60:
        return "moderate_deficit"
    return "strong_deficit"


def _unique_planets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "row_index" in frame.columns:
        return frame.sort_values(["row_index", "node_id"]).drop_duplicates(subset=["row_index"], keep="first").copy()
    return frame.sort_values(["pl_name", "node_id"]).drop_duplicates(subset=["pl_name"], keep="first").copy()


def _pairwise_distances(frame: pd.DataFrame, anchor: pd.Series, z_cols: list[str]) -> pd.Series:
    valid = frame.dropna(subset=z_cols).copy()
    if valid.empty:
        return pd.Series(dtype=float)
    anchor_point = anchor[z_cols].to_numpy(dtype=float)
    distances = np.linalg.norm(z_matrix(valid, z_cols) - anchor_point, axis=1)
    return pd.Series(distances, index=valid.index)


def _local_universe(membership: pd.DataFrame, node_id: str, neighborhood: NodeNeighborhood) -> pd.DataFrame:
    nodes = {str(node_id), *[str(value) for value in neighborhood.n1_nodes], *[str(value) for value in neighborhood.n2_nodes]}
    subset = membership[membership["node_id"].astype(str).isin(nodes)].copy()
    return _unique_planets(subset)


def _radius_values(anchor: pd.Series, node_planets: pd.DataFrame, universe: pd.DataFrame, z_cols: list[str], params: NeighborDeficitParameters) -> dict[str, float | None]:
    node_others = node_planets.copy()
    if "row_index" in node_others.columns and "row_index" in anchor.index:
        node_others = node_others[node_others["row_index"] != anchor["row_index"]].copy()
    elif "pl_name" in node_others.columns:
        node_others = node_others[node_others["pl_name"].astype(str) != str(anchor.get("pl_name", ""))].copy()
    node_distances = _pairwise_distances(node_others, anchor, z_cols)
    universe_others = universe.copy()
    if "row_index" in universe_others.columns and "row_index" in anchor.index:
        universe_others = universe_others[universe_others["row_index"] != anchor["row_index"]].copy()
    elif "pl_name" in universe_others.columns:
        universe_others = universe_others[universe_others["pl_name"].astype(str) != str(anchor.get("pl_name", ""))].copy()
    universe_distances = np.sort(_pairwise_distances(universe_others, anchor, z_cols).to_numpy(dtype=float))
    radii: dict[str, float | None] = {"r_node_median": None, "r_node_q75": None, "r_kNN": None}
    if len(node_distances):
        radii["r_node_median"] = float(np.median(node_distances))
        radii["r_node_q75"] = float(np.percentile(node_distances, 75))
    if len(universe_distances):
        k = min(params.knn_max, max(params.knn_min, int(math.floor(math.sqrt(len(universe))))))
        radii["r_kNN"] = float(universe_distances[min(len(universe_distances), k) - 1])
    return radii


def _count_in_ball(frame: pd.DataFrame, anchor: pd.Series, radius: float, z_cols: list[str]) -> int:
    if frame.empty or radius is None or not np.isfinite(radius):
        return 0
    return int((_pairwise_distances(frame, anchor, z_cols) <= radius).sum())


def _distance_threshold(node_geometry: pd.DataFrame, quantile: float) -> float:
    centers = node_geometry.dropna(subset=["centroid_z_mass", "centroid_z_period", "centroid_z_semimajor"]).copy()
    if len(centers) < 2:
        return np.inf
    matrix = centers[["centroid_z_mass", "centroid_z_period", "centroid_z_semimajor"]].to_numpy(dtype=float)
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    values = distances[np.triu_indices_from(distances, k=1)]
    values = values[np.isfinite(values) & (values > 0)]
    return float(np.quantile(values, quantile)) if len(values) else np.inf


def _analog_nodes(node_id: str, node_geometry: pd.DataFrame, regional_scores: pd.DataFrame, params: NeighborDeficitParameters) -> list[str]:
    merged = node_geometry.merge(regional_scores[["node_id", "shadow_score", "n_members", "r3_valid_fraction"]], on="node_id", how="left")
    target = merged[merged["node_id"].astype(str) == str(node_id)].copy()
    if target.empty:
        return []
    tau = _distance_threshold(merged, params.analog_physical_distance_quantile)
    if not np.isfinite(tau):
        return []
    shadow_cut = float(pd.to_numeric(merged["shadow_score"], errors="coerce").quantile(params.analog_shadow_quantile_max))
    target_center = target[["centroid_z_mass", "centroid_z_period", "centroid_z_semimajor"]].iloc[0].to_numpy(dtype=float)
    candidates: list[str] = []
    for _, row in merged.iterrows():
        other_id = str(row["node_id"])
        if other_id == str(node_id):
            continue
        if float(pd.to_numeric(pd.Series([row.get("n_members")]), errors="coerce").fillna(0.0).iloc[0]) < params.min_node_members:
            continue
        if float(pd.to_numeric(pd.Series([row.get("r3_valid_fraction")]), errors="coerce").fillna(0.0).iloc[0]) < params.min_r3_valid_fraction:
            continue
        shadow_score = float(pd.to_numeric(pd.Series([row.get("shadow_score")]), errors="coerce").fillna(np.inf).iloc[0])
        if shadow_score > shadow_cut:
            continue
        other_center = row[["centroid_z_mass", "centroid_z_period", "centroid_z_semimajor"]].to_numpy(dtype=float)
        if np.any(~np.isfinite(other_center)):
            continue
        distance = float(np.linalg.norm(other_center - target_center))
        if distance <= tau:
            candidates.append(other_id)
    return sorted(candidates)


def compute_anchor_neighbor_deficits(
    config_id: str,
    node_id: str,
    anchor: pd.Series,
    membership: pd.DataFrame,
    graph,
    node_geometry: pd.DataFrame,
    regional_scores: pd.DataFrame,
    z_cols: list[str],
    params: NeighborDeficitParameters,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, float | str | None]]:
    neighborhood = node_neighborhood(graph, node_id)
    node_planets = _unique_planets(membership[membership["node_id"].astype(str) == str(node_id)].copy())
    n1_planets = _unique_planets(membership[membership["node_id"].astype(str).isin(neighborhood.n1_nodes)].copy())
    n2_planets = _unique_planets(membership[membership["node_id"].astype(str).isin(neighborhood.n2_nodes)].copy())
    universe = _local_universe(membership, node_id, neighborhood)
    radii = _radius_values(anchor, node_planets, universe, z_cols, params)
    analog_nodes = _analog_nodes(node_id, node_geometry, regional_scores, params)
    rows: list[dict[str, object]] = []
    neighbor_values: list[float | None] = []
    analog_values: list[float | None] = []
    if len(analog_nodes) < params.analog_min_nodes:
        warnings.append(f"WARNING: {node_id} no tiene suficientes nodos analogos para referencia B.")

    for radius_type, radius_value in radii.items():
        warning_text = None
        reference_used = "N1"
        if radius_value is None or not np.isfinite(radius_value):
            warning_text = "radius_unavailable"
            n_obs = None
            n_exp_neighbors = None
            n_exp_analog = None
        else:
            universe_others = universe.copy()
            if "row_index" in universe_others.columns and "row_index" in anchor.index:
                universe_others = universe_others[universe_others["row_index"] != anchor["row_index"]].copy()
            n_obs = _count_in_ball(universe_others, anchor, float(radius_value), z_cols)
            if len(n1_planets.dropna(subset=z_cols)) >= params.reference_min_planets:
                n_exp_neighbors = float(_count_in_ball(n1_planets, anchor, float(radius_value), z_cols))
            elif len(n2_planets.dropna(subset=z_cols)) >= params.reference_min_planets:
                reference_used = "N2"
                n_exp_neighbors = float(_count_in_ball(n2_planets, anchor, float(radius_value), z_cols))
            else:
                reference_used = "unavailable"
                warning_text = "insufficient_neighbors"
                n_exp_neighbors = None
            analog_counts: list[int] = []
            for analog_node in analog_nodes:
                analog_neighborhood = node_neighborhood(graph, analog_node)
                analog_universe = _local_universe(membership, analog_node, analog_neighborhood)
                analog_node_planets = _unique_planets(membership[membership["node_id"].astype(str) == str(analog_node)].copy())
                analog_medoid = medoid_row(analog_node_planets, z_cols)
                if analog_medoid is None:
                    continue
                analog_universe_others = analog_universe.copy()
                if "row_index" in analog_universe_others.columns and "row_index" in analog_medoid.index:
                    analog_universe_others = analog_universe_others[analog_universe_others["row_index"] != analog_medoid["row_index"]].copy()
                analog_counts.append(_count_in_ball(analog_universe_others, analog_medoid, float(radius_value), z_cols))
            n_exp_analog = float(np.median(analog_counts)) if len(analog_counts) >= params.analog_min_nodes else None
            if n_exp_analog is None and warning_text is None:
                warning_text = "insufficient_analogs"
        delta_neighbors = (n_exp_neighbors - n_obs) if n_exp_neighbors is not None and n_obs is not None else None
        delta_rel_neighbors = delta_rel(n_exp_neighbors, float(n_obs), params.epsilon) if n_obs is not None else None
        delta_analog = (n_exp_analog - n_obs) if n_exp_analog is not None and n_obs is not None else None
        delta_rel_analog = delta_rel(n_exp_analog, float(n_obs), params.epsilon) if n_obs is not None else None
        neighbor_values.append(delta_rel_neighbors)
        analog_values.append(delta_rel_analog)
        rows.append(
            {
                "config_id": config_id,
                "node_id": node_id,
                "anchor_pl_name": str(anchor.get("pl_name", "Unknown")),
                "radius_type": radius_type,
                "radius_value": radius_value,
                "N_obs": n_obs,
                "N_exp_neighbors": n_exp_neighbors,
                "delta_N_neighbors": delta_neighbors,
                "delta_rel_neighbors": delta_rel_neighbors,
                "N_exp_analog": n_exp_analog,
                "delta_N_analog": delta_analog,
                "delta_rel_analog": delta_rel_analog,
                "n_analog_nodes": len(analog_nodes),
                "reference_used": reference_used,
                "warning": warning_text,
            }
        )
    deficit_frame = pd.DataFrame(rows)
    neighbor_clean = [float(value) for value in neighbor_values if value is not None and np.isfinite(value)]
    analog_clean = [float(value) for value in analog_values if value is not None and np.isfinite(value)]
    summary = {
        "delta_rel_neighbors_median_radius": next((float(row["delta_rel_neighbors"]) for _, row in deficit_frame.iterrows() if row["radius_type"] == "r_node_median" and pd.notna(row["delta_rel_neighbors"])), None),
        "delta_rel_neighbors_q75_radius": next((float(row["delta_rel_neighbors"]) for _, row in deficit_frame.iterrows() if row["radius_type"] == "r_node_q75" and pd.notna(row["delta_rel_neighbors"])), None),
        "delta_rel_neighbors_knn_radius": next((float(row["delta_rel_neighbors"]) for _, row in deficit_frame.iterrows() if row["radius_type"] == "r_kNN" and pd.notna(row["delta_rel_neighbors"])), None),
        "delta_rel_neighbors_mean": float(np.mean(neighbor_clean)) if neighbor_clean else None,
        "delta_rel_neighbors_median": float(np.median(neighbor_clean)) if neighbor_clean else None,
        "delta_rel_neighbors_best": delta_rel_best(neighbor_values),
        "delta_rel_analog_best": delta_rel_best(analog_values),
        "deficit_class": classify_deficit(delta_rel_best(neighbor_values)),
    }
    return deficit_frame, summary
