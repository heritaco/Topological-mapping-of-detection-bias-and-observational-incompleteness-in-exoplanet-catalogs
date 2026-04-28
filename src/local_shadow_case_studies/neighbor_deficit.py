from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .graph_context import case_neighborhood
from .r3_geometry import centroid, z_matrix


RADIUS_ORDER = ["r_node_median", "r_node_q75", "r_kNN"]


def delta_rel(n_exp: float | None, n_obs: float, epsilon: float) -> float | None:
    if n_exp is None:
        return None
    return float((n_exp - n_obs) / (n_exp + epsilon))


def classify_deficit(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "not_available"
    if value <= 0.1:
        return "no_deficit"
    if value <= 0.3:
        return "weak_deficit"
    if value <= 0.6:
        return "moderate_deficit"
    return "strong_deficit"


def caution_text() -> str:
    return (
        "Interpret this as local topological deficit under a reference neighborhood, not as confirmed missing planets "
        "or an absolute count of real unseen objects."
    )


def _radius_values(anchor: pd.Series, node_valid: pd.DataFrame, universe_valid: pd.DataFrame) -> dict[str, float | None]:
    anchor_point = anchor[["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].to_numpy(dtype=float)
    node_others = node_valid[node_valid["row_index"] != anchor["row_index"]].copy()
    values: dict[str, float | None] = {"r_node_median": None, "r_node_q75": None, "r_kNN": None}
    if not node_others.empty:
        node_distances = np.linalg.norm(z_matrix(node_others) - anchor_point, axis=1)
        values["r_node_median"] = float(np.median(node_distances))
        values["r_node_q75"] = float(np.percentile(node_distances, 75))
    universe_others = universe_valid[universe_valid["row_index"] != anchor["row_index"]].copy()
    if not universe_others.empty:
        distances = np.sort(np.linalg.norm(z_matrix(universe_others) - anchor_point, axis=1))
        k = min(10, max(3, int(math.floor(math.sqrt(len(universe_valid))))))
        k_index = min(len(distances), k) - 1
        if k_index >= 0:
            values["r_kNN"] = float(distances[k_index])
    return values


def _count_in_ball(anchor: pd.Series, frame: pd.DataFrame, radius: float) -> int:
    if frame.empty or radius is None or not np.isfinite(radius):
        return 0
    anchor_point = anchor[["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].to_numpy(dtype=float)
    subset = frame[frame["r3_valid"]].copy()
    if subset.empty:
        return 0
    distances = np.linalg.norm(z_matrix(subset) - anchor_point, axis=1)
    return int(np.sum(distances <= radius))


def _analog_candidates(
    target_node_id: str,
    node_summary_frame: pd.DataFrame,
    centroid_map: dict[str, np.ndarray],
    tau: float,
    shadow_quantile: float,
    min_members: int,
    min_valid_fraction: float,
) -> list[str]:
    if target_node_id not in centroid_map:
        return []
    shadow_cut = float(pd.to_numeric(node_summary_frame["shadow_score"], errors="coerce").quantile(shadow_quantile))
    target_center = centroid_map[target_node_id]
    candidates: list[str] = []
    for _, row in node_summary_frame.iterrows():
        node_id = str(row["node_id"])
        if node_id == target_node_id or node_id not in centroid_map:
            continue
        n_members = float(pd.to_numeric(pd.Series([row.get("n_members")]), errors="coerce").iloc[0])
        r3_valid_fraction = float(pd.to_numeric(pd.Series([row.get("r3_valid_fraction")]), errors="coerce").fillna(0.0).iloc[0])
        shadow_score = float(pd.to_numeric(pd.Series([row.get("shadow_score")]), errors="coerce").fillna(np.nan).iloc[0])
        if n_members < min_members or r3_valid_fraction < min_valid_fraction or shadow_score > shadow_cut:
            continue
        distance = float(np.linalg.norm(centroid_map[node_id] - target_center))
        if distance <= tau:
            candidates.append(node_id)
    return candidates


def _local_universe_members(node_id: str, membership: pd.DataFrame, n1_nodes: list[str], n2_nodes: list[str]) -> pd.DataFrame:
    node_set = {str(node_id), *[str(value) for value in n1_nodes], *[str(value) for value in n2_nodes]}
    subset = membership[membership["node_id"].astype(str).isin(node_set)].copy()
    if subset.empty:
        return subset
    return subset.sort_values(["row_index", "node_id"]).drop_duplicates(subset=["row_index"], keep="first")


def expected_missing_direction(anchor: pd.Series, node_frame: pd.DataFrame, neighbor_frame: pd.DataFrame) -> str:
    if "rv_proxy" in anchor and pd.notna(anchor["rv_proxy"]):
        node_proxy = pd.to_numeric(node_frame.get("rv_proxy"), errors="coerce").median()
        neighbor_proxy = pd.to_numeric(neighbor_frame.get("rv_proxy"), errors="coerce").median()
        if pd.notna(node_proxy) and pd.notna(neighbor_proxy) and node_proxy >= neighbor_proxy:
            return "menor proxy RV a escala orbital comparable"
    return "menor masa planetaria a periodo/semieje comparable"


def compute_neighbor_deficits(
    case_id: str,
    node_id: str,
    anchor: pd.Series,
    node_frame: pd.DataFrame,
    n1_frame: pd.DataFrame,
    n2_frame: pd.DataFrame,
    membership_all: pd.DataFrame,
    graph,
    node_summary_frame: pd.DataFrame,
    centroid_map: dict[str, np.ndarray],
    analog_tau: float,
    analog_shadow_quantile: float,
    analog_min_nodes: int,
    analog_min_members: int,
    analog_min_valid_fraction: float,
    neighbor_reference_min_size: int,
    analog_count_cap: int,
    epsilon: float,
    warnings: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    universe = _local_universe_members(node_id, membership_all, n1_frame["node_id"].astype(str).unique().tolist(), n2_frame["node_id"].astype(str).unique().tolist())
    universe_valid = universe[universe["r3_valid"]].copy()
    node_valid = node_frame[node_frame["r3_valid"]].copy()
    radius_values = _radius_values(anchor, node_valid, universe_valid)
    neighbor_valid = n1_frame[n1_frame["r3_valid"]].copy()
    second_valid = n2_frame[n2_frame["r3_valid"]].copy()
    rows: list[dict[str, object]] = []
    best_neighbor_delta = None
    best_class = "not_available"

    analogs = _analog_candidates(
        target_node_id=node_id,
        node_summary_frame=node_summary_frame,
        centroid_map=centroid_map,
        tau=analog_tau,
        shadow_quantile=analog_shadow_quantile,
        min_members=analog_min_members,
        min_valid_fraction=analog_min_valid_fraction,
    )[:analog_count_cap]
    if len(analogs) < analog_min_nodes:
        warnings.append(f"WARNING: {node_id} no encontro suficientes nodos analogos de baja sombra para referencia analog.")

    for radius_type in RADIUS_ORDER:
        radius = radius_values.get(radius_type)
        if radius is None or not np.isfinite(radius):
            rows.append(
                {
                    "case_id": case_id,
                    "node_id": node_id,
                    "anchor_pl_name": str(anchor.get("pl_name", "Unknown")),
                    "radius_type": radius_type,
                    "radius_value": None,
                    "N_obs": None,
                    "N_exp_neighbors": None,
                    "delta_N_neighbors": None,
                    "delta_rel_neighbors": None,
                    "deficit_class_neighbors": "not_available",
                    "N_exp_analog": None,
                    "delta_N_analog": None,
                    "delta_rel_analog": None,
                    "deficit_class_analog": "not_available",
                    "n_analog_nodes": len(analogs),
                    "expected_missing_direction": "not_available",
                    "caution_text": caution_text(),
                }
            )
            continue
        n_obs = _count_in_ball(anchor, universe_valid[universe_valid["row_index"] != anchor["row_index"]], radius)

        ref_frame = neighbor_valid if len(neighbor_valid) >= neighbor_reference_min_size else second_valid
        n_exp_neighbors = float(_count_in_ball(anchor, ref_frame, radius)) if not ref_frame.empty else None
        delta_neighbors = (n_exp_neighbors - n_obs) if n_exp_neighbors is not None else None
        delta_rel_neighbors = delta_rel(n_exp_neighbors, n_obs, epsilon)
        class_neighbors = classify_deficit(delta_rel_neighbors)

        analog_counts: list[int] = []
        for analog_node in analogs:
            analog_ctx = case_neighborhood(analog_node, graph, node_summary_frame, membership_all)
            analog_universe = _local_universe_members(analog_node, membership_all, analog_ctx.n1_nodes, analog_ctx.n2_nodes)
            analog_valid = analog_universe[analog_universe["r3_valid"]].copy()
            analog_node_valid = analog_valid[analog_valid["node_id"].astype(str) == analog_node].copy()
            if analog_node_valid.empty:
                continue
            analog_center = centroid(analog_node_valid)
            if analog_center is None:
                continue
            analog_node_valid = analog_node_valid.copy()
            coords = z_matrix(analog_node_valid)
            medoid = analog_node_valid.iloc[int(np.argmin(np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2).sum(axis=1)))]
            analog_counts.append(_count_in_ball(medoid, analog_valid[analog_valid["row_index"] != medoid["row_index"]], radius))
        n_exp_analog = float(np.median(analog_counts)) if len(analog_counts) >= analog_min_nodes else None
        delta_analog = (n_exp_analog - n_obs) if n_exp_analog is not None else None
        delta_rel_analog = delta_rel(n_exp_analog, n_obs, epsilon)
        class_analog = classify_deficit(delta_rel_analog)
        direction = expected_missing_direction(anchor, node_frame, ref_frame if not ref_frame.empty else node_frame)
        row = {
            "case_id": case_id,
            "node_id": node_id,
            "anchor_pl_name": str(anchor.get("pl_name", "Unknown")),
            "radius_type": radius_type,
            "radius_value": float(radius),
            "N_obs": int(n_obs),
            "N_exp_neighbors": n_exp_neighbors,
            "delta_N_neighbors": delta_neighbors,
            "delta_rel_neighbors": delta_rel_neighbors,
            "deficit_class_neighbors": class_neighbors,
            "N_exp_analog": n_exp_analog,
            "delta_N_analog": delta_analog,
            "delta_rel_analog": delta_rel_analog,
            "deficit_class_analog": class_analog,
            "n_analog_nodes": len(analog_counts),
            "expected_missing_direction": direction,
            "caution_text": caution_text(),
        }
        rows.append(row)
        if delta_rel_neighbors is not None and (best_neighbor_delta is None or delta_rel_neighbors > best_neighbor_delta):
            best_neighbor_delta = delta_rel_neighbors
            best_class = class_neighbors

    summary = {
        "delta_rel_neighbors_best": best_neighbor_delta,
        "deficit_class": best_class,
        "n_analog_nodes": len(analogs),
    }
    return pd.DataFrame(rows), summary
