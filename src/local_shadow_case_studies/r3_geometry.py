from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


R3_LOG_COLUMNS = {
    "pl_bmasse": "r3_log_mass",
    "pl_orbper": "r3_log_period",
    "pl_orbsmax": "r3_log_semimajor",
}
R3_Z_COLUMNS = {
    "pl_bmasse": "r3_z_mass",
    "pl_orbper": "r3_z_period",
    "pl_orbsmax": "r3_z_semimajor",
}


@dataclass
class GlobalR3Stats:
    means: dict[str, float]
    stds: dict[str, float]
    n_valid: int


def safe_log10_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    valid = numeric.notna() & np.isfinite(numeric) & (numeric > 0)
    out = pd.Series(np.nan, index=series.index, dtype=float)
    out.loc[valid] = np.log10(numeric.loc[valid].astype(float))
    return out, ~valid


def add_r3_coordinates(frame: pd.DataFrame, warnings: list[str], label: str) -> pd.DataFrame:
    out = frame.copy()
    invalid_any = pd.Series(False, index=out.index)
    for source, target in R3_LOG_COLUMNS.items():
        values, invalid = safe_log10_series(out.get(source, pd.Series(index=out.index, dtype=float)))
        out[target] = values
        invalid_any = invalid_any | invalid
    out["r3_valid"] = out[list(R3_LOG_COLUMNS.values())].notna().all(axis=1)
    lost = int((~out["r3_valid"]).sum())
    if lost > 0:
        warnings.append(f"WARNING: {label} pierde {lost} registros para R3 por valores faltantes/no positivos.")
    return out


def compute_global_r3_stats(frame: pd.DataFrame) -> GlobalR3Stats:
    valid = frame[frame["r3_valid"]].copy()
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for source, log_column in R3_LOG_COLUMNS.items():
        series = pd.to_numeric(valid[log_column], errors="coerce")
        mean = float(series.mean()) if not series.empty else 0.0
        std = float(series.std(ddof=0)) if not series.empty else 1.0
        if not np.isfinite(std) or std <= 0:
            std = 1.0
        means[source] = mean
        stds[source] = std
    return GlobalR3Stats(means=means, stds=stds, n_valid=int(len(valid)))


def apply_r3_standardization(frame: pd.DataFrame, stats: GlobalR3Stats) -> pd.DataFrame:
    out = frame.copy()
    for source, z_column in R3_Z_COLUMNS.items():
        log_column = R3_LOG_COLUMNS[source]
        out[z_column] = (pd.to_numeric(out[log_column], errors="coerce") - stats.means[source]) / stats.stds[source]
    return out


def z_matrix(frame: pd.DataFrame) -> np.ndarray:
    subset = frame[list(R3_Z_COLUMNS.values())].apply(pd.to_numeric, errors="coerce")
    return subset.to_numpy(dtype=float)


def centroid(frame: pd.DataFrame) -> np.ndarray | None:
    valid = frame[frame["r3_valid"]].copy()
    if valid.empty:
        return None
    return np.nanmean(z_matrix(valid), axis=0)


def covariance_diag(frame: pd.DataFrame) -> list[float] | None:
    valid = frame[frame["r3_valid"]].copy()
    if len(valid) < 2:
        return None
    matrix = z_matrix(valid)
    cov = np.cov(matrix.T)
    return np.diag(cov).astype(float).tolist()


def mean_distance_to_centroid(frame: pd.DataFrame, center: np.ndarray | None) -> float | None:
    valid = frame[frame["r3_valid"]].copy()
    if center is None or valid.empty:
        return None
    distances = np.linalg.norm(z_matrix(valid) - center, axis=1)
    return float(np.mean(distances)) if len(distances) else None


def centroid_distance(a: pd.DataFrame, b: pd.DataFrame) -> float | None:
    ca = centroid(a)
    cb = centroid(b)
    if ca is None or cb is None:
        return None
    return float(np.linalg.norm(ca - cb))


def neighbor_overlap_score(node_frame: pd.DataFrame, neighbor_frame: pd.DataFrame) -> float | None:
    node_valid = node_frame[node_frame["r3_valid"]].copy()
    neighbor_valid = neighbor_frame[neighbor_frame["r3_valid"]].copy()
    if node_valid.empty or neighbor_valid.empty:
        return None
    node_center = centroid(node_valid)
    if node_center is None:
        return None
    radius = mean_distance_to_centroid(node_valid, node_center)
    if radius is None:
        return None
    distances = np.linalg.norm(z_matrix(neighbor_valid) - node_center, axis=1)
    return float(np.mean(distances <= radius))


def describe_case_geometry(node_frame: pd.DataFrame, n1_frame: pd.DataFrame, n2_frame: pd.DataFrame) -> dict[str, Any]:
    node_center = centroid(node_frame)
    n1_center = centroid(n1_frame)
    n2_center = centroid(n2_frame)
    return {
        "centroid_v_r3": node_center.tolist() if node_center is not None else None,
        "centroid_N1_r3": n1_center.tolist() if n1_center is not None else None,
        "centroid_N2_r3": n2_center.tolist() if n2_center is not None else None,
        "physical_distance_v_to_N1": centroid_distance(node_frame, n1_frame),
        "physical_distance_v_to_N2": centroid_distance(node_frame, n2_frame),
        "cov_diag": covariance_diag(node_frame),
        "spread_r3": mean_distance_to_centroid(node_frame, node_center),
        "neighbor_overlap_score": neighbor_overlap_score(node_frame, n1_frame),
        "n_r3_valid": int(node_frame["r3_valid"].sum()),
        "r3_valid_fraction": float(node_frame["r3_valid"].mean()) if len(node_frame) else 0.0,
    }


def build_region_membership(case_id: str, node_id: str, node_nodes: list[str], n1_nodes: list[str], n2_nodes: list[str], membership: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    specs = [
        ("node", node_nodes, True),
        ("N1", n1_nodes, False),
        ("N2", n2_nodes, False),
    ]
    for region_type, region_nodes, belongs_to_target in specs:
        if not region_nodes:
            continue
        subset = membership[membership["node_id"].astype(str).isin(region_nodes)].copy()
        if subset.empty:
            continue
        subset = subset.sort_values(["row_index", "node_id"]).drop_duplicates(subset=["row_index"], keep="first")
        subset["case_id"] = case_id
        subset["node_id"] = node_id
        subset["region_type"] = region_type
        subset["belongs_to_target_node"] = belongs_to_target
        frames.append(subset)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

