from __future__ import annotations

import numpy as np
import pandas as pd

from .neighbor_deficit import classify_deficit
from .r3_geometry import R3Columns, anchor_imputed_fraction, anchor_r3_imputation_score, centroid, medoid_row, z_matrix


def unique_planets(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    if "row_index" in frame.columns:
        return frame.sort_values(["row_index", "node_id"]).drop_duplicates(subset=["row_index"], keep="first").copy()
    if "pl_name" not in frame.columns:
        return frame.copy()
    if "node_id" not in frame.columns:
        return frame.sort_values(["pl_name"]).drop_duplicates(subset=["pl_name"], keep="first").copy()
    return frame.sort_values(["pl_name", "node_id"]).drop_duplicates(subset=["pl_name"], keep="first").copy()


def select_anchor(node_members: pd.DataFrame, columns: R3Columns, prefer_method: str) -> tuple[pd.Series | None, str]:
    valid = unique_planets(node_members.dropna(subset=columns.z).copy())
    if valid.empty:
        return None, "No hay miembros con R3 valido."
    medoid = medoid_row(valid, columns.z)
    medoid_point = medoid[columns.z].to_numpy(dtype=float) if medoid is not None else np.zeros(len(columns.z))
    valid["anchor_is_preferred_method"] = valid.get("discoverymethod", pd.Series(index=valid.index, dtype="string")).astype("string").eq(prefer_method)
    valid["r3_imputation_score"] = valid.apply(lambda row: anchor_r3_imputation_score(row, columns), axis=1)
    valid["anchor_imputed_fraction"] = valid.apply(lambda row: anchor_imputed_fraction(row, columns), axis=1)
    valid["distance_to_medoid"] = np.linalg.norm(z_matrix(valid, columns.z) - medoid_point, axis=1)
    valid["metadata_completeness"] = valid[["disc_year", "disc_facility"]].notna().sum(axis=1)
    valid["name_completeness"] = valid.get("pl_name", pd.Series(index=valid.index, dtype="string")).astype("string").notna().astype(int)
    valid = valid.sort_values(
        by=["anchor_is_preferred_method", "anchor_imputed_fraction", "r3_imputation_score", "distance_to_medoid", "metadata_completeness", "name_completeness"],
        ascending=[False, True, True, True, False, False],
    )
    anchor = valid.iloc[0].copy()
    if bool(anchor["anchor_is_preferred_method"]):
        reason = "Selected among R3-valid preferred-method members, prioritizing lower imputation and medoid proximity."
    else:
        reason = "No preferred-method candidate dominated the filters; selected the least-imputed representative nearest to the medoid."
    return anchor, reason


def anchor_representativeness(anchor: pd.Series, node_members: pd.DataFrame, z_columns: list[str], epsilon: float) -> tuple[float, float | None]:
    valid = unique_planets(node_members.dropna(subset=z_columns).copy())
    if valid.empty:
        return 0.0, None
    center = centroid(valid, z_columns)
    if center is None:
        return 0.0, None
    anchor_point = anchor[z_columns].to_numpy(dtype=float)
    distances = np.linalg.norm(z_matrix(valid, z_columns) - center, axis=1)
    spread = float(np.mean(distances)) if len(distances) else epsilon
    spread = spread if np.isfinite(spread) and spread > 0 else epsilon
    distance = float(np.linalg.norm(anchor_point - center))
    value = float(np.exp(-(distance**2) / (2.0 * spread**2)))
    return value, distance


def compute_ati(toi: float, delta_rel_best: float | None, anchor_imputed_fraction_value: float, representativeness: float) -> float:
    deficit_component = max(0.0, float(delta_rel_best or 0.0))
    return float(toi * deficit_component * (1.0 - anchor_imputed_fraction_value) * representativeness)


def expected_incompleteness_direction(discoverymethod: str) -> str:
    if str(discoverymethod) == "Radial Velocity":
        return "menor masa planetaria o menor proxy RV a escala orbital comparable"
    return "vecinos compatibles bajo referencia local en region fisico-orbital comparable"


def build_anchor_interpretation(row: pd.Series) -> str:
    return (
        f"Ancla {row['anchor_pl_name']} en {row['node_id']} con ATI={float(row['ATI']):.3f}, "
        f"deficit {row['deficit_class']} y uso prudente para priorizacion observacional."
    )


def classify_anchor_deficit(value: float) -> str:
    return classify_deficit(value)
