from __future__ import annotations

import numpy as np
import pandas as pd

from .imputation_audit import anchor_imputation_score
from .r3_geometry import z_matrix


def medoid_index(frame: pd.DataFrame) -> int | None:
    valid = frame[frame["r3_valid"]].copy()
    if valid.empty:
        return None
    matrix = z_matrix(valid)
    distances = np.linalg.norm(matrix[:, None, :] - matrix[None, :, :], axis=2)
    sums = distances.sum(axis=1)
    return int(valid.index[np.argmin(sums)])


def select_anchor(node_frame: pd.DataFrame, variables: list[str]) -> tuple[pd.Series | None, str]:
    valid = node_frame[node_frame["r3_valid"]].copy()
    if valid.empty:
        return None, "No hay miembros con R3 valido."
    medoid_idx = medoid_index(valid)
    if medoid_idx is None:
        return None, "No fue posible calcular el medoid del nodo."

    medoid_point = valid.loc[[medoid_idx], ["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].to_numpy(dtype=float)[0]
    valid["anchor_r3_imputation_score"] = valid.apply(lambda row: anchor_imputation_score(row, variables), axis=1)
    valid["anchor_is_rv"] = valid["discoverymethod"].astype("string").fillna("Unknown").eq("Radial Velocity")
    coords = valid[["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].to_numpy(dtype=float)
    valid["distance_to_medoid"] = np.linalg.norm(coords - medoid_point, axis=1)
    valid["name_completeness"] = valid.get("pl_name", pd.Series(index=valid.index, dtype="string")).astype("string").notna().astype(int)
    valid["metadata_completeness"] = valid[["disc_year", "disc_facility"]].notna().sum(axis=1)

    valid = valid.sort_values(
        by=["anchor_is_rv", "anchor_r3_imputation_score", "distance_to_medoid", "name_completeness", "metadata_completeness"],
        ascending=[False, True, True, False, False],
    )
    anchor = valid.iloc[0].copy()
    if bool(anchor["anchor_is_rv"]):
        reason = "Selected among R3-valid RV members, prioritizing lower imputation and proximity to the node medoid."
    else:
        reason = "No RV member satisfied the priority filters; selected the least-imputed R3-valid member nearest to the node medoid."
    return anchor, reason

