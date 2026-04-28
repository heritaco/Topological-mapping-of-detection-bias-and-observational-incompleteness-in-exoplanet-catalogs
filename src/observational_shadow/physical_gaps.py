from __future__ import annotations

import numpy as np
import pandas as pd


def available_physical_variables(frame: pd.DataFrame, requested: list[str], warnings: list[str], config_id: str) -> list[str]:
    available = []
    for variable in requested:
        if variable in frame.columns:
            available.append(variable)
        else:
            warnings.append(f"WARNING: {config_id} no contiene la variable fisica/orbital {variable}; se omite.")
    return available


def add_physical_neighbor_gaps(
    node_profiles: pd.DataFrame,
    neighbor_profiles: pd.DataFrame,
    physical_variables: list[str],
) -> pd.DataFrame:
    out = node_profiles.copy()
    for variable in physical_variables:
        node_col = f"mean_{variable}"
        neighbor_col = f"neighbor_mean_{variable}"
        if node_col not in out.columns or neighbor_col not in neighbor_profiles.columns:
            continue
        out[neighbor_col] = neighbor_profiles[neighbor_col]
        delta_col = f"delta_{variable}"
        scaled_col = f"scaled_delta_{variable}"
        out[delta_col] = pd.to_numeric(out[node_col], errors="coerce") - pd.to_numeric(out[neighbor_col], errors="coerce")
        std = float(pd.to_numeric(out[node_col], errors="coerce").std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            out[scaled_col] = np.nan
        else:
            out[scaled_col] = out[delta_col] / std
    scaled_cols = [f"scaled_delta_{variable}" for variable in physical_variables if f"scaled_delta_{variable}" in out.columns]
    if scaled_cols:
        values = out[scaled_cols].apply(pd.to_numeric, errors="coerce")
        out["physical_neighbor_distance"] = np.sqrt(np.nansum(np.square(values.to_numpy(dtype=float)), axis=1))
        out.loc[values.notna().sum(axis=1) == 0, "physical_neighbor_distance"] = np.nan
    else:
        out["physical_neighbor_distance"] = np.nan
    return out

