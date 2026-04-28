from __future__ import annotations

import numpy as np
import pandas as pd


def add_toi_decomposition(regions: pd.DataFrame) -> pd.DataFrame:
    """Return a table that makes the multiplicative TOI formula auditable.

    Expected optional columns: shadow_score, I_R3, C_phys, S_net, TOI.
    Missing columns are filled with NaN and an interpretation flag.
    """
    df = regions.copy()
    for col in ["shadow_score", "I_R3", "C_phys", "S_net", "TOI"]:
        if col not in df.columns:
            df[col] = np.nan
    df["one_minus_I_R3"] = 1 - df["I_R3"]
    df["TOI_recomputed"] = df["shadow_score"] * df["one_minus_I_R3"] * df["C_phys"] * df["S_net"]
    df["TOI_abs_error"] = (df["TOI"] - df["TOI_recomputed"]).abs()
    df["dominant_toi_driver"] = df.apply(_dominant_toi_driver, axis=1)
    return df


def _dominant_toi_driver(row: pd.Series) -> str:
    factors = {
        "shadow_score": row.get("shadow_score", np.nan),
        "low_imputation": row.get("one_minus_I_R3", np.nan),
        "physical_continuity": row.get("C_phys", np.nan),
        "network_support": row.get("S_net", np.nan),
    }
    valid = {k: v for k, v in factors.items() if pd.notna(v)}
    if not valid:
        return "unknown"
    # For multiplicative scores, small factors usually constrain the final score.
    bottleneck = min(valid, key=valid.get)
    strongest = max(valid, key=valid.get)
    return f"bottleneck={bottleneck}; strongest={strongest}"


def add_ati_decomposition(anchors: pd.DataFrame) -> pd.DataFrame:
    """Return anchor-level ATI decomposition.

    Expected optional columns: TOI, delta_rel_neighbors_best, r3_imputation_score,
    anchor_representativeness, ATI.
    """
    df = anchors.copy()
    for col in ["TOI", "delta_rel_neighbors_best", "r3_imputation_score", "anchor_representativeness", "ATI"]:
        if col not in df.columns:
            df[col] = np.nan
    df["positive_delta_rel_neighbors_best"] = df["delta_rel_neighbors_best"].clip(lower=0)
    df["one_minus_anchor_I_R3"] = 1 - df["r3_imputation_score"]
    df["ATI_recomputed"] = (
        df["TOI"]
        * df["positive_delta_rel_neighbors_best"]
        * df["one_minus_anchor_I_R3"]
        * df["anchor_representativeness"]
    )
    df["ATI_abs_error"] = (df["ATI"] - df["ATI_recomputed"]).abs()
    df["dominant_ati_driver"] = df.apply(_dominant_ati_driver, axis=1)
    return df


def _dominant_ati_driver(row: pd.Series) -> str:
    factors = {
        "TOI_region": row.get("TOI", np.nan),
        "local_deficit": row.get("positive_delta_rel_neighbors_best", np.nan),
        "anchor_low_imputation": row.get("one_minus_anchor_I_R3", np.nan),
        "anchor_representativeness": row.get("anchor_representativeness", np.nan),
    }
    valid = {k: v for k, v in factors.items() if pd.notna(v)}
    if not valid:
        return "unknown"
    bottleneck = min(valid, key=valid.get)
    strongest = max(valid, key=valid.get)
    return f"bottleneck={bottleneck}; strongest={strongest}"


def summarize_deficit_by_radius(deficits: pd.DataFrame) -> pd.DataFrame:
    """Summarize delta_rel by radius so that 'best' is not overinterpreted."""
    if deficits.empty:
        return pd.DataFrame()
    required = {"node_id", "anchor_pl_name", "radius_type", "delta_rel_neighbors"}
    if not required.issubset(deficits.columns):
        return pd.DataFrame()
    pivot = deficits.pivot_table(
        index=["node_id", "anchor_pl_name"],
        columns="radius_type",
        values="delta_rel_neighbors",
        aggfunc="max",
    ).reset_index()
    radius_cols = [c for c in pivot.columns if c not in {"node_id", "anchor_pl_name"}]
    pivot["delta_rel_neighbors_mean"] = pivot[radius_cols].mean(axis=1, skipna=True)
    pivot["delta_rel_neighbors_median"] = pivot[radius_cols].median(axis=1, skipna=True)
    pivot["delta_rel_neighbors_best"] = pivot[radius_cols].max(axis=1, skipna=True).clip(lower=0)
    pivot["best_radius"] = pivot[radius_cols].idxmax(axis=1)
    return pivot
