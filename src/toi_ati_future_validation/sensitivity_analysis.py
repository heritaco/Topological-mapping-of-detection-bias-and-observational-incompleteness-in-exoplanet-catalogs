from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd


def compute_region_sensitivity(regions: pd.DataFrame, weight_grid: dict[str, list[float]], *, epsilon: float = 1e-9) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame()
    df = regions.copy()
    combos = list(product(weight_grid.get("toi", [1.0]), weight_grid.get("imputation", [1.0]), weight_grid.get("deficit", [1.0]), weight_grid.get("representativeness", [1.0])))
    rank_columns: list[str] = []
    base_shadow = pd.to_numeric(df.get("shadow_score"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    base_low_imp = (1 - pd.to_numeric(df.get("I_R3"), errors="coerce").fillna(0)).clip(lower=0) + epsilon
    base_phys = pd.to_numeric(df.get("C_phys"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    base_net = pd.to_numeric(df.get("S_net"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    for idx, (w_shadow, w_imp, w_phys, w_net) in enumerate(combos):
        score = (base_shadow ** w_shadow) * (base_low_imp ** w_imp) * (base_phys ** w_phys) * (base_net ** w_net)
        rank_col = f"region_rank_{idx}"
        df[rank_col] = score.rank(ascending=False, method="average")
        rank_columns.append(rank_col)
    df["TOI_rank_mean_sensitivity"] = df[rank_columns].mean(axis=1)
    df["TOI_rank_std_sensitivity"] = df[rank_columns].std(axis=1, ddof=0)
    df["region_robustness_class"] = df.apply(_region_robustness_class, axis=1)
    return df[["node_id", "TOI_rank_mean_sensitivity", "TOI_rank_std_sensitivity", "region_robustness_class"]]


def compute_anchor_sensitivity(anchors: pd.DataFrame, weight_grid: dict[str, list[float]], *, epsilon: float = 1e-9) -> pd.DataFrame:
    if anchors.empty:
        return pd.DataFrame()
    df = anchors.copy()
    combos = list(product(weight_grid.get("toi", [1.0]), weight_grid.get("deficit", [1.0]), weight_grid.get("imputation", [1.0]), weight_grid.get("representativeness", [1.0])))
    rank_columns: list[str] = []
    base_toi = pd.to_numeric(df.get("TOI"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    base_deficit = pd.to_numeric(df.get("delta_rel_neighbors_best"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    base_low_imp = (1 - pd.to_numeric(df.get("r3_imputation_score"), errors="coerce").fillna(0)).clip(lower=0) + epsilon
    base_repr = pd.to_numeric(df.get("anchor_representativeness"), errors="coerce").fillna(0).clip(lower=0) + epsilon
    for idx, (w_toi, w_deficit, w_imp, w_repr) in enumerate(combos):
        score = (base_toi ** w_toi) * (base_deficit ** w_deficit) * (base_low_imp ** w_imp) * (base_repr ** w_repr)
        rank_col = f"anchor_rank_{idx}"
        df[rank_col] = score.rank(ascending=False, method="average")
        rank_columns.append(rank_col)
    df["ATI_rank_mean_sensitivity"] = df[rank_columns].mean(axis=1)
    df["ATI_rank_std_sensitivity"] = df[rank_columns].std(axis=1, ddof=0)
    return df[["anchor_pl_name", "node_id", "ATI_rank_mean_sensitivity", "ATI_rank_std_sensitivity"]]


def _region_robustness_class(row: pd.Series) -> str:
    rank_std = float(pd.to_numeric(pd.Series([row.get("TOI_rank_std_sensitivity")]), errors="coerce").fillna(0).iloc[0])
    if rank_std <= 2:
        return "robust_region"
    if rank_std <= 5:
        return "moderately_sensitive_region"
    return "fragile_region"

