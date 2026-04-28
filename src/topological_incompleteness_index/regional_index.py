from __future__ import annotations
import numpy as np
import pandas as pd

def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)

def compute_regional_toi(nodes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = nodes.copy()

    shadow = _num(out, "shadow_score")
    if shadow.max() > 1:
        shadow = shadow / shadow.max()

    if "I_R3" in out.columns:
        i_r3 = _num(out, "I_R3").clip(0, 1)
    else:
        i_r3 = _num(out, "mean_imputation_fraction").clip(0, 1)

    dist = _num(out, "physical_distance_v_to_N1", default=np.nan)
    if dist.isna().all():
        dist = _num(out, "physical_neighbor_distance", default=0.0)
    sigma_cfg = cfg.get("physical_continuity_sigma")
    if sigma_cfg is None:
        positive = dist[np.isfinite(dist) & (dist > 0)]
        sigma = float(positive.median()) if len(positive) else 1.0
    else:
        sigma = float(sigma_cfg)
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    c_phys = np.exp(-(dist.fillna(sigma) ** 2) / (2 * sigma**2))

    n_members = _num(out, "n_members", 1).clip(lower=1)
    size_weight = np.log1p(n_members) / np.log1p(float(n_members.max()))
    degree = _num(out, "degree", 0).clip(lower=0)
    dmax = float(degree.max()) if degree.max() > 0 else 1.0
    degree_weight = 0.5 + 0.5 * np.log1p(degree) / np.log1p(dmax)
    s_net = size_weight * degree_weight

    out["toi_shadow_component"] = shadow
    out["toi_imputation_component"] = 1 - i_r3
    out["toi_physical_continuity_component"] = c_phys
    out["toi_network_support_component"] = s_net
    out["toi_score"] = shadow * (1 - i_r3) * c_phys * s_net
    out["toi_sigma_used"] = sigma

    q = float(cfg.get("top_quantile", 0.90))
    threshold = out["toi_score"].quantile(q) if len(out) else np.nan
    out["toi_top_region"] = out["toi_score"] >= threshold
    return out.sort_values("toi_score", ascending=False)
