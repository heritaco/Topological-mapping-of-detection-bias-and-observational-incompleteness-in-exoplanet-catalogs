from __future__ import annotations

import numpy as np
import pandas as pd


def reconstruct_shadow_score(frame: pd.DataFrame, epsilon: float) -> pd.Series:
    top_method_fraction = pd.to_numeric(frame.get("top_method_fraction"), errors="coerce").fillna(0.0)
    method_entropy_norm = pd.to_numeric(frame.get("method_entropy_norm"), errors="coerce").fillna(1.0)
    boundary = pd.to_numeric(frame.get("method_l1_boundary_N1", frame.get("method_l1_boundary")), errors="coerce").fillna(0.0)
    n_members = pd.to_numeric(frame.get("n_members"), errors="coerce").fillna(1.0).clip(lower=1.0)
    n_max = float(n_members.max()) if float(n_members.max()) > 0 else 1.0
    size_weight = np.log1p(n_members) / max(np.log1p(n_max), epsilon)
    return top_method_fraction * (1.0 - method_entropy_norm) * boundary * size_weight


def physical_continuity(distance: pd.Series, sigma: float) -> pd.Series:
    return np.exp(-((distance.astype(float) ** 2) / (2.0 * sigma**2)))


def compute_toi_scores(node_frame: pd.DataFrame, sigma: float, epsilon: float, min_node_members: int, high_priority_quantile: float) -> pd.DataFrame:
    out = node_frame.copy()
    if "shadow_score" not in out.columns or pd.to_numeric(out["shadow_score"], errors="coerce").isna().all():
        out["shadow_score"] = reconstruct_shadow_score(out, epsilon)
    shadow = pd.to_numeric(out["shadow_score"], errors="coerce").fillna(0.0).clip(lower=0.0)
    i_r3 = pd.to_numeric(out["I_R3"], errors="coerce").fillna(1.0).clip(lower=0.0, upper=1.0)
    d_phys = pd.to_numeric(out["physical_distance_v_to_N1"], errors="coerce")
    c_phys_raw = physical_continuity(d_phys.fillna(np.nan), sigma)
    out["C_phys"] = c_phys_raw.where(d_phys.notna(), np.nan)
    c_phys_for_toi = c_phys_raw.where(d_phys.notna(), 0.0)
    out["TOI"] = shadow * (1.0 - i_r3) * c_phys_for_toi * pd.to_numeric(out["S_net"], errors="coerce").fillna(0.0)
    out = out.sort_values("TOI", ascending=False).reset_index(drop=True)
    out["TOI_rank"] = np.arange(1, len(out) + 1)

    p90 = float(out["TOI"].quantile(high_priority_quantile)) if len(out) else np.nan
    p75 = float(out["TOI"].quantile(0.75)) if len(out) else np.nan

    def _classify(row: pd.Series) -> str:
        toi = float(row.get("TOI", 0.0))
        n_members = float(pd.to_numeric(pd.Series([row.get("n_members")]), errors="coerce").fillna(0.0).iloc[0])
        degree = float(pd.to_numeric(pd.Series([row.get("degree")]), errors="coerce").fillna(0.0).iloc[0])
        i_r3_value = float(pd.to_numeric(pd.Series([row.get("I_R3")]), errors="coerce").fillna(1.0).iloc[0])
        if n_members < min_node_members or degree <= 0:
            return "isolated_or_low_support_region"
        if np.isfinite(p90) and toi >= p90 and i_r3_value <= 0.2:
            return "high_toi_region"
        if np.isfinite(p90) and toi >= p90 and i_r3_value > 0.2:
            return "imputation_sensitive_region"
        if np.isfinite(p75) and toi >= p75:
            return "moderate_toi_region"
        return "low_toi_region"

    out["region_class"] = out.apply(_classify, axis=1)
    out["caution_text"] = (
        "TOI es un ranking topologico para priorizacion observacional; no equivale a una confirmacion de objetos ausentes."
    )
    out["interpretation_text"] = out.apply(
        lambda row: (
            f"Nodo {row['node_id']} priorizado como {row['region_class']} con TOI={float(row['TOI']):.3f}, "
            f"shadow_score={float(pd.to_numeric(pd.Series([row.get('shadow_score')]), errors='coerce').fillna(0.0).iloc[0]):.3f}, "
            f"I_R3={float(pd.to_numeric(pd.Series([row.get('I_R3')]), errors='coerce').fillna(1.0).iloc[0]):.3f} y "
            "lectura prudente de posible submuestreo topologico."
        ),
        axis=1,
    )
    return out
