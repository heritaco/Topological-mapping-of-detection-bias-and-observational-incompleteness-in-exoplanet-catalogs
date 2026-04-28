from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=float)))


def compute_robust_anchor_indices(
    anchors: pd.DataFrame,
    stability: pd.DataFrame,
    anchor_sensitivity: pd.DataFrame,
    *,
    penalize_negative_large_radius: bool = True,
    penalize_imputation: bool = True,
    penalize_small_nodes: bool = True,
    epsilon: float = 1e-9,
) -> pd.DataFrame:
    if anchors.empty:
        return pd.DataFrame()
    df = anchors.copy()
    stability_cols = [
        "anchor_pl_name",
        "node_id",
        "delta_rel_mean",
        "delta_rel_median",
        "delta_rel_min",
        "delta_rel_max",
        "delta_rel_std",
        "n_positive_radii",
        "all_radii_positive",
        "any_large_radius_negative",
        "radius_sensitivity_score",
        "stable_deficit_score",
        "deficit_stability_class",
        "n_members",
        "I_R3",
        "S_net",
        "top_method",
        "expected_incompleteness_direction",
        "future_observation_direction",
    ]
    available_stability = [column for column in stability_cols if column in stability.columns]
    df = df.merge(stability[available_stability].drop_duplicates(subset=["anchor_pl_name", "node_id"]), on=["anchor_pl_name", "node_id"], how="left")
    if not anchor_sensitivity.empty:
        df = df.merge(anchor_sensitivity, on=["anchor_pl_name", "node_id"], how="left")

    original = pd.to_numeric(df.get("ATI"), errors="coerce").fillna(0)
    delta_best = pd.to_numeric(df.get("delta_rel_neighbors_best"), errors="coerce").fillna(0).clip(lower=0)
    delta_mean = pd.to_numeric(df.get("delta_rel_mean"), errors="coerce").fillna(0)
    delta_std = pd.to_numeric(df.get("delta_rel_std"), errors="coerce").fillna(0)
    n_positive = pd.to_numeric(df.get("n_positive_radii"), errors="coerce").fillna(0)
    r3_imp = pd.to_numeric(df.get("r3_imputation_score"), errors="coerce").fillna(0).clip(lower=0, upper=1)
    n_members = pd.to_numeric(df.get("n_members"), errors="coerce").fillna(0)
    median_members = max(float(n_members[n_members > 0].median()) if (n_members > 0).any() else 10.0, 10.0)

    stability_factor = (n_positive / 3.0).clip(lower=0, upper=1)
    signal_factor = pd.Series(sigmoid(delta_mean / (delta_std + float(epsilon))), index=df.index)
    radius_penalty_ratio = (delta_mean.clip(lower=0) / (delta_best + float(epsilon))).clip(lower=0)
    large_radius_penalty = pd.Series(1.0, index=df.index)
    if penalize_negative_large_radius:
        large_radius_penalty = np.where(df.get("any_large_radius_negative", False), 0.5, 1.0)
        large_radius_penalty = pd.Series(large_radius_penalty, index=df.index, dtype=float)
    imputation_penalty = pd.Series(1.0, index=df.index, dtype=float)
    if penalize_imputation:
        imputation_penalty = 1.0 - 0.5 * r3_imp
    size_penalty = pd.Series(1.0, index=df.index, dtype=float)
    if penalize_small_nodes:
        size_penalty = 0.5 + 0.5 * np.clip(np.sqrt(n_members / median_members), 0, 1)

    df["ATI_original"] = original
    df["ATI_stable_simple"] = original * stability_factor
    df["ATI_stable_signal_to_noise"] = original * signal_factor
    df["ATI_radius_penalized"] = original * radius_penalty_ratio
    df["ATI_conservative"] = df["ATI_stable_simple"] * large_radius_penalty * imputation_penalty * size_penalty
    df["rank_ATI_original"] = df["ATI_original"].rank(ascending=False, method="min")
    df["rank_ATI_conservative"] = df["ATI_conservative"].rank(ascending=False, method="min")
    df["rank_shift"] = df["rank_ATI_conservative"] - df["rank_ATI_original"]
    df["caution_text"] = df.apply(_anchor_caution_text, axis=1)
    keep = [
        "anchor_pl_name",
        "node_id",
        "ATI_original",
        "ATI_stable_simple",
        "ATI_stable_signal_to_noise",
        "ATI_radius_penalized",
        "ATI_conservative",
        "rank_ATI_original",
        "rank_ATI_conservative",
        "rank_shift",
        "deficit_stability_class",
        "caution_text",
    ]
    extras = [column for column in ["TOI", "delta_rel_neighbors_best", "delta_rel_mean", "stable_deficit_score", "radius_sensitivity_score", "r3_imputation_score", "n_members", "S_net", "top_method", "expected_incompleteness_direction", "future_observation_direction", "ATI_rank_mean_sensitivity", "ATI_rank_std_sensitivity"] if column in df.columns]
    return df[keep + extras]


def compute_robust_region_indices(regions: pd.DataFrame, sensitivity: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame()
    df = regions.copy()
    if not sensitivity.empty:
        df = df.merge(sensitivity, on="node_id", how="left")
    df["region_robustness_class"] = df.apply(_region_interpretation_class, axis=1)
    df["interpretation_text"] = df.apply(_region_interpretation_text, axis=1)
    keep = [
        "node_id",
        "TOI",
        "TOI_rank",
        "TOI_rank_mean_sensitivity",
        "TOI_rank_std_sensitivity",
        "shadow_score",
        "I_R3",
        "C_phys",
        "S_net",
        "region_robustness_class",
        "interpretation_text",
    ]
    existing = [column for column in keep if column in df.columns]
    return df[existing].rename(columns={"TOI": "TOI_original", "TOI_rank": "TOI_rank_original"})


def _anchor_caution_text(row: pd.Series) -> str:
    messages: list[str] = []
    if str(row.get("deficit_stability_class", "")) in {"radius_sensitive_deficit", "unstable_due_to_large_radius"}:
        messages.append("Deficit sensible al radio; usar como caso exploratorio.")
    if float(pd.to_numeric(pd.Series([row.get("r3_imputation_score")]), errors="coerce").fillna(0).iloc[0]) > 0.2:
        messages.append("La imputacion del ancla no es despreciable.")
    if float(pd.to_numeric(pd.Series([row.get("n_members")]), errors="coerce").fillna(0).iloc[0]) < 10:
        messages.append("Nodo pequeno; el soporte local puede ser fragil.")
    if not messages:
        return "Caso util para priorizacion observacional prudente."
    return " ".join(messages)


def _region_interpretation_class(row: pd.Series) -> str:
    rank_std = float(pd.to_numeric(pd.Series([row.get("TOI_rank_std_sensitivity")]), errors="coerce").fillna(0).iloc[0])
    i_r3 = float(pd.to_numeric(pd.Series([row.get("I_R3")]), errors="coerce").fillna(0).iloc[0])
    if i_r3 > 0.2:
        return "imputation_sensitive_region"
    if rank_std <= 2:
        return "robust_priority_region"
    if rank_std <= 5:
        return "moderately_sensitive_region"
    return "fragile_region"


def _region_interpretation_text(row: pd.Series) -> str:
    label = str(row.get("region_robustness_class", "unknown"))
    if label == "robust_priority_region":
        return "La region conserva prioridad alta bajo cambios razonables de pesos y ofrece una lectura regional defendible."
    if label == "imputation_sensitive_region":
        return "La region mantiene interes, pero su lectura depende mas de la imputacion en R^3."
    if label == "moderately_sensitive_region":
        return "La region cambia de rango bajo algunos pesos y conviene tratarla como prioridad intermedia."
    return "La region es fragil a cambios de pesos y requiere auditoria tecnica antes de una lectura fuerte."
