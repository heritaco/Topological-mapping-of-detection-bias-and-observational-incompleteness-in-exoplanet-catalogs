from __future__ import annotations

import numpy as np
import pandas as pd

from .validation import (
    classify_deficit,
    compute_delta_n,
    compute_delta_rel,
    deficit_stability_label,
    suspicious_delta_rel,
)

DEFICIT_COLUMN_ALIASES = {
    "anchor_pl_name": ["anchor_pl_name", "pl_name", "planet_name", "anchor_name"],
    "node_id": ["node_id", "node", "mapper_node_id"],
    "radius_type": ["radius_type", "radius", "radius_name"],
    "radius_value": ["radius_value", "radius_size", "radius_numeric"],
    "N_obs": ["N_obs", "n_obs", "observed_neighbors"],
    "N_exp_neighbors": ["N_exp_neighbors", "n_exp_neighbors", "expected_neighbors"],
    "delta_N_neighbors": ["delta_N_neighbors", "delta_n_neighbors", "Delta_N_neighbors"],
    "delta_rel_neighbors": ["delta_rel_neighbors", "Delta_rel_neighbors", "relative_deficit_neighbors"],
    "N_exp_analog": ["N_exp_analog", "n_exp_analog", "expected_analog_neighbors"],
    "delta_N_analog": ["delta_N_analog", "delta_n_analog", "Delta_N_analog"],
    "delta_rel_analog": ["delta_rel_analog", "Delta_rel_analog", "relative_deficit_analog"],
    "n_analog_nodes": ["n_analog_nodes", "analog_nodes_count"],
    "reference_used": ["reference_used", "reference", "reference_source"],
    "warning": ["warning", "warnings", "note"],
}

ORDERED_RADII = ["r_kNN", "r_node_median", "r_node_q75"]


def add_toi_decomposition(regions: pd.DataFrame) -> pd.DataFrame:
    """Return a table that makes the multiplicative TOI formula auditable."""
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
    bottleneck = min(valid, key=valid.get)
    strongest = max(valid, key=valid.get)
    return f"bottleneck={bottleneck}; strongest={strongest}"


def add_ati_decomposition(anchors: pd.DataFrame) -> pd.DataFrame:
    """Return anchor-level ATI decomposition."""
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


def normalize_deficit_columns(deficits: pd.DataFrame) -> pd.DataFrame:
    """Normalize deficit tables so downstream reporting uses stable names."""
    if deficits is None or deficits.empty:
        return pd.DataFrame()
    df = deficits.copy()
    rename_map: dict[str, str] = {}
    for canonical, aliases in DEFICIT_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    df = df.rename(columns=rename_map)
    return df


def audit_deficit_formulas(deficits: pd.DataFrame, *, epsilon: float = 1e-9) -> tuple[pd.DataFrame, dict[str, int]]:
    """Audit deficit formulas and recompute when the stored values mismatch."""
    df = normalize_deficit_columns(deficits)
    if df.empty:
        return df, {"raw_delta_rel_gt_one_count": 0, "recomputed_delta_rel_gt_one_count": 0, "mismatch_count": 0}

    required = ["anchor_pl_name", "node_id", "radius_type", "N_obs", "N_exp_neighbors"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"No se puede construir la tabla de deficit por radio porque falta {', '.join(missing)}.")

    for column in [
        "radius_value",
        "delta_N_neighbors",
        "delta_rel_neighbors",
        "N_exp_analog",
        "delta_N_analog",
        "delta_rel_analog",
        "n_analog_nodes",
        "reference_used",
        "warning",
    ]:
        if column not in df.columns:
            df[column] = np.nan

    df["delta_N_neighbors_recomputed"] = df.apply(
        lambda row: compute_delta_n(row.get("N_exp_neighbors"), row.get("N_obs")),
        axis=1,
    )
    df["delta_rel_neighbors_recomputed"] = df.apply(
        lambda row: compute_delta_rel(row.get("N_exp_neighbors"), row.get("N_obs"), epsilon=epsilon),
        axis=1,
    )

    def _formula_check(row: pd.Series) -> str:
        expected = row.get("N_exp_neighbors")
        observed = row.get("N_obs")
        stored_dn = row.get("delta_N_neighbors")
        stored_dr = row.get("delta_rel_neighbors")
        recomputed_dn = row.get("delta_N_neighbors_recomputed")
        recomputed_dr = row.get("delta_rel_neighbors_recomputed")
        if pd.isna(expected) or pd.isna(observed):
            return "missing_expected_or_observed"
        if pd.isna(stored_dn) or pd.isna(stored_dr):
            return "mismatch_recomputed_used"
        dn_ok = np.isclose(float(stored_dn), float(recomputed_dn), atol=1e-8, rtol=1e-6)
        dr_ok = np.isclose(float(stored_dr), float(recomputed_dr), atol=1e-8, rtol=1e-6)
        return "ok" if dn_ok and dr_ok else "mismatch_recomputed_used"

    df["deficit_formula_check"] = df.apply(_formula_check, axis=1)
    df["Delta_N_neighbors"] = np.where(
        df["deficit_formula_check"].eq("ok"),
        df["delta_N_neighbors"],
        df["delta_N_neighbors_recomputed"],
    )
    df["Delta_rel_neighbors"] = np.where(
        df["deficit_formula_check"].eq("ok"),
        df["delta_rel_neighbors"],
        df["delta_rel_neighbors_recomputed"],
    )

    if "N_exp_analog" in df.columns:
        df["Delta_N_analog"] = df.apply(
            lambda row: compute_delta_n(row.get("N_exp_analog"), row.get("N_obs")),
            axis=1,
        )
        df["Delta_rel_analog"] = df.apply(
            lambda row: compute_delta_rel(row.get("N_exp_analog"), row.get("N_obs"), epsilon=epsilon),
            axis=1,
        )
    else:
        df["Delta_N_analog"] = np.nan
        df["Delta_rel_analog"] = np.nan

    df["deficit_class_neighbors"] = df.apply(
        lambda row: classify_deficit(row.get("Delta_rel_neighbors"), row.get("N_exp_neighbors")),
        axis=1,
    )
    df["deficit_class_analog"] = df.apply(
        lambda row: classify_deficit(row.get("Delta_rel_analog"), row.get("N_exp_analog")),
        axis=1,
    )
    df["formula_warning"] = df.apply(
        lambda row: "delta_rel_gt_one_suspected_delta_n"
        if suspicious_delta_rel(
            row.get("delta_rel_neighbors"),
            row.get("delta_N_neighbors"),
            row.get("N_exp_neighbors"),
        )
        else "",
        axis=1,
    )
    df["interpretation_short"] = df.apply(_interpretation_short, axis=1)

    audit = {
        "raw_delta_rel_gt_one_count": int((pd.to_numeric(df["delta_rel_neighbors"], errors="coerce") > 1).sum()),
        "recomputed_delta_rel_gt_one_count": int((pd.to_numeric(df["Delta_rel_neighbors"], errors="coerce") > 1).sum()),
        "recomputed_delta_n_max": float(pd.to_numeric(df["delta_N_neighbors_recomputed"], errors="coerce").max(skipna=True))
        if not df["delta_N_neighbors_recomputed"].dropna().empty
        else 0.0,
        "recomputed_delta_rel_max": float(pd.to_numeric(df["delta_rel_neighbors_recomputed"], errors="coerce").max(skipna=True))
        if not df["delta_rel_neighbors_recomputed"].dropna().empty
        else 0.0,
        "mismatch_count": int(df["deficit_formula_check"].eq("mismatch_recomputed_used").sum()),
    }
    return df, audit


def summarize_deficit_by_radius(deficits: pd.DataFrame) -> pd.DataFrame:
    """Summarize delta_rel by radius so that 'best' is not overinterpreted."""
    if deficits.empty:
        return pd.DataFrame()
    required = {"node_id", "anchor_pl_name", "radius_type", "Delta_rel_neighbors"}
    if not required.issubset(deficits.columns):
        return pd.DataFrame()
    pivot = deficits.pivot_table(
        index=["node_id", "anchor_pl_name"],
        columns="radius_type",
        values="Delta_rel_neighbors",
        aggfunc="max",
    ).reset_index()
    radius_cols = [column for column in ORDERED_RADII if column in pivot.columns]
    if not radius_cols:
        radius_cols = [c for c in pivot.columns if c not in {"node_id", "anchor_pl_name"}]
    pivot["delta_rel_neighbors_mean"] = pivot[radius_cols].mean(axis=1, skipna=True)
    pivot["delta_rel_neighbors_median"] = pivot[radius_cols].median(axis=1, skipna=True)
    pivot["delta_rel_neighbors_best"] = pivot[radius_cols].max(axis=1, skipna=True).clip(lower=0)
    pivot["best_radius"] = pivot[radius_cols].idxmax(axis=1)
    pivot["deficit_stability_label"] = pivot[radius_cols].apply(lambda row: deficit_stability_label(row.tolist()), axis=1)
    return pivot


def build_top_anchor_radius_tables(
    top_anchors: pd.DataFrame,
    audited_deficits: pd.DataFrame,
    *,
    top_n: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if top_anchors.empty or audited_deficits.empty:
        return pd.DataFrame(), pd.DataFrame()
    selected = top_anchors.head(top_n)[["anchor_pl_name", "node_id", "ATI", "TOI"]].copy()
    merged = audited_deficits.merge(selected, on=["anchor_pl_name", "node_id"], how="inner")
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()
    merged["radius_order"] = merged["radius_type"].map({name: index for index, name in enumerate(ORDERED_RADII)})
    merged = merged.sort_values(["ATI", "anchor_pl_name", "node_id", "radius_order"], ascending=[False, True, True, True])

    detail_columns = [
        "anchor_pl_name",
        "node_id",
        "ATI",
        "TOI",
        "radius_type",
        "radius_value",
        "N_obs",
        "N_exp_neighbors",
        "Delta_N_neighbors",
        "Delta_rel_neighbors",
        "delta_N_neighbors_recomputed",
        "delta_rel_neighbors_recomputed",
        "N_exp_analog",
        "Delta_N_analog",
        "Delta_rel_analog",
        "deficit_class_neighbors",
        "deficit_class_analog",
        "deficit_formula_check",
        "interpretation_short",
    ]
    detail = merged[[column for column in detail_columns if column in merged.columns]].copy()
    detail = detail.rename(columns={"deficit_formula_check": "formula_check"})

    summary_rows = []
    for (anchor_name, node_id), group in merged.groupby(["anchor_pl_name", "node_id"], dropna=False):
        group = group.sort_values("radius_order")
        delta_values = group["delta_rel_neighbors_recomputed"].tolist()
        best_idx = group["delta_rel_neighbors_recomputed"].fillna(-np.inf).idxmax()
        best_row = group.loc[best_idx] if best_idx in group.index else group.iloc[0]
        summary_rows.append({
            "anchor_pl_name": anchor_name,
            "node_id": node_id,
            "ATI": group["ATI"].iloc[0] if "ATI" in group.columns else np.nan,
            "TOI": group["TOI"].iloc[0] if "TOI" in group.columns else np.nan,
            "best_radius_type": best_row.get("radius_type"),
            "N_obs_best": best_row.get("N_obs"),
            "N_exp_neighbors_best": best_row.get("N_exp_neighbors"),
            "Delta_N_neighbors_best": best_row.get("delta_N_neighbors_recomputed"),
            "Delta_rel_neighbors_best": max(0.0, float(best_row.get("delta_rel_neighbors_recomputed", np.nan))) if pd.notna(best_row.get("delta_rel_neighbors_recomputed")) else np.nan,
            "mean_Delta_rel_neighbors": pd.Series(delta_values, dtype=float).mean(skipna=True),
            "median_Delta_rel_neighbors": pd.Series(delta_values, dtype=float).median(skipna=True),
            "max_Delta_rel_neighbors": pd.Series(delta_values, dtype=float).max(skipna=True),
            "deficit_stability_label": deficit_stability_label(delta_values),
            "interpretation_short": _group_interpretation(group["delta_rel_neighbors_recomputed"].tolist()),
        })
    summary = pd.DataFrame(summary_rows).sort_values(["ATI", "Delta_rel_neighbors_best"], ascending=[False, False]).reset_index(drop=True)
    return detail, summary


def _interpretation_short(row: pd.Series) -> str:
    label = row.get("deficit_class_neighbors")
    if label == "consistent_positive_deficit":
        return "Deficit positivo estable en la referencia local."
    if label == "radius_sensitive_deficit":
        return "Deficit positivo pero sensible a la escala local."
    if label == "undefined_reference":
        return "No hay referencia local suficiente para un deficit interpretable."
    if label == "overpopulated_reference":
        return "La referencia local contiene mas vecinos observados que el caso ancla."
    return "Usar como priorizacion observacional prudente, no como conteo absoluto."


def _group_interpretation(values: list[float]) -> str:
    label = deficit_stability_label(values)
    if label == "consistent_positive_deficit":
        return "El deficit aparece en las tres escalas locales y resulta mas estable."
    if label == "radius_sensitive_deficit":
        return "El deficit depende del radio y debe leerse como exploratorio."
    if label == "no_consistent_deficit":
        return "No hay evidencia consistente de deficit local en las tres escalas."
    return "No hay suficientes referencias locales para una lectura estable."
