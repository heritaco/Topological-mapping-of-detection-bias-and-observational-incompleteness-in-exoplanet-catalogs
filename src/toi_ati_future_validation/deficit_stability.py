from __future__ import annotations

import numpy as np
import pandas as pd

DEFICIT_COLUMN_ALIASES = {
    "anchor_pl_name": ["anchor_pl_name", "pl_name", "planet_name", "anchor_name"],
    "node_id": ["node_id", "node", "mapper_node_id"],
    "radius_type": ["radius_type", "radius", "radius_name"],
    "radius_value": ["radius_value", "radius_size", "radius_numeric"],
    "N_obs": ["N_obs", "n_obs", "observed_neighbors"],
    "N_exp_neighbors": ["N_exp_neighbors", "n_exp_neighbors", "expected_neighbors"],
    "delta_N_neighbors": ["delta_N_neighbors", "Delta_N_neighbors", "delta_n_neighbors"],
    "delta_rel_neighbors": ["delta_rel_neighbors", "Delta_rel_neighbors", "relative_deficit_neighbors"],
}


def normalize_deficit_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    df = frame.copy()
    rename_map: dict[str, str] = {}
    for canonical, aliases in DEFICIT_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in df.columns:
                rename_map[alias] = canonical
                break
    return df.rename(columns=rename_map)


def audit_and_recompute_deficits(frame: pd.DataFrame, *, epsilon: float = 1e-9) -> tuple[pd.DataFrame, dict[str, object]]:
    df = normalize_deficit_columns(frame)
    required = ["anchor_pl_name", "node_id", "radius_type", "N_obs", "N_exp_neighbors"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Cannot compute deficit stability because required columns are missing: {', '.join(missing)}")
    for column in ["radius_value", "delta_N_neighbors", "delta_rel_neighbors"]:
        if column not in df.columns:
            df[column] = np.nan

    df["delta_N_neighbors_recomputed"] = pd.to_numeric(df["N_exp_neighbors"], errors="coerce") - pd.to_numeric(df["N_obs"], errors="coerce")
    df["delta_rel_neighbors_recomputed"] = df["delta_N_neighbors_recomputed"] / (pd.to_numeric(df["N_exp_neighbors"], errors="coerce") + float(epsilon))

    def _formula_check(row: pd.Series) -> str:
        if pd.isna(row.get("N_obs")) or pd.isna(row.get("N_exp_neighbors")):
            return "missing_expected_or_observed"
        original_dn = pd.to_numeric(pd.Series([row.get("delta_N_neighbors")]), errors="coerce").iloc[0]
        original_dr = pd.to_numeric(pd.Series([row.get("delta_rel_neighbors")]), errors="coerce").iloc[0]
        if pd.isna(original_dn) or pd.isna(original_dr):
            return "mismatch_recomputed_used"
        if abs(float(original_dn) - float(row["delta_N_neighbors_recomputed"])) <= 1e-6 and abs(float(original_dr) - float(row["delta_rel_neighbors_recomputed"])) <= 1e-6:
            return "ok"
        return "mismatch_recomputed_used"

    df["deficit_formula_check"] = df.apply(_formula_check, axis=1)
    audit = {
        "raw_delta_rel_max": float(pd.to_numeric(df["delta_rel_neighbors"], errors="coerce").max(skipna=True))
        if not df["delta_rel_neighbors"].dropna().empty
        else None,
        "recomputed_delta_rel_max": float(pd.to_numeric(df["delta_rel_neighbors_recomputed"], errors="coerce").max(skipna=True))
        if not df["delta_rel_neighbors_recomputed"].dropna().empty
        else None,
        "recomputed_delta_n_max": float(pd.to_numeric(df["delta_N_neighbors_recomputed"], errors="coerce").max(skipna=True))
        if not df["delta_N_neighbors_recomputed"].dropna().empty
        else None,
        "mismatch_count": int(df["deficit_formula_check"].eq("mismatch_recomputed_used").sum()),
    }
    return df, audit


def classify_deficit_profile(
    delta_rel_kNN: float | None,
    delta_rel_node_median: float | None,
    delta_rel_node_q75: float | None,
) -> str:
    values = [value for value in [delta_rel_kNN, delta_rel_node_median, delta_rel_node_q75] if pd.notna(value)]
    if not values:
        return "no_deficit_or_overpopulated"
    positives = sum(value > 0 for value in values)
    all_positive = positives == len(values)
    if all_positive and max(values) <= 0.15:
        return "small_but_stable_deficit"
    if all_positive:
        return "stable_positive_deficit"
    if positives == 0:
        return "no_deficit_or_overpopulated"
    if pd.notna(delta_rel_kNN) and delta_rel_kNN > 0 and ((pd.notna(delta_rel_node_median) and delta_rel_node_median < 0) or (pd.notna(delta_rel_node_q75) and delta_rel_node_q75 < 0)):
        return "unstable_due_to_large_radius"
    return "radius_sensitive_deficit"


def compute_deficit_stability(
    deficits: pd.DataFrame,
    anchors: pd.DataFrame,
    regions: pd.DataFrame,
    *,
    radii: list[str],
) -> pd.DataFrame:
    if deficits.empty:
        return pd.DataFrame()
    pivot = deficits.pivot_table(
        index=["anchor_pl_name", "node_id"],
        columns="radius_type",
        values="delta_rel_neighbors_recomputed",
        aggfunc="first",
    ).reset_index()
    for radius in radii:
        if radius not in pivot.columns:
            pivot[radius] = np.nan
    pivot = pivot.rename(
        columns={
            "r_kNN": "delta_rel_kNN",
            "r_node_median": "delta_rel_node_median",
            "r_node_q75": "delta_rel_node_q75",
        }
    )
    delta_cols = ["delta_rel_kNN", "delta_rel_node_median", "delta_rel_node_q75"]
    pivot["delta_rel_mean"] = pivot[delta_cols].mean(axis=1, skipna=True)
    pivot["delta_rel_median"] = pivot[delta_cols].median(axis=1, skipna=True)
    pivot["delta_rel_min"] = pivot[delta_cols].min(axis=1, skipna=True)
    pivot["delta_rel_max"] = pivot[delta_cols].max(axis=1, skipna=True)
    pivot["delta_rel_std"] = pivot[delta_cols].std(axis=1, skipna=True, ddof=0)
    pivot["n_positive_radii"] = (pivot[delta_cols] > 0).sum(axis=1)
    pivot["n_negative_radii"] = (pivot[delta_cols] < 0).sum(axis=1)
    pivot["all_radii_positive"] = pivot["n_positive_radii"].eq(len(delta_cols))
    pivot["any_large_radius_negative"] = (pivot["delta_rel_node_median"] < 0) | (pivot["delta_rel_node_q75"] < 0)
    pivot["radius_sensitivity_score"] = pivot["delta_rel_max"] - pivot["delta_rel_min"]
    pivot["stable_deficit_score"] = pivot[delta_cols].clip(lower=0).mean(axis=1, skipna=True) * (pivot["n_positive_radii"] / len(delta_cols))
    pivot["deficit_stability_class"] = pivot.apply(
        lambda row: classify_deficit_profile(
            row.get("delta_rel_kNN"),
            row.get("delta_rel_node_median"),
            row.get("delta_rel_node_q75"),
        ),
        axis=1,
    )
    pivot["interpretation_text"] = pivot.apply(_stability_interpretation, axis=1)

    join_columns = [
        column
        for column in [
            "anchor_pl_name",
            "node_id",
            "ATI",
            "TOI",
            "r3_imputation_score",
            "anchor_representativeness",
            "deficit_class",
            "expected_incompleteness_direction",
            "discoverymethod",
            "disc_year",
            "disc_facility",
        ]
        if column in anchors.columns
    ]
    out = pivot.merge(anchors[join_columns].drop_duplicates(subset=["anchor_pl_name", "node_id"]), on=["anchor_pl_name", "node_id"], how="left")
    region_columns = [column for column in ["node_id", "n_members", "shadow_score", "I_R3", "C_phys", "S_net", "top_method"] if column in regions.columns]
    out = out.merge(regions[region_columns].drop_duplicates(subset=["node_id"]), on="node_id", how="left")
    ordered = [
        "anchor_pl_name",
        "node_id",
        "ATI",
        "TOI",
        "delta_rel_kNN",
        "delta_rel_node_median",
        "delta_rel_node_q75",
        "delta_rel_mean",
        "delta_rel_median",
        "delta_rel_min",
        "delta_rel_max",
        "delta_rel_std",
        "n_positive_radii",
        "n_negative_radii",
        "all_radii_positive",
        "any_large_radius_negative",
        "radius_sensitivity_score",
        "stable_deficit_score",
        "deficit_stability_class",
        "interpretation_text",
    ]
    trailing = [column for column in out.columns if column not in ordered]
    return out[ordered + trailing]


def _stability_interpretation(row: pd.Series) -> str:
    label = str(row.get("deficit_stability_class", "unknown"))
    if label == "stable_positive_deficit":
        return "El deficit topologico local permanece positivo en los tres radios y ofrece una lectura mas estable."
    if label == "small_but_stable_deficit":
        return "El deficit es pequeno pero consistente en las tres escalas locales, por lo que funciona como caso estable y prudente."
    if label == "radius_sensitive_deficit":
        return "El deficit cambia con el radio local y debe leerse como caso exploratorio sensible a escala."
    if label == "unstable_due_to_large_radius":
        return "El deficit aparece en kNN pero se revierte en radios mayores, lo que sugiere fragilidad bajo escalas mas amplias."
    return "No hay evidencia consistente de deficit local positivo bajo la referencia topologica usada."

