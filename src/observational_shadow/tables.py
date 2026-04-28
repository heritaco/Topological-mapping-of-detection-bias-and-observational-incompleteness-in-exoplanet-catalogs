from __future__ import annotations

import pandas as pd


def write_csv(frame: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def build_top_candidates(node_metrics: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    cols = [
        "config_id",
        "node_id",
        "n_members",
        "top_method",
        "top_method_fraction",
        "method_entropy_norm",
        "mean_imputation_fraction",
        "method_l1_boundary",
        "physical_neighbor_distance",
        "shadow_score",
        "shadow_class",
        "expected_incompleteness_direction",
        "interpretation_text",
    ]
    available = [column for column in cols if column in node_metrics.columns]
    return (
        node_metrics.sort_values("shadow_score", ascending=False, na_position="last")
        .loc[:, available]
        .head(top_n)
        .reset_index(drop=True)
    )


def build_component_summary(node_metrics: pd.DataFrame) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    rows = []
    for (config_id, component_id), group in node_metrics.groupby(["config_id", "component_id"], dropna=False):
        dominant = group["top_method"].mode(dropna=True)
        high_fraction = (group["shadow_class"] == "high_confidence_shadow").mean()
        rows.append(
            {
                "config_id": config_id,
                "component_id": component_id,
                "n_nodes": int(len(group)),
                "n_members": int(pd.to_numeric(group["n_members"], errors="coerce").sum()),
                "mean_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").mean()),
                "max_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").max()),
                "dominant_method": str(dominant.iloc[0]) if not dominant.empty else "Unknown",
                "mean_imputation_fraction": float(pd.to_numeric(group["mean_imputation_fraction"], errors="coerce").mean()),
                "fraction_high_confidence_shadow": float(high_fraction),
                "interpretation_text": "Componente con regiones candidatas a incompletitud observacional." if high_fraction > 0 else "Componente sin nodos de alta sombra segun cortes heuristicos.",
            }
        )
    return pd.DataFrame(rows).sort_values(["config_id", "max_shadow_score"], ascending=[True, False]).reset_index(drop=True)


def build_method_summary(node_metrics: pd.DataFrame) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    rows = []
    for method, group in node_metrics.groupby("top_method", dropna=False):
        directions = group["expected_incompleteness_direction"].mode(dropna=True)
        rows.append(
            {
                "top_method": method,
                "n_nodes": int(len(group)),
                "mean_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").mean()),
                "median_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").median()),
                "n_high_confidence_shadow": int((group["shadow_class"] == "high_confidence_shadow").sum()),
                "mean_imputation_fraction": float(pd.to_numeric(group["mean_imputation_fraction"], errors="coerce").mean()),
                "most_common_expected_incompleteness_direction": str(directions.iloc[0]) if not directions.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("mean_shadow_score", ascending=False).reset_index(drop=True)


def build_config_comparison(node_metrics: pd.DataFrame) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    rows = []
    for config_id, group in node_metrics.groupby("config_id", dropna=False):
        dominant = group.sort_values("shadow_score", ascending=False)["top_method"].dropna()
        rows.append(
            {
                "config_id": config_id,
                "n_nodes": int(len(group)),
                "mean_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").mean()),
                "median_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").median()),
                "max_shadow_score": float(pd.to_numeric(group["shadow_score"], errors="coerce").max()),
                "n_high_confidence_shadow": int((group["shadow_class"] == "high_confidence_shadow").sum()),
                "mean_imputation_fraction": float(pd.to_numeric(group["mean_imputation_fraction"], errors="coerce").mean()),
                "dominant_shadow_method": str(dominant.iloc[0]) if not dominant.empty else "Unknown",
            }
        )
    return pd.DataFrame(rows).sort_values("max_shadow_score", ascending=False).reset_index(drop=True)

