from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_tex_table(df: pd.DataFrame, path: Path, caption: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        body = "\\begin{table}[ht]\n\\centering\n\\caption{" + caption + "}\nSin filas disponibles.\n\\end{table}\n"
        path.write_text(body, encoding="utf-8")
        return
    tex = df.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
    content = "\n".join(
        [
            "\\begin{table}[ht]",
            "\\centering",
            f"\\caption{{{caption}}}",
            tex,
            "\\end{table}",
            "",
        ]
    )
    path.write_text(content, encoding="utf-8")


def build_summary_global_bias_metrics(
    global_rows: pd.DataFrame,
    permutation_tests: pd.DataFrame,
) -> pd.DataFrame:
    if global_rows.empty:
        return pd.DataFrame()
    summary = global_rows.copy()
    if permutation_tests.empty:
        summary["global_nmi_z_score"] = np.nan
        summary["global_nmi_empirical_p_value"] = np.nan
        summary["global_purity_z_score"] = np.nan
        return summary
    nmi_tests = permutation_tests[permutation_tests["metric"] == "node_method_nmi"][
        ["config_id", "z_score", "empirical_p_value"]
    ].rename(columns={"z_score": "global_nmi_z_score", "empirical_p_value": "global_nmi_empirical_p_value"})
    purity_tests = permutation_tests[permutation_tests["metric"] == "weighted_mean_purity"][
        ["config_id", "z_score"]
    ].rename(columns={"z_score": "global_purity_z_score"})
    summary = summary.merge(nmi_tests, on="config_id", how="left")
    summary = summary.merge(purity_tests, on="config_id", how="left")
    return summary


def build_top_enriched_nodes(
    node_metrics: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    top_n: int = 25,
) -> pd.DataFrame:
    if node_metrics.empty or enrichment_df.empty:
        return pd.DataFrame()
    best_per_node = (
        enrichment_df.sort_values(["fdr_q_value", "z_score"], ascending=[True, False])
        .drop_duplicates(subset=["config_id", "node_id"])
        .rename(columns={"method": "enriched_method"})
    )
    merged = best_per_node.merge(
        node_metrics[
            [
                "config_id",
                "node_id",
                "n_members",
                "top_method",
                "top_method_fraction",
                "mean_imputation_fraction",
                "is_peripheral",
            ]
        ],
        on=["config_id", "node_id"],
        how="left",
    )
    columns = [
        "config_id",
        "node_id",
        "n_members",
        "enriched_method",
        "top_method",
        "top_method_fraction",
        "z_score",
        "empirical_p_value",
        "fdr_q_value",
        "mean_imputation_fraction",
        "is_peripheral",
    ]
    return merged.loc[:, columns].head(top_n).reset_index(drop=True)


def build_peripheral_bias_nodes(
    node_metrics: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    purity_threshold: float = 0.75,
) -> pd.DataFrame:
    if node_metrics.empty:
        return pd.DataFrame()
    best = (
        enrichment_df.sort_values(["fdr_q_value", "z_score"], ascending=[True, False])
        .drop_duplicates(subset=["config_id", "node_id"])
        .rename(columns={"method": "enriched_method"})
    )
    merged = node_metrics.merge(best[["config_id", "node_id", "enriched_method", "z_score", "fdr_q_value"]], on=["config_id", "node_id"], how="left")
    filtered = merged[
        merged["is_peripheral"].fillna(False)
        & (pd.to_numeric(merged["top_method_fraction"], errors="coerce") >= purity_threshold)
    ].copy()
    return filtered[
        [
            "config_id",
            "node_id",
            "n_members",
            "top_method",
            "top_method_fraction",
            "enriched_method",
            "z_score",
            "fdr_q_value",
            "mean_imputation_fraction",
            "component_id",
            "degree",
        ]
    ].sort_values(["config_id", "top_method_fraction", "n_members"], ascending=[True, False, False]).reset_index(drop=True)
