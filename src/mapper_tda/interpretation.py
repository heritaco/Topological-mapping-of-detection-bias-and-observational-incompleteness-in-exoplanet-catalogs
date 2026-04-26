from __future__ import annotations

import pandas as pd


def generate_interpretation_summary(metrics_df: pd.DataFrame) -> dict[str, str]:
    if metrics_df.empty:
        return {
            "density_sensitivity": "No se generaron metricas suficientes para interpretar sensibilidad a densidad.",
            "imputation_audit": "No se generaron metricas suficientes para auditar imputacion.",
            "lens_sensitivity": "No se generaron metricas suficientes para interpretar sensibilidad al lens.",
            "global_summary": "No se generaron grafos Mapper validos.",
        }

    by_config = metrics_df.set_index("config_id")
    density_text = "No hubo pares comparables con y sin `pl_dens`."
    if {"joint", "joint_no_density"}.issubset(set(metrics_df["space"])) and "pca2" in set(metrics_df["lens"]):
        joint = metrics_df[(metrics_df["space"] == "joint") & (metrics_df["lens"] == "pca2")]
        joint_nd = metrics_df[(metrics_df["space"] == "joint_no_density") & (metrics_df["lens"] == "pca2")]
        if not joint.empty and not joint_nd.empty:
            delta = float(joint.iloc[0]["beta_1"] - joint_nd.iloc[0]["beta_1"])
            if delta > 0:
                density_text = (
                    "Adding the density-derived feature increases the number of independent cycles, "
                    "suggesting that the mass-radius-density relation contributes additional Mapper complexity."
                )
            elif delta < 0:
                density_text = (
                    "Adding `pl_dens` reduces the number of independent cycles, suggesting that the derived density "
                    "feature may simplify or collapse part of the Mapper geometry."
                )
            else:
                density_text = (
                    "Adding `pl_dens` leaves the main cycle count unchanged in the default comparison, so the global "
                    "Mapper complexity is not strongly altered by the derived density feature."
                )

    high_imputation = metrics_df[pd.to_numeric(metrics_df["mean_node_imputation_fraction"], errors="coerce") > 0.30]
    if not high_imputation.empty:
        imputation_text = (
            "This configuration contains a high average node-level imputation fraction; its topology should be "
            "interpreted cautiously."
        )
    else:
        imputation_text = (
            "The average node-level imputation fraction stays below the 0.30 caution threshold in the principal "
            "configurations, which supports a more stable descriptive interpretation."
        )

    lens_text = "No se pudo comparar estabilidad entre lenses."
    pca = metrics_df[metrics_df["lens"] == "pca2"].set_index("space")
    density = metrics_df[metrics_df["lens"] == "density"].set_index("space")
    common = pca.index.intersection(density.index)
    if len(common):
        delta = (pca.loc[common, "n_nodes"] - density.loc[common, "n_nodes"]).abs().mean()
        if delta <= 2:
            lens_text = "The main structure is moderately stable across the PCA and density lenses."
        else:
            lens_text = "The Mapper graph is lens-sensitive, so conclusions should be treated as exploratory."

    largest = metrics_df.sort_values(["beta_1", "n_nodes"], ascending=False).iloc[0]
    global_summary = (
        f"La configuracion con mayor complejidad combinada fue `{largest['config_id']}`, con {int(largest['n_nodes'])} "
        f"nodos y beta_1={int(largest['beta_1'])}. Esto no prueba clases planetarias finales: resume estructura "
        "inducida por una matriz completada y auditada."
    )

    return {
        "density_sensitivity": density_text,
        "imputation_audit": imputation_text,
        "lens_sensitivity": lens_text,
        "global_summary": global_summary,
    }
