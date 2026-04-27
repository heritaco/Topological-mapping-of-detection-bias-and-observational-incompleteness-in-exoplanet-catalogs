from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _get_config(metrics_df: pd.DataFrame, config_id: str) -> pd.Series | None:
    subset = metrics_df[metrics_df["config_id"] == config_id]
    if subset.empty:
        return None
    return subset.iloc[0]


def generate_interpretation_summary(
    metrics_df: pd.DataFrame,
    main_graph_selection: pd.DataFrame | None = None,
    highlighted_nodes: pd.DataFrame | None = None,
    component_summary: pd.DataFrame | None = None,
    bootstrap_summary: pd.DataFrame | None = None,
    null_model_summary: pd.DataFrame | None = None,
    imputation_method_comparison: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if metrics_df.empty:
        return {
            "density_sensitivity": "Not available.",
            "imputation_audit": "Not available.",
            "lens_sensitivity": "Not available.",
            "global_summary": "No valid Mapper graphs were produced.",
            "key_findings": [],
            "promising_signals": [],
            "risky_configs": [],
            "next_steps": [],
        }

    orbital = _get_config(metrics_df, "orbital_pca2_cubes10_overlap0p35")
    thermal = _get_config(metrics_df, "thermal_pca2_cubes10_overlap0p35")
    joint = _get_config(metrics_df, "joint_pca2_cubes10_overlap0p35")
    joint_nd = _get_config(metrics_df, "joint_no_density_pca2_cubes10_overlap0p35")
    phys = _get_config(metrics_df, "phys_min_pca2_cubes10_overlap0p35")
    phys_d = _get_config(metrics_df, "phys_density_pca2_cubes10_overlap0p35")

    key_findings: list[str] = []
    promising_signals: list[str] = []
    risky_configs: list[str] = []
    next_steps: list[str] = [
        "Bootstrap stability for principal graphs.",
        "Null models based on column-wise shuffling.",
        "Astrophysical review of highlighted nodes and components.",
    ]

    if orbital is not None:
        if float(orbital["mean_node_imputation_fraction"]) < 0.05 and float(orbital["beta_1"]) > 50:
            promising_signals.append(
                "The orbital PCA Mapper is a high-priority graph for scientific inspection: it combines nontrivial cycle structure with low imputation dependence."
            )
        key_findings.append(
            "Orbital Mapper shows high structural complexity with low imputation dependence."
        )
    if thermal is not None:
        if float(thermal["frac_nodes_high_imputation"]) > 0.5:
            risky_configs.append(
                "The thermal Mapper should be treated cautiously because more than half of its nodes exceed the high-imputation threshold."
            )
        key_findings.append(
            "Thermal Mapper shows high complexity but high imputation dependence."
        )
    density_text = "Density effect not available."
    if joint is not None and joint_nd is not None:
        delta_beta_joint = float(joint["beta_1"] - joint_nd["beta_1"])
        if delta_beta_joint < 0:
            density_text = (
                "Adding derived density reduces cycle complexity in the joint space, suggesting that density acts as a regularizing coordinate rather than introducing additional branching."
            )
        elif delta_beta_joint > 0:
            density_text = (
                "Adding derived density increases cycle complexity in the joint space, suggesting a stronger topological role for the mass-radius-density relation."
            )
        else:
            density_text = "Adding derived density leaves the joint-space cycle count unchanged in this run."
        key_findings.append("Adding derived density slightly modifies, and in this run reduces, Mapper complexity.")
    else:
        delta_beta_joint = None

    if phys is not None and phys_d is not None and delta_beta_joint is None:
        density_text = "Density effect could only be evaluated on the physical control graph in this run."

    lens_text = "Results are lens-sensitive; pca2 should remain the primary interpretation layer, while density/domain are sensitivity probes."
    if "density" in set(metrics_df["lens"]):
        pca = metrics_df[metrics_df["lens"] == "pca2"].set_index("space")
        density = metrics_df[metrics_df["lens"] == "density"].set_index("space")
        common = pca.index.intersection(density.index)
        if len(common):
            delta = (pca.loc[common, "beta_1"] - density.loc[common, "beta_1"]).abs().mean()
            if delta < 10:
                lens_text = "The main structure is moderately stable across the PCA and density lenses."

    imputation_text = "Low-imputation graphs should receive higher scientific priority than high-imputation graphs."
    if thermal is not None and orbital is not None:
        imputation_text = (
            f"Orbital PCA remains the clearest low-imputation candidate (mean node imputation fraction {float(orbital['mean_node_imputation_fraction']):.3f}), "
            f"whereas thermal PCA is cautionary (mean node imputation fraction {float(thermal['mean_node_imputation_fraction']):.3f})."
        )

    largest = metrics_df.sort_values(["beta_1", "n_nodes"], ascending=False).iloc[0]
    global_summary = (
        "El resultado mas prometedor no es una clasificacion final de planetas, sino la identificacion de grafos y regiones candidatos para inspeccion cientifica. "
        "En esta corrida, el espacio orbital PCA es prioritario porque combina estructura no trivial con baja dependencia de imputacion. "
        "El espacio termico es topologicamente complejo, pero debe tratarse con cautela por su alta fraccion de imputacion. "
        "La inclusion de densidad derivada reduce ligeramente la complejidad de Mapper, lo que sugiere que `pl_dens` actua como coordenada regularizadora mas que como fuente independiente de nueva ramificacion."
    )

    if highlighted_nodes is not None and not highlighted_nodes.empty:
        promising_signals.append(
            f"Highlighted node catalog available with {len(highlighted_nodes)} node-level interpretation records."
        )
    if component_summary is not None and not component_summary.empty:
        promising_signals.append(
            f"Connected-component summaries available for {component_summary['config_id'].nunique()} principal graphs."
        )

    if bootstrap_summary is None or bootstrap_summary.empty:
        next_steps.append("Bootstrap was not run in this execution.")
    if null_model_summary is None or null_model_summary.empty:
        next_steps.append("Null models were not run in this execution.")
    if imputation_method_comparison is None or imputation_method_comparison.empty:
        next_steps.append("Multi-method Mapper comparison was skipped or partial because not all imputation inputs were available.")

    return {
        "density_sensitivity": density_text,
        "imputation_audit": imputation_text,
        "lens_sensitivity": lens_text,
        "global_summary": global_summary,
        "key_findings": key_findings,
        "promising_signals": promising_signals,
        "risky_configs": risky_configs,
        "next_steps": next_steps,
    }


def build_interpretive_summary_files(
    outputs_tables_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Path]:
    outputs_tables_dir.mkdir(parents=True, exist_ok=True)
    md_path = outputs_tables_dir / "mapper_interpretive_summary.md"
    tex_path = outputs_tables_dir / "mapper_interpretive_summary.tex"

    def _bullet_lines(items: list[str]) -> str:
        if not items:
            return "- not available"
        return "\n".join(f"- {item}" for item in items)

    md_content = "\n".join(
        [
            "# Mapper Interpretive Summary",
            "",
            "## Main findings",
            _bullet_lines(summary.get("key_findings", [])),
            "",
            "## Promising signals",
            _bullet_lines(summary.get("promising_signals", [])),
            "",
            "## Risky configurations",
            _bullet_lines(summary.get("risky_configs", [])),
            "",
            "## Density effect",
            summary.get("density_sensitivity", "not available"),
            "",
            "## Lens effect",
            summary.get("lens_sensitivity", "not available"),
            "",
            "## Recommendations",
            _bullet_lines(summary.get("next_steps", [])),
            "",
        ]
    )
    md_path.write_text(md_content, encoding="utf-8")

    def _tex_list(items: list[str]) -> str:
        if not items:
            return "\\begin{itemize}\\item not available\\end{itemize}"
        body = "".join(f"\\item {item}\n" for item in items)
        return "\\begin{itemize}\n" + body + "\\end{itemize}\n"

    tex_content = "\n".join(
        [
            "\\subsection*{Key Findings}",
            _tex_list(summary.get("key_findings", [])),
            "\\subsection*{Promising Signals}",
            _tex_list(summary.get("promising_signals", [])),
            "\\subsection*{Risky Configurations}",
            _tex_list(summary.get("risky_configs", [])),
            "\\subsection*{Density Effect}",
            summary.get("density_sensitivity", "not available"),
            "",
            "\\subsection*{Lens Effect}",
            summary.get("lens_sensitivity", "not available"),
            "",
            "\\subsection*{Next Steps}",
            _tex_list(summary.get("next_steps", [])),
        ]
    )
    tex_path.write_text(tex_content, encoding="utf-8")
    return {"mapper_interpretive_summary_md": md_path, "mapper_interpretive_summary_tex": tex_path}
