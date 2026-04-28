from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .anchor_selection import select_anchor
from .case_selection import select_case_nodes
from .config import (
    load_local_case_config,
    resolved_audit_outputs_dir,
    resolved_latex_dir,
    resolved_local_outputs_dir,
    resolved_mapper_outputs_dir,
    resolved_shadow_outputs_dir,
)
from .graph_context import build_graph, case_neighborhood
from .imputation_audit import add_variable_status_columns, summarize_r3_imputation
from .io import (
    discover_available_config_ids,
    git_commit_hash,
    load_required_case_inputs,
    manifest_input_paths,
    save_log,
    write_json,
    write_text,
)
from .latex import write_latex_report
from .method_contrast import summarize_method_context
from .neighbor_deficit import compute_neighbor_deficits
from .paths import ensure_latex_dir, ensure_output_tree, repo_relative
from .plotting import (
    placeholder_figure,
    plot_case_anchor_deficit,
    plot_case_ego_network,
    plot_case_imputation_audit,
    plot_case_method_composition,
    plot_case_r3_projections,
    plot_case_rv_proxy_distribution,
    plot_three_case_confidence_matrix,
    plot_three_case_deficit_comparison,
    plot_three_case_shadow_vs_physical_distance,
)
from .r3_geometry import add_r3_coordinates, apply_r3_standardization, build_region_membership, compute_global_r3_stats, describe_case_geometry
from .tables import confidence_level, final_interpretation, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estudios locales de sombra observacional para comunidades RV del Mapper orbital.")
    parser.add_argument("--config", default="configs/local_shadow_case_studies.yaml", help="Ruta YAML/JSON de configuracion.")
    parser.add_argument("--seed", type=int, default=None, help="Override de semilla para layouts.")
    return parser.parse_args()


def _rv_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rv_proxy"] = pd.to_numeric(out["r3_log_mass"], errors="coerce") - (1.0 / 3.0) * pd.to_numeric(out["r3_log_period"], errors="coerce")
    st_mass = pd.to_numeric(out.get("st_mass"), errors="coerce")
    out["rv_proxy_with_star"] = np.nan
    valid_star = st_mass.notna() & np.isfinite(st_mass) & (st_mass > 0)
    out.loc[valid_star, "rv_proxy_with_star"] = (
        pd.to_numeric(out.loc[valid_star, "r3_log_mass"], errors="coerce")
        - (1.0 / 3.0) * pd.to_numeric(out.loc[valid_star, "r3_log_period"], errors="coerce")
        - (2.0 / 3.0) * np.log10(st_mass.loc[valid_star].astype(float))
    )
    return out


def _unique_by_row_index(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "row_index" not in frame.columns:
        return frame.copy()
    return frame.sort_values(["row_index", "node_id"]).drop_duplicates(subset=["row_index"], keep="first").copy()


def _case_readme() -> str:
    return """# Local Shadow Case Studies

## Pregunta
Este subproyecto pregunta si una comunidad Mapper de alta sombra observacional puede leerse como una ficha local de incompletitud topologica: un nodo, su vecindario, sus miembros en R3 y un exoplaneta ancla.

## Por que comunidades RV
Los reportes previos mostraron que varios candidatos de alta sombra en el Mapper orbital estan dominados por `Radial Velocity`. Eso hace plausible una lectura prudente de incompletitud hacia menor masa planetaria o menor proxy de detectabilidad RV.

## Que es R3
El espacio principal del caso local es:
`R3 = (log10(pl_bmasse), log10(pl_orbper), log10(pl_orbsmax))`

## Como se selecciona el planeta ancla
El ancla debe tener R3 valido, se priorizan miembros `Radial Velocity`, baja imputacion en masa/periodo/semieje, mejor trazabilidad observacional y cercania al medoid del nodo.

## Como se calcula el deficit local
Para cada ancla se compara el conteo observado de vecinos compatibles dentro de radios locales contra referencias prudentes basadas en vecinos topologicos y nodos analogos de menor sombra. Esto produce un `deficit topologico local`, no un conteo absoluto de planetas reales faltantes.

## Como correr
`python -m src.local_shadow_case_studies.run_local_shadow_cases --config configs/local_shadow_case_studies.yaml`

## Donde quedan los resultados
- Tablas CSV: `outputs/local_shadow_case_studies/tables/`
- Figuras PDF: `outputs/local_shadow_case_studies/figures/`
- Metadata y manifest: `outputs/local_shadow_case_studies/metadata/` y `outputs/local_shadow_case_studies/run_manifest.json`
- Log: `outputs/local_shadow_case_studies/logs/`
- Reporte LaTeX: `latex/local_shadow_case_studies/main.tex`

## Como interpretar
Leer cada caso como un candidato a incompletitud observacional o region fisico-orbital submuestreada. El lenguaje correcto es `deficit topologico local`, `vecinos compatibles esperados bajo referencia local` y `posible incompletitud hacia menor masa o menor senal RV`.

## Advertencias
- No afirmar `planetas faltantes confirmados`.
- No afirmar `faltan exactamente N exoplanetas reales`.
- El resultado depende del Mapper usado, de la referencia local y de la trazabilidad de imputacion.
"""


def _interpretation_summary(case_comparison: pd.DataFrame) -> str:
    lines = [
        "# Interpretation Summary",
        "",
        "## Resumen ejecutivo",
        "Tres comunidades RV del Mapper orbital se analizaron como fichas locales de posible incompletitud observacional. Cada caso se resume con un planeta ancla, su vecindario topologico y un deficit relativo de vecinos compatibles bajo referencias locales.",
        "",
        "## Tabla breve",
    ]
    if case_comparison.empty:
        lines.append("No se generaron casos comparables.")
    else:
        lines.append("| case_id | node_id | anchor | delta_rel_neighbors_best | confidence_level |")
        lines.append("| --- | --- | --- | --- | --- |")
        for _, row in case_comparison.iterrows():
            lines.append(
                f"| {row['case_id']} | {row['node_id']} | {row['anchor_pl_name']} | "
                f"{row.get('delta_rel_neighbors_best')} | {row.get('confidence_level')} |"
            )
    lines.extend(
        [
            "",
            "## Ranking de confianza",
        ]
    )
    if not case_comparison.empty:
        ranked = case_comparison.sort_values(by=["confidence_level", "delta_rel_neighbors_best"], ascending=[True, False])
        for _, row in ranked.iterrows():
            lines.append(
                f"- {row['node_id']}: anchor={row['anchor_pl_name']}, deficit={row.get('delta_rel_neighbors_best')}, "
                f"direccion={row.get('expected_missing_direction')}, confianza={row.get('confidence_level')}."
            )
    lines.extend(
        [
            "",
            "## Advertencia",
            "Estos resultados no equivalen a confirmaciones de objetos ausentes. Son indicadores locales y modelo-dependientes de deficit topologico de vecinos compatibles.",
            "",
            "Frase lista para reporte: Un cluster RV de alta sombra puede interpretarse como una vecindad fisica plausible vista a traves de una ventana observacional incompleta; el planeta ancla no prueba planetas faltantes, pero si localiza un deficit topologico local de vecinos compatibles.",
        ]
    )
    return "\n".join(lines) + "\n"


def _automatic_interpretation(node_id: str, top_method: str, shadow_score: float, ir3: float | None, deficit_class: str, direction: str) -> str:
    ir3_text = "NA" if ir3 is None else f"{ir3:.3f}"
    return (
        f"Nodo {node_id} dominado por {top_method}; candidato a incompletitud observacional con shadow_score={shadow_score:.3f}, "
        f"I_R3={ir3_text} y lectura prudente de deficit topologico local {deficit_class} hacia {direction}. "
        "No implica una prediccion definitiva sobre objetos reales no observados."
    )


def _assert_prudent_text(text: str) -> None:
    banned = ["planetas faltantes confirmados", "descubrimos planetas", "faltan exactamente"]
    lower = text.lower()
    for phrase in banned:
        if phrase in lower:
            raise AssertionError(f"Se detecto una frase prohibida en la interpretacion: {phrase}")


def main() -> None:
    args = parse_args()
    config = load_local_case_config(path=args.config, overrides={"seed": args.seed})
    mapper_outputs_dir = resolved_mapper_outputs_dir(config)
    audit_outputs_dir = resolved_audit_outputs_dir(config)
    shadow_outputs_dir = resolved_shadow_outputs_dir(config)
    local_outputs_dir = resolved_local_outputs_dir(config)
    latex_dir = ensure_latex_dir(resolved_latex_dir(config))
    output_tree = ensure_output_tree(local_outputs_dir)
    warnings: list[str] = []

    available_config_ids = discover_available_config_ids(mapper_outputs_dir)
    if config.primary_config_id not in available_config_ids:
        raise RuntimeError(f"No existe la configuracion primaria {config.primary_config_id} en outputs/mapper.")

    inputs = load_required_case_inputs(
        mapper_outputs_dir=mapper_outputs_dir,
        audit_outputs_dir=audit_outputs_dir,
        shadow_outputs_dir=shadow_outputs_dir,
        config_id=config.primary_config_id,
        physical_csv_path=config.physical_csv_path,
        local_metadata_dir=output_tree["metadata"],
        warnings=warnings,
    )

    selected_nodes, replacements = select_case_nodes(
        shadow_metrics=inputs["shadow_metrics"],
        top_candidates=inputs["top_candidates"],
        config_id=config.primary_config_id,
        requested_node_ids=config.requested_node_ids,
        required_method=config.top_method_required,
        n_cases=config.n_cases,
        min_members=config.min_case_members,
        warnings=warnings,
    )

    membership = add_r3_coordinates(inputs["membership"], warnings, label=config.primary_config_id)
    membership = apply_r3_standardization(membership, compute_global_r3_stats(membership))
    membership = add_variable_status_columns(membership, config.r3_variables)
    membership = _rv_proxy(membership)
    graph = build_graph(inputs["edge_table"], inputs["node_table"])

    centroid_map: dict[str, np.ndarray] = {}
    node_valid_summary_rows: list[dict[str, object]] = []
    for node_name, group in membership.groupby(membership["node_id"].astype(str)):
        valid = group[group["r3_valid"]].copy()
        if not valid.empty:
            centroid_map[str(node_name)] = valid[["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].mean().to_numpy(dtype=float)
        node_valid_summary_rows.append(
            {
                "node_id": str(node_name),
                "n_members": int(group["row_index"].nunique()),
                "r3_valid_fraction": float(group["r3_valid"].mean()) if len(group) else 0.0,
            }
        )
    node_valid_summary = pd.DataFrame(node_valid_summary_rows)
    node_summary_frame = inputs["shadow_metrics"][inputs["shadow_metrics"]["config_id"].astype(str) == config.primary_config_id].merge(
        node_valid_summary, on="node_id", how="left", suffixes=("", "_r3")
    )

    case_node_rows: list[dict[str, object]] = []
    case_anchor_rows: list[dict[str, object]] = []
    case_method_rows: list[pd.DataFrame] = []
    case_member_rows: list[pd.DataFrame] = []
    case_deficit_rows: list[pd.DataFrame] = []
    case_contexts: dict[str, object] = {}

    for case_index, node_id in enumerate(selected_nodes, start=1):
        case_id = f"case_{case_index}"
        node_frame = _unique_by_row_index(membership[membership["node_id"].astype(str) == str(node_id)].copy())
        context = case_neighborhood(node_id, graph, inputs["node_table"], membership)
        case_contexts[case_id] = context
        n1_frame = _unique_by_row_index(membership[membership["node_id"].astype(str).isin(context.n1_nodes)].copy())
        n2_frame = _unique_by_row_index(membership[membership["node_id"].astype(str).isin(context.n2_nodes)].copy())
        node_shadow = node_summary_frame[node_summary_frame["node_id"].astype(str) == str(node_id)].copy()
        if node_shadow.empty:
            warnings.append(f"WARNING: {node_id} no tiene fila en node_observational_shadow_metrics.")
            continue
        shadow_row = node_shadow.iloc[0]

        method_summary, method_composition = summarize_method_context(case_id, node_id, node_frame, n1_frame, n2_frame, config.epsilon)
        geometry_summary = describe_case_geometry(node_frame, n1_frame, n2_frame)
        imputation_summary = summarize_r3_imputation(node_frame, config.r3_variables)

        anchor, anchor_reason = select_anchor(node_frame, config.r3_variables)
        if anchor is None:
            warnings.append(f"WARNING: {node_id} no produjo planeta ancla por falta de R3 valido.")
            deficit_df = pd.DataFrame()
            deficit_summary = {"delta_rel_neighbors_best": None, "deficit_class": "not_available", "n_analog_nodes": 0}
            anchor_name = "Unknown"
        else:
            deficit_df, deficit_summary = compute_neighbor_deficits(
                case_id=case_id,
                node_id=node_id,
                anchor=anchor,
                node_frame=node_frame,
                n1_frame=n1_frame,
                n2_frame=n2_frame,
                membership_all=membership,
                graph=graph,
                node_summary_frame=node_summary_frame,
                centroid_map=centroid_map,
                analog_tau=config.analog_centroid_distance_tau,
                analog_shadow_quantile=config.analog_shadow_quantile,
                analog_min_nodes=config.analog_min_nodes,
                analog_min_members=config.analog_min_members,
                analog_min_valid_fraction=config.analog_min_r3_valid_fraction,
                neighbor_reference_min_size=config.neighbor_reference_min_size,
                analog_count_cap=config.analog_count_cap,
                epsilon=config.epsilon,
                warnings=warnings,
            )
            anchor_name = str(anchor.get("pl_name", "Unknown"))

        expected_direction = (
            str(deficit_df["expected_missing_direction"].dropna().iloc[0])
            if not deficit_df.empty and deficit_df["expected_missing_direction"].dropna().any()
            else "menor masa planetaria a periodo/semieje comparable"
        )
        interpretation_text = _automatic_interpretation(
            node_id=node_id,
            top_method=str(method_summary["top_method"]),
            shadow_score=float(pd.to_numeric(pd.Series([shadow_row.get("shadow_score")]), errors="coerce").fillna(0.0).iloc[0]),
            ir3=imputation_summary.get("I_R3"),
            deficit_class=str(deficit_summary["deficit_class"]),
            direction=expected_direction,
        )
        _assert_prudent_text(interpretation_text)

        node_row = {
            "case_id": case_id,
            "config_id": config.primary_config_id,
            "node_id": node_id,
            "n_members": int(node_frame["row_index"].nunique()),
            "degree": context.graph_metrics["degree"],
            "component_id": shadow_row.get("component_id"),
            "component_size_nodes": context.graph_metrics["component_size_nodes"],
            "component_size_members": context.graph_metrics["component_size_members"],
            "is_peripheral": shadow_row.get("is_peripheral"),
            "betweenness_centrality": context.graph_metrics["betweenness_centrality"],
            "closeness_centrality": context.graph_metrics["closeness_centrality"],
            "clustering_coefficient": context.graph_metrics["clustering_coefficient"],
            "distance_to_largest_node": context.graph_metrics["distance_to_largest_node"],
            "distance_to_component_core": context.graph_metrics["distance_to_component_core"],
            "is_articulation_point": context.graph_metrics["is_articulation_point"],
            "eccentricity": context.graph_metrics["eccentricity"],
            "shadow_score": shadow_row.get("shadow_score"),
            "shadow_class": shadow_row.get("shadow_class"),
            "top_method": method_summary["top_method"],
            "top_method_fraction": method_summary["top_method_fraction"],
            "method_entropy_norm": method_summary["method_entropy_norm"],
            "method_l1_boundary_N1": method_summary["method_l1_boundary_N1"],
            "method_l1_boundary_N2": method_summary["method_l1_boundary_N2"],
            "physical_distance_v_to_N1": geometry_summary["physical_distance_v_to_N1"],
            "physical_distance_v_to_N2": geometry_summary["physical_distance_v_to_N2"],
            "mean_imputation_fraction": shadow_row.get("mean_imputation_fraction"),
            "mean_physically_derived_fraction": shadow_row.get("mean_physically_derived_fraction"),
            "n_r3_valid": geometry_summary["n_r3_valid"],
            "r3_valid_fraction": geometry_summary["r3_valid_fraction"],
            "I_R3": imputation_summary.get("I_R3"),
            "imputed_fraction_pl_bmasse": imputation_summary.get("imputed_fraction_pl_bmasse"),
            "imputed_fraction_pl_orbper": imputation_summary.get("imputed_fraction_pl_orbper"),
            "imputed_fraction_pl_orbsmax": imputation_summary.get("imputed_fraction_pl_orbsmax"),
            "physically_derived_fraction_pl_bmasse": imputation_summary.get("physically_derived_fraction_pl_bmasse"),
            "physically_derived_fraction_pl_orbper": imputation_summary.get("physically_derived_fraction_pl_orbper"),
            "physically_derived_fraction_pl_orbsmax": imputation_summary.get("physically_derived_fraction_pl_orbsmax"),
            "observed_fraction_pl_bmasse": imputation_summary.get("observed_fraction_pl_bmasse"),
            "observed_fraction_pl_orbper": imputation_summary.get("observed_fraction_pl_orbper"),
            "observed_fraction_pl_orbsmax": imputation_summary.get("observed_fraction_pl_orbsmax"),
            "neighbor_overlap_score": geometry_summary["neighbor_overlap_score"],
            "spread_r3": geometry_summary["spread_r3"],
            "interpretation_text": interpretation_text,
        }
        case_node_rows.append(node_row)

        if anchor is not None:
            node_center = np.array(geometry_summary["centroid_v_r3"]) if geometry_summary["centroid_v_r3"] is not None else None
            n1_center = np.array(geometry_summary["centroid_N1_r3"]) if geometry_summary["centroid_N1_r3"] is not None else None
            anchor_point = anchor[["r3_z_mass", "r3_z_period", "r3_z_semimajor"]].to_numpy(dtype=float)
            distance_to_node = float(np.linalg.norm(anchor_point - node_center)) if node_center is not None else None
            distance_to_neighbor = float(np.linalg.norm(anchor_point - n1_center)) if n1_center is not None else None
            case_anchor_rows.append(
                {
                    "case_id": case_id,
                    "node_id": node_id,
                    "anchor_pl_name": anchor.get("pl_name"),
                    "discoverymethod": anchor.get("discoverymethod"),
                    "disc_year": anchor.get("disc_year"),
                    "disc_facility": anchor.get("disc_facility"),
                    "pl_bmasse": anchor.get("pl_bmasse"),
                    "pl_orbper": anchor.get("pl_orbper"),
                    "pl_orbsmax": anchor.get("pl_orbsmax"),
                    "r3_log_mass": anchor.get("r3_log_mass"),
                    "r3_log_period": anchor.get("r3_log_period"),
                    "r3_log_semimajor": anchor.get("r3_log_semimajor"),
                    "anchor_rv_proxy": anchor.get("rv_proxy"),
                    "anchor_r3_imputation_score": anchor.get("anchor_r3_imputation_score"),
                    "anchor_distance_to_node_centroid": distance_to_node,
                    "anchor_distance_to_neighbor_centroid": distance_to_neighbor,
                    "reason_for_anchor_selection": anchor_reason,
                }
            )

        case_method_rows.append(method_composition)

        region_members = build_region_membership(case_id, node_id, [node_id], context.n1_nodes, context.n2_nodes, membership)
        if not region_members.empty:
            region_members = add_variable_status_columns(region_members, config.r3_variables)
            region_members = _rv_proxy(region_members)
            region_members["is_anchor"] = region_members["pl_name"].astype("string").fillna("").eq(anchor_name)
            case_member_rows.append(
                region_members[
                    [
                        "case_id",
                        "node_id",
                        "region_type",
                        "pl_name",
                        "discoverymethod",
                        "pl_bmasse",
                        "pl_orbper",
                        "pl_orbsmax",
                        "r3_log_mass",
                        "r3_log_period",
                        "r3_log_semimajor",
                        "r3_z_mass",
                        "r3_z_period",
                        "r3_z_semimajor",
                        "imputation_status_pl_bmasse",
                        "imputation_status_pl_orbper",
                        "imputation_status_pl_orbsmax",
                        "rv_proxy",
                        "is_anchor",
                        "belongs_to_target_node",
                        "r3_valid",
                        "row_index",
                    ]
                ].copy()
            )

        case_deficit_rows.append(deficit_df)

    case_node_summary = pd.DataFrame(case_node_rows)
    case_anchor_planets = pd.DataFrame(case_anchor_rows)
    case_method_composition = pd.concat(case_method_rows, ignore_index=True) if case_method_rows else pd.DataFrame()
    case_r3_planet_members = pd.concat(case_member_rows, ignore_index=True) if case_member_rows else pd.DataFrame()
    anchor_neighbor_deficit = pd.concat(case_deficit_rows, ignore_index=True) if case_deficit_rows else pd.DataFrame()

    best_analog = (
        anchor_neighbor_deficit.groupby("case_id", as_index=False)["delta_rel_analog"].max().rename(columns={"delta_rel_analog": "delta_rel_analog_best"})
        if not anchor_neighbor_deficit.empty
        else pd.DataFrame(columns=["case_id", "delta_rel_analog_best"])
    )
    case_comparison_summary = case_node_summary.merge(case_anchor_planets[["case_id", "anchor_pl_name"]], on="case_id", how="left")
    if not anchor_neighbor_deficit.empty:
        best_neighbors = anchor_neighbor_deficit.groupby("case_id", as_index=False).agg(
            delta_rel_neighbors_best=("delta_rel_neighbors", "max"),
            expected_missing_direction=("expected_missing_direction", "first"),
            caution_text=("caution_text", "first"),
        )
        case_comparison_summary = case_comparison_summary.merge(best_neighbors, on="case_id", how="left")
        case_comparison_summary = case_comparison_summary.merge(best_analog, on="case_id", how="left")
    else:
        case_comparison_summary["delta_rel_neighbors_best"] = None
        case_comparison_summary["expected_missing_direction"] = "not_available"
        case_comparison_summary["delta_rel_analog_best"] = None
        case_comparison_summary["caution_text"] = "Interpretation not available."
    case_comparison_summary["deficit_class"] = case_comparison_summary.apply(
        lambda row: "strong_deficit" if pd.notna(row.get("delta_rel_neighbors_best")) and float(row["delta_rel_neighbors_best"]) > 0.6 else (
            "moderate_deficit" if pd.notna(row.get("delta_rel_neighbors_best")) and float(row["delta_rel_neighbors_best"]) > 0.3 else (
                "weak_deficit" if pd.notna(row.get("delta_rel_neighbors_best")) and float(row["delta_rel_neighbors_best"]) > 0.1 else "no_deficit"
            )
        ),
        axis=1,
    )
    case_comparison_summary["confidence_level"] = case_comparison_summary.apply(confidence_level, axis=1)
    case_comparison_summary["final_interpretation"] = case_comparison_summary.apply(final_interpretation, axis=1)

    tables_dir = output_tree["tables"]
    write_csv(case_node_summary, tables_dir / "case_node_summary.csv")
    write_csv(case_anchor_planets, tables_dir / "case_anchor_planets.csv")
    write_csv(anchor_neighbor_deficit, tables_dir / "anchor_neighbor_deficit.csv")
    write_csv(case_method_composition, tables_dir / "case_method_composition.csv")
    write_csv(case_r3_planet_members.drop(columns=["row_index"], errors="ignore"), tables_dir / "case_r3_planet_members.csv")
    write_csv(case_comparison_summary, tables_dir / "case_comparison_summary.csv")

    figures_dir = output_tree["figures"]
    for _, row in case_node_summary.iterrows():
        node_id = str(row["node_id"])
        case_id = str(row["case_id"])
        context = case_contexts.get(case_id)
        plot_case_ego_network(
            figures_dir / f"case_{node_id}_ego_network.pdf",
            node_id=node_id,
            graph=graph,
            node_metrics=node_summary_frame,
            n1_nodes=getattr(context, "n1_nodes", []),
            n2_nodes=getattr(context, "n2_nodes", []),
            shadow_score=float(pd.to_numeric(pd.Series([row.get('shadow_score')]), errors="coerce").fillna(np.nan).iloc[0]),
            seed=config.seed,
        )
        members = case_r3_planet_members[case_r3_planet_members["case_id"] == case_id].copy()
        anchor_name = case_anchor_planets.loc[case_anchor_planets["case_id"] == case_id, "anchor_pl_name"].iloc[0] if not case_anchor_planets[case_anchor_planets["case_id"] == case_id].empty else None
        plot_case_r3_projections(figures_dir / f"case_{node_id}_r3_projections.pdf", node_id, members, anchor_name)
        plot_case_method_composition(figures_dir / f"case_{node_id}_method_node_vs_neighbors.pdf", node_id, case_method_composition[case_method_composition["case_id"] == case_id].copy())
        plot_case_imputation_audit(figures_dir / f"case_{node_id}_imputation_r3_audit.pdf", node_id, row.to_dict())
        plot_case_anchor_deficit(figures_dir / f"case_{node_id}_anchor_deficit.pdf", node_id, anchor_neighbor_deficit[anchor_neighbor_deficit["case_id"] == case_id].copy())
        plot_case_rv_proxy_distribution(figures_dir / f"case_{node_id}_rv_proxy_distribution.pdf", node_id, members, anchor_name)

    plot_three_case_shadow_vs_physical_distance(figures_dir / "three_case_shadow_vs_physical_distance.pdf", case_comparison_summary)
    plot_three_case_deficit_comparison(figures_dir / "three_case_deficit_comparison.pdf", case_comparison_summary)
    plot_three_case_confidence_matrix(figures_dir / "three_case_confidence_matrix.pdf", case_comparison_summary)

    expected_figures = [figures_dir / "three_case_shadow_vs_physical_distance.pdf", figures_dir / "three_case_deficit_comparison.pdf", figures_dir / "three_case_confidence_matrix.pdf"]
    for path in expected_figures:
        if not path.exists():
            placeholder_figure(path, path.stem, "La figura no pudo generarse con los datos disponibles.")

    readme_path = local_outputs_dir / "README.md"
    write_text(readme_path, _case_readme())
    interpretation_text = _interpretation_summary(case_comparison_summary)
    write_text(local_outputs_dir / "interpretation_summary.md", interpretation_text)
    write_latex_report(latex_dir, case_comparison_summary, config.requested_node_ids, selected_nodes, replacements)

    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit_hash(),
        "mapper_configuration_used": config.primary_config_id,
        "requested_nodes": config.requested_node_ids,
        "analyzed_nodes": selected_nodes,
        "replacements": replacements,
        "input_paths": manifest_input_paths(inputs),
        "output_paths": {
            "root": repo_relative(local_outputs_dir),
            "figures": repo_relative(figures_dir),
            "tables": repo_relative(tables_dir),
            "metadata": repo_relative(output_tree["metadata"]),
            "logs": repo_relative(output_tree["logs"]),
            "latex": repo_relative(latex_dir / "main.tex"),
        },
        "variables_r3": config.r3_variables,
        "radius_parameters": ["r_node_median", "r_node_q75", "r_kNN"],
        "thresholds": {
            "analog_shadow_quantile": config.analog_shadow_quantile,
            "analog_centroid_distance_tau": config.analog_centroid_distance_tau,
            "analog_min_nodes": config.analog_min_nodes,
            "analog_min_members": config.analog_min_members,
            "analog_min_r3_valid_fraction": config.analog_min_r3_valid_fraction,
            "neighbor_reference_min_size": config.neighbor_reference_min_size,
        },
        "warnings": warnings,
        "n_planets_analyzed": int(membership["row_index"].nunique()),
        "n_planets_r3_valid": int(membership.loc[membership["r3_valid"], "row_index"].nunique()),
    }
    write_json(local_outputs_dir / "run_manifest.json", manifest)
    save_log(output_tree["logs"] / "local_shadow_case_studies.log", warnings)

    print("Local shadow case studies completado.")
    print(f"Nodos analizados: {', '.join(selected_nodes)}")
    print(f"Outputs: {local_outputs_dir}")
    print(f"LaTeX: {latex_dir / 'main.tex'}")


if __name__ == "__main__":
    main()
