from __future__ import annotations

import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .anchor_index import (
    build_anchor_interpretation,
    classify_anchor_deficit,
    compute_ati,
    expected_incompleteness_direction,
    anchor_representativeness,
    select_anchor,
    unique_planets,
)
from .config import load_config
from .io import git_commit_hash, load_pipeline_inputs, save_log, write_csv, write_json, write_text
from .neighbor_deficit import NeighborDeficitParameters, compute_anchor_neighbor_deficits
from .network_metrics import add_network_support, build_graph, graph_metrics, node_neighborhood
from .paths import ensure_latex_dir, ensure_output_tree, repo_relative, resolve_repo_path
from .plotting import (
    placeholder_figure,
    plot_ati_vs_delta,
    plot_deficit_distribution_by_radius,
    plot_method_distribution_high_toi,
    plot_rank_matrix,
    plot_toi_vs_physical_distance,
    plot_toi_vs_shadow,
    plot_top_anchors,
    plot_top_anchor_rv_proxy,
    plot_top_regions,
)
from .r3_geometry import R3Columns, build_r3_frame, centroid, centroid_distance, node_r3_imputation_summary
from .regional_index import compute_toi_scores
from .reporting import build_interpretation_summary, validate_prudent_text, write_latex_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute TOI and ATI scores for Mapper regions and anchor planets.")
    parser.add_argument("--config", default="configs/topological_incompleteness_index.yaml")
    parser.add_argument("--config-id", default=None)
    return parser.parse_args()


def _rv_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["rv_proxy"] = pd.to_numeric(out["r3_log_mass"], errors="coerce") - (1.0 / 3.0) * pd.to_numeric(out["r3_log_period"], errors="coerce")
    st_mass = pd.to_numeric(out.get("st_mass"), errors="coerce")
    out["rv_proxy_with_star"] = np.nan
    valid = st_mass.notna() & np.isfinite(st_mass) & (st_mass > 0)
    out.loc[valid, "rv_proxy_with_star"] = (
        pd.to_numeric(out.loc[valid, "r3_log_mass"], errors="coerce")
        - (1.0 / 3.0) * pd.to_numeric(out.loc[valid, "r3_log_period"], errors="coerce")
        - (2.0 / 3.0) * np.log10(st_mass.loc[valid].astype(float))
    )
    return out


def _node_geometry_table(membership: pd.DataFrame, columns: R3Columns, graph, skipped_items: list[dict[str, object]], warnings: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = membership.groupby(membership["node_id"].astype(str))
    for node_id, group in grouped:
        node_members = unique_planets(group.copy())
        valid = node_members[node_members["r3_valid"]].copy()
        log_center = centroid(valid, columns.log)
        z_center = centroid(valid, columns.z)
        spread = None
        if z_center is not None and not valid.empty:
            spread = float(np.mean(np.linalg.norm(valid[columns.z].to_numpy(dtype=float) - z_center, axis=1)))
        summary = node_r3_imputation_summary(node_members, columns)
        rows.append(
            {
                "config_id": str(node_members["config_id"].astype(str).iloc[0]) if "config_id" in node_members.columns and not node_members.empty else None,
                "node_id": str(node_id),
                "centroid_log_mass": log_center[0] if log_center is not None else None,
                "centroid_log_period": log_center[1] if log_center is not None else None,
                "centroid_log_semimajor": log_center[2] if log_center is not None else None,
                "centroid_z_mass": z_center[0] if z_center is not None else None,
                "centroid_z_period": z_center[1] if z_center is not None else None,
                "centroid_z_semimajor": z_center[2] if z_center is not None else None,
                "spread_r3": spread,
                **summary,
            }
        )
        if valid.empty:
            skipped_items.append({"reason": "no_r3_valid", "node_id": str(node_id)})
        neighborhood = node_neighborhood(graph, str(node_id))
        if not neighborhood.n1_nodes:
            skipped_items.append({"reason": "no_neighbors", "node_id": str(node_id)})
    return pd.DataFrame(rows)


def _attach_physical_distances(regional_base: pd.DataFrame, node_geometry: pd.DataFrame, membership: pd.DataFrame, graph) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    geometry_map = node_geometry.set_index("node_id")
    for _, row in regional_base.iterrows():
        node_id = str(row["node_id"])
        neighborhood = node_neighborhood(graph, node_id)
        center = geometry_map.loc[node_id, ["centroid_z_mass", "centroid_z_period", "centroid_z_semimajor"]].to_numpy(dtype=float) if node_id in geometry_map.index else None
        n1_members = unique_planets(membership[membership["node_id"].astype(str).isin(neighborhood.n1_nodes)].copy())
        n2_members = unique_planets(membership[membership["node_id"].astype(str).isin(neighborhood.n2_nodes)].copy())
        n1_center = centroid(n1_members[n1_members["r3_valid"]], ["r3_z_mass", "r3_z_period", "r3_z_semimajor"])
        n2_center = centroid(n2_members[n2_members["r3_valid"]], ["r3_z_mass", "r3_z_period", "r3_z_semimajor"])
        rows.append(
            {
                "node_id": node_id,
                "physical_distance_v_to_N1": centroid_distance(center, n1_center),
                "physical_distance_v_to_N2": centroid_distance(center, n2_center),
            }
        )
    return regional_base.merge(pd.DataFrame(rows), on="node_id", how="left")


def _summary_table(regions: pd.DataFrame, anchors: pd.DataFrame) -> pd.DataFrame:
    top_region = regions.sort_values("TOI", ascending=False).head(1) if not regions.empty else pd.DataFrame()
    top_anchor = anchors.sort_values("ATI", ascending=False).head(1) if not anchors.empty else pd.DataFrame()
    row = {
        "n_nodes_analyzed": int(len(regions)),
        "n_high_toi_regions": int((regions["region_class"].astype(str) == "high_toi_region").sum()) if not regions.empty else 0,
        "n_anchor_planets": int(len(anchors)),
        "max_TOI": float(pd.to_numeric(top_region["TOI"], errors="coerce").iloc[0]) if not top_region.empty else None,
        "max_ATI": float(pd.to_numeric(top_anchor["ATI"], errors="coerce").iloc[0]) if not top_anchor.empty else None,
        "top_region_node_id": str(top_region["node_id"].iloc[0]) if not top_region.empty else None,
        "top_anchor_pl_name": str(top_anchor["anchor_pl_name"].iloc[0]) if not top_anchor.empty else None,
        "median_delta_rel_neighbors_best": float(pd.to_numeric(anchors["delta_rel_neighbors_best"], errors="coerce").median()) if not anchors.empty else None,
        "n_moderate_or_strong_deficit": int(anchors["deficit_class"].astype(str).isin(["moderate_deficit", "strong_deficit"]).sum()) if not anchors.empty else 0,
        "most_common_expected_direction": anchors["expected_incompleteness_direction"].astype(str).mode().iloc[0] if not anchors.empty and not anchors["expected_incompleteness_direction"].astype(str).mode().empty else None,
    }
    return pd.DataFrame([row])


def main() -> None:
    args = parse_args()
    overrides = {"analysis": {"config_id": args.config_id}} if args.config_id else None
    config = load_config(args.config, overrides=overrides)

    base_dir = resolve_repo_path(config.outputs.base_dir, None)
    tables_dir = resolve_repo_path(config.outputs.tables_dir, None)
    figures_dir = resolve_repo_path(config.outputs.figures_dir, None)
    metadata_dir = resolve_repo_path(config.outputs.metadata_dir, None)
    logs_dir = resolve_repo_path(config.outputs.logs_dir, None)
    latex_dir = ensure_latex_dir(resolve_repo_path(config.outputs.latex_dir, None))
    output_tree = ensure_output_tree(base_dir, figures_dir, tables_dir, metadata_dir, logs_dir)
    warnings: list[str] = []
    skipped_items: list[dict[str, object]] = []

    inputs = load_pipeline_inputs(config_root := resolve_repo_path(None, base_dir).parents[1], config, metadata_dir, warnings)
    columns = R3Columns(
        mass=config.analysis.r3_variables["mass"],
        period=config.analysis.r3_variables["period"],
        semimajor=config.analysis.r3_variables["semimajor"],
    )
    membership_r3, r3_meta = build_r3_frame(inputs["membership"], columns, warnings, skipped_items)
    membership_r3 = _rv_proxy(membership_r3)

    graph = build_graph(inputs["edges"], all_nodes=sorted(inputs["node_shadow_metrics"]["node_id"].astype(str).unique().tolist()))
    net = graph_metrics(graph, membership_r3, epsilon=config.analysis.epsilon)

    regional_base = inputs["node_shadow_metrics"].merge(net, on="node_id", how="left", suffixes=("", "_graph"))
    if "component_id" not in regional_base.columns and "component_id_graph" in regional_base.columns:
        regional_base["component_id"] = regional_base["component_id_graph"]
    node_geometry = _node_geometry_table(membership_r3, columns, graph, skipped_items, warnings)
    regional_base = regional_base.merge(node_geometry[["node_id", "n_r3_valid", "r3_valid_fraction", "I_R3"]], on="node_id", how="left")
    regional_base = _attach_physical_distances(regional_base, node_geometry, membership_r3, graph)
    regional_base = add_network_support(regional_base, epsilon=config.analysis.epsilon)
    regional_scores = compute_toi_scores(
        regional_base,
        sigma=config.toi.physical_sigma,
        epsilon=config.analysis.epsilon,
        min_node_members=config.toi.min_node_members,
        high_priority_quantile=config.toi.high_priority_quantile,
    )
    regional_scores = regional_scores[
        [
            "config_id",
            "node_id",
            "n_members",
            "n_r3_valid",
            "r3_valid_fraction",
            "degree",
            "component_id",
            "component_size_nodes",
            "shadow_score",
            "top_method",
            "top_method_fraction",
            "method_entropy_norm",
            "method_l1_boundary_N1" if "method_l1_boundary_N1" in regional_scores.columns else "method_l1_boundary",
            "I_R3",
            "physical_distance_v_to_N1",
            "C_phys",
            "size_weight",
            "degree_weight",
            "S_net",
            "TOI",
            "TOI_rank",
            "region_class",
            "interpretation_text",
            "caution_text",
        ]
    ].rename(columns={"method_l1_boundary": "method_l1_boundary_N1"})

    anchor_records: list[dict[str, object]] = []
    deficit_frames: list[pd.DataFrame] = []
    params = NeighborDeficitParameters(
        epsilon=config.analysis.epsilon,
        knn_min=config.neighbor_deficit.knn_min,
        knn_max=config.neighbor_deficit.knn_max,
        analog_min_nodes=config.neighbor_deficit.analog_min_nodes,
        analog_shadow_quantile_max=config.neighbor_deficit.analog_shadow_quantile_max,
        analog_physical_distance_quantile=config.neighbor_deficit.analog_physical_distance_quantile,
        reference_min_planets=config.neighbor_deficit.reference_min_planets,
        min_node_members=config.toi.min_node_members,
        min_r3_valid_fraction=config.analysis.min_r3_valid_fraction_for_analogs,
    )
    regional_map = regional_scores.set_index("node_id")
    for _, region in regional_scores.iterrows():
        node_id = str(region["node_id"])
        node_members = unique_planets(membership_r3[membership_r3["node_id"].astype(str) == node_id].copy())
        if node_members.empty:
            skipped_items.append({"reason": "missing_membership", "node_id": node_id})
            continue
        anchor, reason = select_anchor(node_members, columns, config.ati.prefer_method)
        if anchor is None:
            skipped_items.append({"reason": "no_r3_valid", "node_id": node_id})
            continue
        deficits, deficit_summary = compute_anchor_neighbor_deficits(
            config_id=config.analysis.config_id,
            node_id=node_id,
            anchor=anchor,
            membership=membership_r3,
            graph=graph,
            node_geometry=node_geometry,
            regional_scores=regional_scores,
            z_cols=columns.z,
            params=params,
            warnings=warnings,
        )
        deficit_frames.append(deficits)
        representativeness, distance_to_node_centroid = anchor_representativeness(anchor, node_members, columns.z, config.analysis.epsilon)
        anchor_imputed_fraction_value = float(anchor.get("anchor_imputed_fraction", 0.0))
        toi_value = float(pd.to_numeric(pd.Series([regional_map.loc[node_id, "TOI"]]), errors="coerce").fillna(0.0).iloc[0]) if node_id in regional_map.index else 0.0
        ati = compute_ati(toi_value, float(deficit_summary["delta_rel_neighbors_best"] or 0.0), anchor_imputed_fraction_value, representativeness)
        record = {
            "config_id": config.analysis.config_id,
            "node_id": node_id,
            "anchor_pl_name": anchor.get("pl_name"),
            "discoverymethod": anchor.get("discoverymethod"),
            "disc_year": anchor.get("disc_year"),
            "disc_facility": anchor.get("disc_facility"),
            "pl_bmasse": anchor.get(columns.mass),
            "pl_orbper": anchor.get(columns.period),
            "pl_orbsmax": anchor.get(columns.semimajor),
            "r3_log_mass": anchor.get("r3_log_mass"),
            "r3_log_period": anchor.get("r3_log_period"),
            "r3_log_semimajor": anchor.get("r3_log_semimajor"),
            "r3_imputation_score": anchor.get("r3_imputation_score"),
            "distance_to_node_centroid": distance_to_node_centroid,
            "anchor_representativeness": representativeness,
            "rv_proxy": anchor.get("rv_proxy"),
            "rv_proxy_with_star": anchor.get("rv_proxy_with_star"),
            "delta_rel_neighbors_best": deficit_summary["delta_rel_neighbors_best"],
            "delta_rel_neighbors_mean": deficit_summary["delta_rel_neighbors_mean"],
            "delta_rel_analog_best": deficit_summary["delta_rel_analog_best"],
            "TOI": toi_value,
            "ATI": ati,
            "ATI_rank": None,
            "deficit_class": classify_anchor_deficit(float(deficit_summary["delta_rel_neighbors_best"] or 0.0)),
            "expected_incompleteness_direction": expected_incompleteness_direction(str(anchor.get("discoverymethod", "Unknown"))),
            "anchor_selection_reason": reason,
            "interpretation_text": None,
            "caution_text": "ATI prioriza submuestreo topologico local; no equivale a una conclusion sobre objetos ausentes.",
        }
        record["interpretation_text"] = build_anchor_interpretation(pd.Series(record))
        anchor_records.append(record)

    anchor_scores = pd.DataFrame(anchor_records)
    if not anchor_scores.empty:
        anchor_scores = anchor_scores.sort_values("ATI", ascending=False).reset_index(drop=True)
        anchor_scores["ATI_rank"] = np.arange(1, len(anchor_scores) + 1)
        validate_prudent_text("\n".join(anchor_scores["interpretation_text"].astype(str).tolist()))
    deficits_all = pd.concat(deficit_frames, ignore_index=True) if deficit_frames else pd.DataFrame()

    summary = _summary_table(regional_scores, anchor_scores)
    interpretation_summary = build_interpretation_summary(regional_scores, anchor_scores, summary, deficits_all)

    regional_out = output_tree["tables"] / "regional_toi_scores.csv"
    anchor_out = output_tree["tables"] / "anchor_ati_scores.csv"
    deficit_out = output_tree["tables"] / "anchor_neighbor_deficits.csv"
    geometry_out = output_tree["tables"] / "r3_node_geometry.csv"
    summary_out = output_tree["tables"] / "toi_ati_summary.csv"
    skipped_out = output_tree["tables"] / "skipped_items.csv"

    write_csv(regional_scores, regional_out)
    write_csv(anchor_scores, anchor_out)
    write_csv(deficits_all, deficit_out)
    write_csv(node_geometry, geometry_out)
    write_csv(summary, summary_out)
    write_csv(pd.DataFrame(skipped_items), skipped_out)

    if config.report.make_figures:
        plot_top_regions(regional_scores, output_tree["figures"] / "top_regions_toi_score.pdf")
        plot_top_anchors(anchor_scores, output_tree["figures"] / "top_anchor_ati_score.pdf")
        plot_toi_vs_shadow(regional_scores, output_tree["figures"] / "toi_vs_shadow_score.pdf")
        plot_toi_vs_physical_distance(regional_scores, output_tree["figures"] / "toi_vs_physical_distance.pdf")
        plot_ati_vs_delta(anchor_scores, output_tree["figures"] / "ati_vs_delta_rel_neighbors.pdf")
        plot_rank_matrix(regional_scores, anchor_scores, output_tree["figures"] / "toi_ati_rank_matrix.pdf")
        plot_deficit_distribution_by_radius(deficits_all, output_tree["figures"] / "deficit_distribution_by_radius.pdf")
        plot_top_anchor_rv_proxy(anchor_scores, output_tree["figures"] / "top_anchor_rv_proxy.pdf")
        plot_method_distribution_high_toi(regional_scores, output_tree["figures"] / "method_distribution_high_toi.pdf")
    expected_figures = [
        "top_regions_toi_score.pdf",
        "top_anchor_ati_score.pdf",
        "toi_vs_shadow_score.pdf",
        "toi_vs_physical_distance.pdf",
        "ati_vs_delta_rel_neighbors.pdf",
        "toi_ati_rank_matrix.pdf",
        "deficit_distribution_by_radius.pdf",
        "top_anchor_rv_proxy.pdf",
        "method_distribution_high_toi.pdf",
    ]
    for filename in expected_figures:
        path = output_tree["figures"] / filename
        if not path.exists():
            placeholder_figure(path, filename, "La figura no pudo generarse con los datos disponibles.")

    if config.report.make_summary:
        write_text(interpretation_summary, output_tree["base"] / "interpretation_summary.md")

    latex_path = latex_dir / "topological_incompleteness_index.tex"
    if config.report.make_latex:
        write_latex_report(latex_path)

    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit_hash(),
        "config": config.to_dict(),
        "input_paths": inputs["input_paths"],
        "attempted_paths": inputs["attempted_paths"],
        "output_paths": {
            "base_dir": repo_relative(output_tree["base"]),
            "tables_dir": repo_relative(output_tree["tables"]),
            "figures_dir": repo_relative(output_tree["figures"]),
            "metadata_dir": repo_relative(output_tree["metadata"]),
            "logs_dir": repo_relative(output_tree["logs"]),
            "latex_path": repo_relative(latex_path),
        },
        "n_nodes": int(len(regional_scores)),
        "n_planets": int(membership_r3["row_index"].nunique()) if "row_index" in membership_r3.columns else int(membership_r3["pl_name"].astype(str).nunique()),
        "n_anchors": int(len(anchor_scores)),
        "r3_variables": config.analysis.r3_variables,
        "thresholds": {
            "high_priority_quantile": config.toi.high_priority_quantile,
            "physical_sigma": config.toi.physical_sigma,
            "analog_shadow_quantile_max": config.neighbor_deficit.analog_shadow_quantile_max,
            "analog_physical_distance_quantile": config.neighbor_deficit.analog_physical_distance_quantile,
        },
        "warnings": warnings,
        "fallbacks_used": [warning for warning in warnings if "fallback" in warning.lower() or "reconstru" in warning.lower()],
        "r3_meta": r3_meta,
    }
    write_json(manifest, output_tree["metadata"] / "run_manifest.json")
    save_log(warnings, output_tree["logs"] / "topological_incompleteness_index.log")

    readme = """# Topological Incompleteness Index

TOI y ATI no descubren planetas faltantes; construyen un ranking topologico de regiones y planetas ancla donde el catalogo parece observacionalmente incompleto.

## Objetivo
Priorizar regiones Mapper y planetas ancla donde la combinacion de sombra observacional, continuidad fisica, soporte de red e imputacion en R^3 sugiere submuestreo topologico.

## Diferencia entre TOI y ATI
- TOI: indice regional para nodos Mapper.
- ATI: indice de planeta ancla que combina TOI con deficit local y representatividad.

## Como correr
python -m src.topological_incompleteness_index.run_topological_incompleteness --config configs/topological_incompleteness_index.yaml

## Inputs requeridos
- dataset imputado o base
- membership nodo-planeta
- edges Mapper
- node_observational_shadow_metrics.csv
- top_shadow_candidates.csv
- node_method_bias_metrics.csv
- node_method_fraction_matrix.csv

## Outputs generados
- CSVs en outputs/topological_incompleteness_index/tables/
- PDF en outputs/topological_incompleteness_index/figures_pdf/
- interpretation_summary.md
- metadata/run_manifest.json
- LaTeX en latex/04_topological_incompleteness/topological_incompleteness_index.tex

## Como interpretar delta_rel_neighbors_best
Es un resumen util para priorizacion, pero no debe leerse solo. Conviene revisarlo junto con los valores por radio, el promedio y la mediana.

## Por que no se afirma descubrimiento de planetas
Los indices usan referencias locales y topologicas. No modelan completitud instrumental ni equivalen a una conclusion sobre objetos ausentes.

## Como leer las figuras
- top_regions_toi_score.pdf: regiones con mayor TOI
- top_anchor_ati_score.pdf: anclas con mayor ATI
- toi_vs_shadow_score.pdf: como se apoya TOI en la sombra observacional
- deficit_distribution_by_radius.pdf: estabilidad del deficit relativo por radio

## Limitaciones
- no hay funcion de completitud instrumental
- N_exp es referencia local
- delta_rel_best puede inflar
- R^3 simplifica fisica
- proxy RV no es amplitud RV real
- la imputacion puede afectar masa

## Proximos pasos
- agregar completitud instrumental
- validar con catalogos futuros
- incorporar propiedades estelares
- comparar por metodo de descubrimiento
"""
    write_text(readme, resolve_repo_path("README_TOPOLOGICAL_INCOMPLETENESS_INDEX.md", output_tree["base"] / "README_TOPOLOGICAL_INCOMPLETENESS_INDEX.md"))

    print("Topological incompleteness index completado.")
    print(f"Config ID: {config.analysis.config_id}")
    print(f"Outputs: {output_tree['base']}")
    print(f"LaTeX: {latex_path}")


if __name__ == "__main__":
    main()
