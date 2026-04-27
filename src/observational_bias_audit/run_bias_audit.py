from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    load_audit_config,
    resolved_audit_outputs_dir,
    resolved_latex_dir,
    resolved_mapper_outputs_dir,
)
from .io import (
    append_log,
    discover_available_config_ids,
    git_commit_hash,
    load_edge_table,
    load_graph_payload,
    load_node_table,
    load_or_rebuild_membership,
    load_physical_catalog,
    save_log,
    write_json,
)
from .latex import write_latex_report
from .metrics import (
    build_central_vs_peripheral_summary,
    build_component_method_summary,
    build_global_bias_row,
    build_node_metrics,
)
from .paths import ensure_audit_output_tree, ensure_latex_dir
from .permutation import run_permutation_audit
from .plotting import (
    placeholder_figure,
    plot_component_method_composition,
    plot_config_comparison,
    plot_enrichment_heatmap,
    plot_graph_by_continuous_metric,
    plot_graph_by_dominant_method,
    plot_node_method_heatmap,
    plot_null_histogram,
    plot_scatter,
)
from .tables import (
    build_peripheral_bias_nodes,
    build_summary_global_bias_metrics,
    build_top_enriched_nodes,
    write_csv,
    write_tex_table,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Auditoria observacional post-hoc sobre grafos Mapper ya construidos.")
    parser.add_argument("--config", default="configs/observational_bias_audit.yaml", help="Ruta al archivo de configuracion YAML/JSON-compatible.")
    parser.add_argument("--n-permutations", type=int, default=None, help="Override del numero de permutaciones.")
    parser.add_argument("--seed", type=int, default=None, help="Override de la semilla reproducible.")
    return parser.parse_args()


def _safe_group_mode(series: pd.Series) -> str:
    mode = series.mode(dropna=True)
    return str(mode.iloc[0]) if not mode.empty else "Unknown"


def _write_readme(outputs_dir: Path, config_ids: list[str]) -> None:
    content = f"""# Observational Bias Audit

## Pregunta
Evalua si la topologia observada en los grafos Mapper, especialmente en el espacio orbital, esta asociada al metodo de descubrimiento y otros sesgos observacionales del catalogo.

## Inputs
- Artifacts de Mapper en `outputs/mapper/`
- Catalogo fisico/imputado alineado con Mapper
- Metadata observacional con `pl_name`, `discoverymethod`, `disc_year` y `disc_facility`

## Outputs
- Figuras PDF en `outputs/observational_bias_audit/figures/`
- Tablas CSV y `.tex` en `outputs/observational_bias_audit/tables/`
- Metadata, membresias reconstruidas y distribuciones nulas en `outputs/observational_bias_audit/metadata/`
- Logs en `outputs/observational_bias_audit/logs/`

## Ejecucion
`python -m src.observational_bias_audit.run_bias_audit --config configs/observational_bias_audit.yaml`

## Metricas principales
- `top_method_fraction`: pureza nodal por metodo de descubrimiento.
- `method_entropy` y `method_entropy_norm`: mezcla interna de metodos por nodo.
- `node_method_nmi`: asociacion global entre incidencias nodo-metodo.
- `enrichment_ratio`, `z_score` y `fdr_q_value`: sobrerrepresentacion local de metodos en nodos.

## Limitaciones
- No implementa la via contrafactual ni rehace Mapper.
- Los resultados dependen de la metadata observacional y de la trazabilidad de imputacion ya existente.
- Nodos pequenos o perifericos pueden mostrar pureza alta por tamano muestral.

## Configuraciones analizadas
{chr(10).join(f"- `{config_id}`" for config_id in config_ids)}
"""
    (outputs_dir / "README.md").write_text(content, encoding="utf-8")


def _interpretation(summary_global: pd.DataFrame, top_enriched_nodes: pd.DataFrame, peripheral_bias_nodes: pd.DataFrame) -> str:
    if summary_global.empty:
        return "No se pudieron calcular metricas globales para interpretar la auditoria."
    orbital = summary_global[summary_global["config_id"] == "orbital_pca2_cubes10_overlap0p35"]
    target = orbital.iloc[0] if not orbital.empty else summary_global.iloc[0]
    nmi = pd.to_numeric(pd.Series([target.get("node_method_nmi")]), errors="coerce").iloc[0]
    nmi_z = pd.to_numeric(pd.Series([target.get("global_nmi_z_score")]), errors="coerce").iloc[0]
    nmi_p = pd.to_numeric(pd.Series([target.get("global_nmi_empirical_p_value")]), errors="coerce").iloc[0]
    mean_imputation = pd.to_numeric(pd.Series([target.get("mean_node_imputation_fraction")]), errors="coerce").iloc[0]
    significant_nodes = top_enriched_nodes[
        pd.to_numeric(top_enriched_nodes.get("fdr_q_value"), errors="coerce").fillna(1.0) <= 0.10
    ]
    peripheral_significant = peripheral_bias_nodes[
        pd.to_numeric(peripheral_bias_nodes.get("fdr_q_value"), errors="coerce").fillna(1.0) <= 0.10
    ]
    if np.isfinite(nmi) and np.isfinite(nmi_z) and np.isfinite(nmi_p) and nmi_z >= 2.0 and nmi_p <= 0.05 and not significant_nodes.empty:
        if not peripheral_significant.empty and len(peripheral_significant) >= max(1, len(significant_nodes) // 2):
            if np.isfinite(mean_imputation) and mean_imputation >= 0.25:
                return (
                    "La NMI observada supera claramente la nula y aparecen nodos enriquecidos con q-value bajo, "
                    "pero buena parte de la senal se concentra en ramas o periferias y convive con imputacion no trivial. "
                    "Hay evidencia de asociacion entre topologia Mapper y metodo de descubrimiento, aunque no conviene "
                    "separarla de la dependencia de imputacion sin una segunda via contrafactual."
                )
            return (
                "La NMI observada supera claramente la nula y los nodos con enriquecimiento significativo se concentran "
                "en zonas perifericas. La evidencia sugiere que el sesgo observacional existe y parece localizarse sobre "
                "todo en ramas o periferias, no necesariamente en el nucleo completo del grafo."
            )
        if np.isfinite(mean_imputation) and mean_imputation >= 0.25:
            return (
                "La NMI observada es mayor que la nula y hay nodos con enriquecimiento significativo, pero la senal "
                "coincide con una dependencia apreciable de imputacion. Hay evidencia de asociacion topologia-metodo, "
                "aunque su interpretacion debe condicionarse a la trazabilidad de imputacion."
            )
        return (
            "La NMI observada es mayor que la nula y hay nodos con pureza alta y enriquecimiento significativo. "
            "Esto constituye evidencia de asociacion entre la topologia Mapper y el metodo de descubrimiento."
        )
    if not peripheral_bias_nodes.empty and peripheral_bias_nodes["top_method_fraction"].max() >= 0.80:
        return (
            "No hay evidencia global fuerte de que el metodo de descubrimiento domine toda la topologia, pero si aparecen "
            "nodos pequenos o perifericos con pureza alta. El sesgo observacional parece localizado en ramas o regiones "
            "perifericas mas que en todo el grafo."
        )
    return (
        "La NMI observada no supera claramente la distribucion nula o no aparecen nodos con enriquecimiento robusto. "
        "En esta auditoria no hay evidencia fuerte de que el metodo de descubrimiento domine la topologia Mapper."
    )


def main() -> None:
    args = parse_args()
    config = load_audit_config(
        path=args.config,
        overrides={"n_permutations": args.n_permutations, "seed": args.seed},
    )
    mapper_outputs_dir = resolved_mapper_outputs_dir(config)
    audit_outputs_dir = resolved_audit_outputs_dir(config)
    latex_dir = ensure_latex_dir(resolved_latex_dir(config))
    output_tree = ensure_audit_output_tree(audit_outputs_dir)
    log_lines: list[str] = []

    available_config_ids = discover_available_config_ids(mapper_outputs_dir)
    requested_config_ids = config.all_config_ids()
    selected_config_ids: list[str] = []
    for config_id in requested_config_ids:
        if config_id in available_config_ids:
            selected_config_ids.append(config_id)
        else:
            append_log(log_lines, f"WARNING: no existe la configuracion requerida {config_id}; se omite.")
    if not selected_config_ids:
        raise RuntimeError("No se encontro ninguna configuracion Mapper disponible para la auditoria.")

    node_metrics_frames: list[pd.DataFrame] = []
    component_frames: list[pd.DataFrame] = []
    component_composition_frames: list[pd.DataFrame] = []
    count_matrix_frames: list[pd.DataFrame] = []
    fraction_matrix_frames: list[pd.DataFrame] = []
    permutation_frames: list[pd.DataFrame] = []
    enrichment_frames: list[pd.DataFrame] = []
    global_rows: list[dict[str, object]] = []
    input_paths: dict[str, str] = {}
    heatmap_criteria: dict[str, str] = {}

    for config_id in selected_config_ids:
        graph_payload = load_graph_payload(mapper_outputs_dir, config_id)
        graph_config = graph_payload.get("config", {})
        input_method = str(graph_config.get("input_method", "iterative"))
        physical_path, physical_df = load_physical_catalog(config.physical_csv_path, input_method=input_method)
        input_paths[f"{config_id}_physical_csv"] = str(physical_path)
        node_table = load_node_table(mapper_outputs_dir, config_id)
        edge_table = load_edge_table(mapper_outputs_dir, config_id)
        membership_df, membership_source = load_or_rebuild_membership(
            mapper_outputs_dir=mapper_outputs_dir,
            audit_metadata_dir=output_tree["metadata"],
            config_id=config_id,
            physical_df=physical_df,
        )
        append_log(log_lines, f"{config_id}: membership source -> {membership_source}")

        if "discoverymethod" not in physical_df.columns:
            raise RuntimeError(f"Falta discoverymethod en el catalogo fisico usado por {config_id}: {physical_path}")
        if membership_df.empty:
            raise RuntimeError(f"No fue posible construir la membresia nodo-planeta para {config_id}.")

        observed_columns = [column for column in ["pl_name", "discoverymethod", "disc_year", "disc_facility", "disc_telescope", "disc_instrument"] if column in physical_df.columns]
        join_columns = ["row_index", *observed_columns]
        trace_columns = [column for column in physical_df.columns if column.endswith("_was_imputed") or column.endswith("_was_physically_derived")]
        if trace_columns:
            join_columns.extend(trace_columns)
        membership_joined = membership_df.merge(
            physical_df.reset_index(drop=False).rename(columns={"index": "row_index"})[join_columns].copy(),
            on="row_index",
            how="left",
            suffixes=("", "_catalog"),
        )
        if membership_joined["discoverymethod"].isna().all():
            raise RuntimeError(f"Tras el join de membresia con catalogo, discoverymethod quedo vacio para {config_id}.")
        membership_joined["discoverymethod"] = membership_joined["discoverymethod"].astype("string").fillna("Unknown")
        membership_joined["disc_facility"] = membership_joined.get("disc_facility", pd.Series(index=membership_joined.index, dtype="string")).astype("string").fillna("Unknown")
        membership_joined["disc_year"] = pd.to_numeric(membership_joined.get("disc_year"), errors="coerce")
        membership_joined.to_csv(output_tree["metadata"] / f"membership_with_observational_metadata_{config_id}.csv", index=False)

        node_metrics, count_matrix, fraction_matrix, node_metadata = build_node_metrics(
            config_id=config_id,
            membership_with_metadata=membership_joined,
            node_table=node_table,
            edge_table=edge_table,
            peripheral_degree_threshold=config.peripheral_degree_threshold,
            peripheral_component_max_nodes=config.peripheral_component_max_nodes,
        )
        if node_metadata.get("excluded_zero_member_nodes"):
            append_log(log_lines, f"WARNING: {config_id} excluyo nodos sin miembros tras el join: {node_metadata['excluded_zero_member_nodes']}")
        append_log(log_lines, f"{config_id}: {node_metadata['peripheral_rule']}")

        component_summary, component_composition = build_component_method_summary(
            config_id=config_id,
            membership_with_metadata=membership_joined,
            node_metrics=node_metrics,
            edge_table=edge_table,
        )
        global_row = build_global_bias_row(
            config_id=config_id,
            node_metrics=node_metrics,
            membership_with_metadata=membership_joined,
            count_matrix=count_matrix,
            edge_table=edge_table,
        )
        permutation_df, enrichment_df, null_distribution = run_permutation_audit(
            config_id=config_id,
            membership_with_metadata=membership_joined,
            node_metrics=node_metrics,
            edge_table=edge_table,
            n_permutations=config.n_permutations,
            seed=config.seed,
        )
        null_distribution.to_csv(output_tree["metadata"] / f"permutation_null_distribution_{config_id}.csv", index=False)

        node_metrics_frames.append(node_metrics)
        component_frames.append(component_summary)
        component_composition_frames.append(component_composition)
        count_matrix_frames.append(count_matrix)
        fraction_matrix_frames.append(fraction_matrix)
        permutation_frames.append(permutation_df)
        enrichment_frames.append(enrichment_df)
        global_rows.append(global_row)

        if config_id == config.primary_config_id:
            figures_dir = output_tree["figures"]
            plot_graph_by_dominant_method(
                node_metrics=node_metrics,
                edge_table=edge_table,
                path=figures_dir / "orbital_graph_by_dominant_discovery_method.pdf",
                seed=config.seed,
            )
            plot_graph_by_continuous_metric(
                node_metrics=node_metrics,
                edge_table=edge_table,
                value_column="top_method_fraction",
                title="Mapper orbital coloreado por pureza de metodo dominante",
                path=figures_dir / "orbital_graph_by_method_purity.pdf",
                seed=config.seed,
                cmap="viridis",
            )
            plot_graph_by_continuous_metric(
                node_metrics=node_metrics,
                edge_table=edge_table,
                value_column="method_entropy_norm",
                title="Mapper orbital coloreado por entropia normalizada",
                path=figures_dir / "orbital_graph_by_method_entropy.pdf",
                seed=config.seed,
                cmap="magma",
            )
            heatmap_criteria[config_id] = plot_node_method_heatmap(
                fraction_matrix=fraction_matrix,
                node_metrics=node_metrics,
                path=figures_dir / "orbital_node_method_fraction_heatmap.pdf",
                top_n_nodes=config.heatmap_top_n_nodes,
            )
            plot_enrichment_heatmap(
                enrichment_df=enrichment_df,
                node_metrics=node_metrics,
                path=figures_dir / "orbital_method_enrichment_zscores.pdf",
                top_n_nodes=config.enrichment_top_n_nodes,
                top_n_methods=config.enrichment_top_n_methods,
            )
            observed_nmi = permutation_df[permutation_df["metric"] == "node_method_nmi"]["observed"]
            observed_nmi_value = float(observed_nmi.iloc[0]) if not observed_nmi.empty else np.nan
            plot_null_histogram(
                null_distribution=null_distribution,
                metric="node_method_nmi",
                observed_value=observed_nmi_value,
                title="Distribucion nula de NMI para el Mapper orbital",
                path=figures_dir / "orbital_permutation_null_nmi.pdf",
            )
            plot_scatter(
                node_metrics=node_metrics,
                x_column="mean_imputation_fraction",
                y_column="top_method_fraction",
                title="Pureza de metodo vs imputacion media nodal",
                path=figures_dir / "orbital_purity_vs_imputation.pdf",
            )
            plot_scatter(
                node_metrics=node_metrics,
                x_column="n_members",
                y_column="top_method_fraction",
                title="Pureza de metodo vs tamano de nodo",
                path=figures_dir / "orbital_purity_vs_node_size.pdf",
            )
            plot_component_method_composition(
                component_composition=component_composition,
                path=figures_dir / "orbital_component_method_composition.pdf",
            )

    node_metrics_all = pd.concat(node_metrics_frames, ignore_index=True) if node_metrics_frames else pd.DataFrame()
    component_all = pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame()
    count_matrix_all = pd.concat(count_matrix_frames, ignore_index=True) if count_matrix_frames else pd.DataFrame()
    fraction_matrix_all = pd.concat(fraction_matrix_frames, ignore_index=True) if fraction_matrix_frames else pd.DataFrame()
    permutation_all = pd.concat(permutation_frames, ignore_index=True) if permutation_frames else pd.DataFrame()
    enrichment_all = pd.concat(enrichment_frames, ignore_index=True) if enrichment_frames else pd.DataFrame()
    component_composition_all = pd.concat(component_composition_frames, ignore_index=True) if component_composition_frames else pd.DataFrame()
    summary_global = build_summary_global_bias_metrics(pd.DataFrame(global_rows), permutation_all)
    top_enriched_nodes = build_top_enriched_nodes(node_metrics=node_metrics_all, enrichment_df=enrichment_all, top_n=30)
    peripheral_bias_nodes = build_peripheral_bias_nodes(node_metrics=node_metrics_all, enrichment_df=enrichment_all)
    central_vs_peripheral = (
        pd.concat(
            [build_central_vs_peripheral_summary(group.copy()) for _, group in node_metrics_all.groupby("config_id")],
            ignore_index=True,
        )
        if not node_metrics_all.empty
        else pd.DataFrame()
    )

    write_csv(node_metrics_all, output_tree["tables"] / "node_method_bias_metrics.csv")
    write_csv(component_all, output_tree["tables"] / "component_method_bias_summary.csv")
    write_csv(count_matrix_all, output_tree["tables"] / "node_method_count_matrix.csv")
    write_csv(fraction_matrix_all, output_tree["tables"] / "node_method_fraction_matrix.csv")
    write_csv(permutation_all, output_tree["tables"] / "global_permutation_tests.csv")
    write_csv(enrichment_all, output_tree["tables"] / "node_method_enrichment.csv")
    write_csv(summary_global, output_tree["tables"] / "summary_global_bias_metrics.csv")
    write_csv(top_enriched_nodes, output_tree["tables"] / "top_enriched_nodes.csv")
    write_csv(peripheral_bias_nodes, output_tree["tables"] / "peripheral_bias_nodes.csv")
    write_csv(central_vs_peripheral, output_tree["tables"] / "central_vs_peripheral_bias.csv")

    write_tex_table(summary_global, output_tree["tables"] / "summary_global_bias_metrics.tex", "Resumen global de metricas de sesgo observacional.")
    write_tex_table(top_enriched_nodes, output_tree["tables"] / "top_enriched_nodes.tex", "Nodos con mayor enriquecimiento por metodo de descubrimiento.")
    write_tex_table(central_vs_peripheral, output_tree["tables"] / "central_vs_peripheral_bias.tex", "Comparacion entre nodos centrales y perifericos.")

    plot_config_comparison(summary_global, output_tree["figures"] / "config_comparison_bias_metrics.pdf")

    primary_figures = [
        "orbital_graph_by_dominant_discovery_method.pdf",
        "orbital_graph_by_method_purity.pdf",
        "orbital_graph_by_method_entropy.pdf",
        "orbital_node_method_fraction_heatmap.pdf",
        "orbital_method_enrichment_zscores.pdf",
        "orbital_permutation_null_nmi.pdf",
        "orbital_purity_vs_imputation.pdf",
        "orbital_purity_vs_node_size.pdf",
        "orbital_component_method_composition.pdf",
        "config_comparison_bias_metrics.pdf",
    ]
    for filename in primary_figures:
        path = output_tree["figures"] / filename
        if not path.exists():
            placeholder_figure(path, filename.replace("_", " "), "La figura no pudo generarse con los datos disponibles.")

    interpretation_text = _interpretation(summary_global, top_enriched_nodes, peripheral_bias_nodes)
    (audit_outputs_dir / "interpretation_summary.md").write_text(interpretation_text + "\n", encoding="utf-8")
    _write_readme(audit_outputs_dir, selected_config_ids)

    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit_hash(),
        "configuration": config.to_dict(),
        "n_permutations": config.n_permutations,
        "seed": config.seed,
        "input_paths": input_paths,
        "output_root": str(audit_outputs_dir),
        "mapper_configurations_analyzed": selected_config_ids,
        "heatmap_criteria": heatmap_criteria,
    }
    write_json(audit_outputs_dir / "run_manifest.json", manifest)
    write_json(output_tree["metadata"] / "analysis_metadata.json", {"selected_config_ids": selected_config_ids, "available_config_ids": available_config_ids})

    write_latex_report(
        latex_dir=latex_dir,
        outputs_dir=audit_outputs_dir,
        interpretation_text=interpretation_text,
    )

    save_log(output_tree["logs"] / "bias_audit.log", log_lines)
    print("Observational bias audit completado.")
    print(f"Configuraciones analizadas: {', '.join(selected_config_ids)}")
    print(f"Outputs: {audit_outputs_dir}")
    print(f"LaTeX: {latex_dir / 'main.tex'}")


if __name__ == "__main__":
    main()
