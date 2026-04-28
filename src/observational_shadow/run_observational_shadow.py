from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import (
    load_shadow_config,
    resolved_audit_outputs_dir,
    resolved_latex_dir,
    resolved_mapper_outputs_dir,
    resolved_shadow_outputs_dir,
)
from .io import (
    discover_available_config_ids,
    git_commit_hash,
    load_edge_table,
    load_graph_payload,
    load_membership_with_catalog,
    load_node_table,
    load_physical_catalog,
    save_log,
    write_json,
)
from .latex import write_latex_report
from .node_profiles import add_interpretations, build_node_shadow_profiles
from .paths import ensure_latex_dir, ensure_output_tree, repo_relative
from .plotting import (
    placeholder_figure,
    plot_component_summary,
    plot_config_comparison,
    plot_graph_by_shadow_class,
    plot_graph_by_shadow_score,
    plot_scatter,
    plot_shadow_by_method,
    plot_top_candidates,
)
from .shadow_metrics import classify_shadow_nodes, compute_shadow_scores
from .tables import (
    build_component_summary,
    build_config_comparison,
    build_method_summary,
    build_top_candidates,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Indice de sombra observacional sobre comunidades Mapper.")
    parser.add_argument("--config", default="configs/observational_shadow.yaml", help="Ruta YAML/JSON de configuracion.")
    parser.add_argument("--seed", type=int, default=None, help="Override de semilla para layouts.")
    return parser.parse_args()


def _write_readme(outputs_dir: Path, config_ids: list[str]) -> None:
    content = f"""# Observational Shadow

## Pregunta
Este subproyecto pregunta si las comunidades Mapper pueden senalar regiones topologicas submuestreadas por la funcion de seleccion observacional del catalogo de exoplanetas.

## Inputs usados
- Grafos, nodos y aristas Mapper en `outputs/mapper/`.
- Membresias enriquecidas de la auditoria observacional cuando existen.
- Catalogos imputados/fisicos con `discoverymethod`, `disc_year`, `disc_facility` y variables fisico-orbitales.

## Como correr
`python -m src.observational_shadow.run_observational_shadow --config configs/observational_shadow.yaml`

## Outputs
- Figuras PDF: `outputs/observational_shadow/figures/`
- Tablas CSV: `outputs/observational_shadow/tables/`
- Metadata y manifest: `outputs/observational_shadow/metadata/` y `outputs/observational_shadow/run_manifest.json`
- Logs: `outputs/observational_shadow/logs/`
- Reporte LaTeX: `latex/observational_shadow/main.tex`

## Interpretacion de `shadow_score`
`shadow_score` combina pureza por metodo dominante, baja entropia, baja imputacion, contraste de composicion con vecinos y peso por tamano nodal. Un valor alto debe leerse como candidato a incompletitud observacional o posible frontera de seleccion, no como prueba de planetas faltantes.

## Limitaciones
Mapper no prueba poblaciones no observadas. Los cortes de clase son heuristicos. Nodos pequenos pueden inflar la pureza. La imputacion puede alterar algunas regiones. Sin funciones de completitud instrumental no se estiman cantidades absolutas de objetos no observados.

## Configuraciones analizadas
{chr(10).join(f"- `{config_id}`" for config_id in config_ids)}
"""
    (outputs_dir / "README.md").write_text(content, encoding="utf-8")


def _write_interpretation(outputs_dir: Path, node_metrics: pd.DataFrame, top_candidates: pd.DataFrame) -> str:
    high = node_metrics[node_metrics["shadow_class"] == "high_confidence_shadow"].copy()
    method_counts = high["top_method"].value_counts().head(6)
    lines = [
        "# Interpretacion automatica: sombra observacional",
        "",
        f"- Nodos con alta sombra y baja imputacion: {len(high)}",
        "- Metodos dominantes mas frecuentes entre nodos de alta sombra:",
    ]
    if method_counts.empty:
        lines.append("  - No se encontraron nodos de alta confianza con los cortes heuristicos actuales.")
    else:
        lines.extend([f"  - {method}: {count}" for method, count in method_counts.items()])
    lines.extend(["", "## Ejemplos de comunidades candidatas", ""])
    for _, row in top_candidates.head(8).iterrows():
        lines.append(
            f"- `{row['config_id']}` / `{row['node_id']}`: {row['top_method']}, "
            f"shadow={row['shadow_score']:.3f}, n={int(row['n_members'])}. "
            f"{row['expected_incompleteness_direction']}."
        )
    lines.extend(
        [
            "",
            "Estos resultados no prueban planetas faltantes. Senalan regiones topologicas submuestreadas o posibles fronteras de seleccion donde futuras observaciones podrian encontrar vecinos fisico-orbitales similares.",
            "",
            "Conclusion: Mapper permite pasar de una auditoria de sesgo a una priorizacion prudente de vecindarios fisico-orbitales potencialmente incompletos.",
        ]
    )
    text = "\n".join(lines) + "\n"
    (outputs_dir / "interpretation_summary.md").write_text(text, encoding="utf-8")
    return text


def _primary_edge_table(mapper_outputs_dir: Path, primary_config_id: str, warnings: list[str]) -> pd.DataFrame:
    return load_edge_table(mapper_outputs_dir, primary_config_id, warnings)


def main() -> None:
    args = parse_args()
    config = load_shadow_config(path=args.config, overrides={"seed": args.seed})
    mapper_outputs_dir = resolved_mapper_outputs_dir(config)
    audit_outputs_dir = resolved_audit_outputs_dir(config)
    shadow_outputs_dir = resolved_shadow_outputs_dir(config)
    latex_dir = ensure_latex_dir(resolved_latex_dir(config))
    output_tree = ensure_output_tree(shadow_outputs_dir)
    warnings: list[str] = []
    input_paths: dict[str, str] = {}
    config_metadata: dict[str, dict[str, object]] = {}

    available_config_ids = discover_available_config_ids(mapper_outputs_dir)
    selected_config_ids = []
    for config_id in config.all_config_ids():
        if config_id in available_config_ids:
            selected_config_ids.append(config_id)
        else:
            warnings.append(f"WARNING: no existe la configuracion {config_id}; se omite.")
    if not selected_config_ids:
        raise RuntimeError("No se encontro ninguna configuracion Mapper disponible para sombra observacional.")

    profile_frames: list[pd.DataFrame] = []
    for config_id in selected_config_ids:
        graph_payload = load_graph_payload(mapper_outputs_dir, config_id)
        graph_config = graph_payload.get("config", {})
        input_method = str(graph_config.get("input_method", "iterative"))
        physical_path, physical_df = load_physical_catalog(config.physical_csv_path, input_method=input_method)
        input_paths[f"{config_id}_physical_catalog"] = repo_relative(physical_path)
        node_table = load_node_table(mapper_outputs_dir, config_id)
        edge_table = load_edge_table(mapper_outputs_dir, config_id, warnings)
        membership, membership_source = load_membership_with_catalog(
            mapper_outputs_dir=mapper_outputs_dir,
            audit_outputs_dir=audit_outputs_dir,
            shadow_metadata_dir=output_tree["metadata"],
            config_id=config_id,
            physical_df=physical_df,
            warnings=warnings,
        )
        input_paths[f"{config_id}_membership"] = str(membership_source)
        profiles, metadata = build_node_shadow_profiles(
            config_id=config_id,
            membership=membership,
            node_table=node_table,
            edge_table=edge_table,
            requested_physical_variables=config.physical_variables,
            peripheral_degree_threshold=config.peripheral_degree_threshold,
            peripheral_component_max_nodes=config.peripheral_component_max_nodes,
            epsilon=config.epsilon,
            warnings=warnings,
        )
        profiles = compute_shadow_scores(profiles, has_imputation=bool(metadata["has_imputation"]))
        profiles = classify_shadow_nodes(
            profiles,
            percentile=config.shadow_percentile,
            imputation_threshold=config.imputation_threshold,
            min_members=config.min_high_confidence_members,
        )
        profiles = add_interpretations(profiles)
        profile_frames.append(profiles)
        config_metadata[config_id] = metadata

    node_metrics = pd.concat(profile_frames, ignore_index=True) if profile_frames else pd.DataFrame()
    component_summary = build_component_summary(node_metrics)
    method_summary = build_method_summary(node_metrics)
    config_summary = build_config_comparison(node_metrics)
    top_candidates = build_top_candidates(node_metrics, config.top_n_candidates)

    tables_dir = output_tree["tables"]
    write_csv(node_metrics, tables_dir / "node_observational_shadow_metrics.csv")
    write_csv(top_candidates, tables_dir / "top_shadow_candidates.csv")
    write_csv(component_summary, tables_dir / "component_shadow_summary.csv")
    write_csv(method_summary, tables_dir / "method_shadow_summary.csv")
    write_csv(config_summary, tables_dir / "config_shadow_comparison.csv")

    primary_metrics = node_metrics[node_metrics["config_id"] == config.primary_config_id].copy()
    primary_edges = _primary_edge_table(mapper_outputs_dir, config.primary_config_id, warnings)
    figures_dir = output_tree["figures"]
    plot_graph_by_shadow_score(primary_metrics, primary_edges, figures_dir / "orbital_graph_by_shadow_score.pdf", seed=config.seed)
    plot_graph_by_shadow_class(primary_metrics, primary_edges, figures_dir / "orbital_graph_by_shadow_class.pdf", seed=config.seed)
    plot_scatter(
        primary_metrics,
        "mean_imputation_fraction",
        "shadow_score",
        "Sombra observacional vs imputacion media",
        "Fraccion media de imputacion",
        "shadow_score",
        figures_dir / "orbital_shadow_vs_imputation.pdf",
    )
    plot_scatter(
        primary_metrics,
        "n_members",
        "shadow_score",
        "Sombra observacional vs tamano de nodo",
        "n_members",
        "shadow_score",
        figures_dir / "orbital_shadow_vs_node_size.pdf",
    )
    plot_scatter(
        primary_metrics,
        "method_l1_boundary",
        "shadow_score",
        "Contraste metodo-vecindario vs sombra observacional",
        "method_l1_boundary",
        "shadow_score",
        figures_dir / "orbital_method_boundary_vs_shadow.pdf",
        color_by="top_method",
    )
    plot_scatter(
        primary_metrics,
        "physical_neighbor_distance",
        "shadow_score",
        "Distancia fisico-orbital al vecindario vs sombra",
        "physical_neighbor_distance",
        "shadow_score",
        figures_dir / "orbital_physical_neighbor_distance_vs_shadow.pdf",
    )
    plot_top_candidates(top_candidates[top_candidates["config_id"] == config.primary_config_id], figures_dir / "orbital_top_shadow_candidates.pdf")
    plot_shadow_by_method(primary_metrics, figures_dir / "orbital_shadow_by_method.pdf")
    plot_component_summary(component_summary, figures_dir / "orbital_shadow_component_summary.pdf", config.primary_config_id)
    plot_config_comparison(config_summary, figures_dir / "config_shadow_comparison.pdf")

    expected_figures = [
        "orbital_graph_by_shadow_score.pdf",
        "orbital_graph_by_shadow_class.pdf",
        "orbital_shadow_vs_imputation.pdf",
        "orbital_shadow_vs_node_size.pdf",
        "orbital_method_boundary_vs_shadow.pdf",
        "orbital_physical_neighbor_distance_vs_shadow.pdf",
        "orbital_top_shadow_candidates.pdf",
        "orbital_shadow_by_method.pdf",
        "orbital_shadow_component_summary.pdf",
        "config_shadow_comparison.pdf",
    ]
    for filename in expected_figures:
        path = figures_dir / filename
        if not path.exists():
            placeholder_figure(path, filename, "La figura no pudo generarse con los datos disponibles.")

    interpretation_text = _write_interpretation(shadow_outputs_dir, node_metrics, top_candidates)
    _write_readme(shadow_outputs_dir, selected_config_ids)
    write_latex_report(latex_dir)

    manifest = {
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "commit_hash": git_commit_hash(),
        "configuration": config.to_dict(),
        "input_paths": input_paths,
        "output_paths": {
            "root": repo_relative(shadow_outputs_dir),
            "figures": repo_relative(figures_dir),
            "tables": repo_relative(tables_dir),
            "latex": repo_relative(latex_dir / "main.tex"),
        },
        "mapper_configurations_analyzed": selected_config_ids,
        "variables_used": {
            "physical_variables": config.physical_variables,
            "available_by_config": {key: value.get("physical_variables_available", []) for key, value in config_metadata.items()},
        },
        "shadow_parameters": {
            "epsilon": config.epsilon,
            "shadow_percentile": config.shadow_percentile,
            "imputation_threshold": config.imputation_threshold,
            "min_high_confidence_members": config.min_high_confidence_members,
        },
        "n_nodes_analyzed": int(len(node_metrics)),
        "n_high_confidence_shadow": int((node_metrics["shadow_class"] == "high_confidence_shadow").sum()) if not node_metrics.empty else 0,
        "warnings": warnings,
    }
    write_json(shadow_outputs_dir / "run_manifest.json", manifest)
    write_json(output_tree["metadata"] / "analysis_metadata.json", {"config_metadata": config_metadata})
    save_log(output_tree["logs"] / "observational_shadow.log", warnings)

    print("Analisis de sombra observacional completado.")
    print(f"Configuraciones analizadas: {', '.join(selected_config_ids)}")
    print(f"Outputs: {shadow_outputs_dir}")
    print(f"LaTeX: {latex_dir / 'main.tex'}")
    print(interpretation_text.splitlines()[2] if interpretation_text else "")


if __name__ == "__main__":
    main()

