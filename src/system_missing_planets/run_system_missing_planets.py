from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import SystemMissingPlanetsConfig, validate_prudent_text
from .detectability import attach_detectability
from .features import (
    assign_priority_class,
    build_candidate_interpretation,
    build_data_quality_score,
    build_system_metadata,
    compute_priority_scores,
    estimate_analog_support_score,
    estimate_candidate_properties,
    summarize_systems,
)
from .gap_model import build_gap_statistics, expand_gap_candidates, find_candidate_gaps
from .io import ensure_output_tree, load_catalog, repo_relative, write_csv, write_json, write_text
from .plotting import make_figures
from .topology_prior import attach_topology_to_candidates, load_topology_resources
from .validation import run_leave_one_out_validation, validation_error_by_system


LOGGER = logging.getLogger("system_missing_planets")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prioritize intra-system orbital intervals for candidate missing planets.")
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["all", "single"], required=True)
    parser.add_argument("--hostname", default=None)
    parser.add_argument("--min-planets-per-system", type=int, default=2)
    parser.add_argument("--min-gap-ratio", type=float, default=2.8)
    parser.add_argument("--high-gap-ratio", type=float, default=5.0)
    parser.add_argument("--max-candidates-per-gap", type=int, default=4)
    parser.add_argument("--n-analogs", type=int, default=35)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--toi-table", default=None)
    parser.add_argument("--ati-table", default=None)
    parser.add_argument("--shadow-table", default=None)
    parser.add_argument("--node-membership-table", default=None)
    parser.add_argument("--make-figures", action="store_true")
    parser.add_argument("--make-latex-summary", action="store_true")
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _build_config_from_args(args: argparse.Namespace) -> SystemMissingPlanetsConfig:
    config = SystemMissingPlanetsConfig(
        catalog=args.catalog,
        output_dir=args.output_dir,
        mode=args.mode,
        hostname=args.hostname,
        min_planets_per_system=args.min_planets_per_system,
        min_gap_ratio=args.min_gap_ratio,
        high_gap_ratio=args.high_gap_ratio,
        max_candidates_per_gap=args.max_candidates_per_gap,
        n_analogs=args.n_analogs,
        random_state=args.random_state,
        toi_table=args.toi_table,
        ati_table=args.ati_table,
        shadow_table=args.shadow_table,
        node_membership_table=args.node_membership_table,
        make_figures=args.make_figures,
        make_latex_summary=args.make_latex_summary,
    )
    config.validate()
    return config


def _latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for source, target in replacements.items():
        value = value.replace(source, target)
    return value


def build_latex_summary(
    candidates: pd.DataFrame,
    system_summary: pd.DataFrame,
    validation_summary: dict[str, Any],
    output_path: Path,
) -> None:
    top_candidates = candidates.sort_values("candidate_priority_score", ascending=False).head(5) if not candidates.empty else pd.DataFrame()
    top_systems = system_summary.sort_values("max_candidate_priority_score", ascending=False).head(5) if not system_summary.empty else pd.DataFrame()
    lines = [
        r"\section{Busqueda intra-sistema de candidatos a planetas no detectados}",
        r"\subsection{Motivacion}",
        "Este modulo traslada la incompletitud topologica global a la escala de sistemas planetarios conocidos. "
        "El objetivo no es afirmar detecciones fisicas directas, sino priorizar intervalos orbitales donde podria existir submuestreo intra-sistema.",
        r"\subsection{Metodologia}",
        "Cada sistema se ordena por periodo orbital y se analizan gaps adyacentes. "
        "Para gaps grandes se generan candidatos sinteticos en escala logaritmica de periodo y semieje mayor, "
        "con apoyo adicional de analogos globales y proxies de detectabilidad por transito o velocidad radial.",
        r"\subsection{Uso de resultados previos TOI/ATI/sombra}",
        "Cuando existen tablas previas, el modulo incorpora TOI regional, ATI por planeta ancla y sombra observacional por nodo Mapper. "
        "Esto convierte la topologia global mas la sombra observacional mas la arquitectura intra-sistema en un ranking de intervalos orbitales prioritarios.",
        r"\subsection{Resultados principales}",
        "Los resultados priorizan candidatos topologicos y regiones sugeridas para seguimiento observacional. "
        "No se detectan planetas reales ni se afirma un conteo verdadero de planetas ausentes.",
    ]
    if top_systems.empty:
        lines.append("No se identificaron sistemas con candidatos evaluables en la corrida actual.")
    else:
        lines.extend([r"\begin{itemize}"])
        for _, row in top_systems.iterrows():
            lines.append(
                rf"\item {_latex_escape(row['hostname'])}: max score={float(pd.to_numeric(pd.Series([row['max_candidate_priority_score']]), errors='coerce').fillna(0.0).iloc[0]):.3f}, "
                rf"n candidatos={int(row['n_candidate_missing_planets'])}."
            )
        lines.extend([r"\end{itemize}"])
    if not top_candidates.empty:
        lines.extend([r"\paragraph{Top candidatos}"])
        lines.extend([r"\begin{itemize}"])
        for _, row in top_candidates.iterrows():
            lines.append(
                rf"\item {_latex_escape(row['hostname'])}: intervalo entre {_latex_escape(row['inner_planet'])} y {_latex_escape(row['outer_planet'])}, "
                rf"periodo sugerido={float(row['candidate_period_days']):.3f} dias, clase={_latex_escape(row['candidate_priority_class'])}."
            )
        lines.extend([r"\end{itemize}"])
    lines.extend(
        [
            r"\subsection{Validacion leave-one-out}",
            "La validacion interna retira temporalmente planetas intermedios conocidos y verifica si el modulo vuelve a priorizar candidatos cercanos en periodo orbital.",
            (
                f"Se ejecutaron {int(validation_summary.get('n_holdout_tests', 0) or 0)} pruebas. "
                f"El error mediano en log10(P) fue {float(validation_summary.get('median_abs_logP_error') or np.nan):.3f}, "
                f"con recall dentro de 0.1 dex de {float(validation_summary.get('recall_within_0p1_dex') or np.nan):.3f} "
                f"y dentro de 0.2 dex de {float(validation_summary.get('recall_within_0p2_dex') or np.nan):.3f}."
            ),
            r"\subsection{Limitaciones}",
            "El ranking no modela completitud instrumental ni sustituye analisis de inyeccion-recuperacion. "
            "Las estimaciones de masa, radio y detectabilidad son proxies para priorizacion, no inferencias fisicas definitivas.",
            r"\subsection{Trabajo futuro}",
            "Los siguientes pasos incluyen incorporar completitud instrumental, seguimiento observacional real y validacion cruzada con catalogos futuros.",
        ]
    )
    text = "\n\n".join(lines) + "\n"
    validate_prudent_text(text)
    write_text(text, output_path)


def run_pipeline(config: SystemMissingPlanetsConfig) -> dict[str, Any]:
    config.validate()
    output_tree = ensure_output_tree(config.output_dir)
    catalog_path, catalog = load_catalog(config.catalog, LOGGER)
    system_metadata_full = build_system_metadata(catalog, config.min_planets_per_system)
    LOGGER.info("found %s multi-planet systems", len(system_metadata_full))
    if config.mode == "single":
        if config.hostname not in set(catalog["hostname"].astype(str)):
            raise ValueError(f"No se encontro hostname='{config.hostname}' en el catalogo cargado.")
        target_hosts = {str(config.hostname)}
    else:
        target_hosts = set(system_metadata_full["hostname"].astype(str))
    target_catalog = catalog[catalog["hostname"].astype(str).isin(target_hosts)].copy()
    system_metadata = system_metadata_full[system_metadata_full["hostname"].astype(str).isin(target_hosts)].copy()
    if config.mode == "single" and system_metadata.empty:
        LOGGER.warning(
            "hostname=%s no cumple el umbral multi-planeta para gaps; se generara una salida controlada con la informacion disponible.",
            config.hostname,
        )
        system_metadata = build_system_metadata(target_catalog, min_planets_per_system=1)

    topology = load_topology_resources(
        toi_table=config.toi_table,
        ati_table=config.ati_table,
        shadow_table=config.shadow_table,
        node_membership_table=config.node_membership_table,
        logger=LOGGER,
    )

    gap_stats = build_gap_statistics(catalog, system_metadata_full)
    candidate_gaps = find_candidate_gaps(
        target_catalog,
        system_metadata,
        gap_stats,
        min_gap_ratio=config.min_gap_ratio,
        max_candidates_per_gap=config.max_candidates_per_gap,
    )
    candidates = expand_gap_candidates(candidate_gaps)
    LOGGER.info("generated %s candidate rows before enrichment", len(candidates))
    candidates = attach_topology_to_candidates(candidates, topology)
    candidates = estimate_candidate_properties(candidates, catalog, config.n_analogs)
    candidates["analog_support_score"] = estimate_analog_support_score(candidates, config.n_analogs) if not candidates.empty else pd.Series(dtype=float)
    candidates = attach_detectability(candidates, catalog)
    candidates["data_quality_score"] = build_data_quality_score(candidates, catalog) if not candidates.empty else pd.Series(dtype=float)
    if not candidates.empty:
        candidates["candidate_priority_score"] = compute_priority_scores(candidates, config.weights.to_dict())
        candidates["candidate_priority_class"] = assign_priority_class(candidates, config.high_gap_ratio)
        candidates["interpretation"] = candidates.apply(build_candidate_interpretation, axis=1)
    else:
        candidates["candidate_priority_score"] = pd.Series(dtype=float)
        candidates["candidate_priority_class"] = pd.Series(dtype="string")
        candidates["interpretation"] = pd.Series(dtype="string")

    validation, validation_summary = run_leave_one_out_validation(
        target_catalog,
        system_metadata,
        gap_stats,
        min_gap_ratio=config.min_gap_ratio,
        max_candidates_per_gap=config.max_candidates_per_gap,
    )
    validation_by_system = validation_error_by_system(validation)
    candidates = candidates.merge(validation_by_system, on="hostname", how="left") if not candidates.empty else candidates

    adjacency_target = gap_stats.adjacency[gap_stats.adjacency["hostname"].astype(str).isin(target_hosts)].copy()
    system_summary = summarize_systems(system_metadata, candidates, adjacency_target)
    high_priority_candidates = candidates[candidates["candidate_priority_class"].astype(str) == "high"].copy() if not candidates.empty else candidates.copy()
    high_priority_systems = system_summary[system_summary["system_priority_class"].astype(str) == "high"].copy() if not system_summary.empty else system_summary.copy()

    candidate_columns = [
        "hostname",
        "candidate_id",
        "inner_planet",
        "outer_planet",
        "candidate_rank_in_gap",
        "candidate_period_days",
        "candidate_semimajor_au",
        "candidate_position_method",
        "gap_period_ratio",
        "gap_log_width",
        "expected_missing_count",
        "candidate_mass_median",
        "candidate_mass_q16",
        "candidate_mass_q84",
        "candidate_radius_median",
        "candidate_radius_q16",
        "candidate_radius_q84",
        "dominant_discoverymethod_system",
        "inner_discoverymethod",
        "outer_discoverymethod",
        "gap_score",
        "topology_score",
        "analog_support_score",
        "missing_detectability_score",
        "data_quality_score",
        "candidate_priority_score",
        "candidate_priority_class",
        "gap_shadow_score",
        "gap_TOI_score",
        "gap_ATI_score",
        "transit_probability_proxy",
        "transit_depth_proxy",
        "rv_K_proxy",
        "likely_missing_due_to",
        "interpretation",
        "analog_support_count",
        "validation_error_if_available",
    ]
    for column in candidate_columns:
        if column not in candidates.columns:
            candidates[column] = np.nan
    candidates_out = candidates[candidate_columns].sort_values(["candidate_priority_score", "hostname", "candidate_rank_in_gap"], ascending=[False, True, True]) if not candidates.empty else pd.DataFrame(columns=candidate_columns)
    system_summary_out = system_summary[
        [
            "hostname",
            "n_observed_planets",
            "n_candidate_missing_planets",
            "max_gap_period_ratio",
            "max_candidate_priority_score",
            "mean_candidate_priority_score",
            "dominant_discoverymethod_system",
            "best_candidate_id",
            "best_candidate_period_days",
            "best_candidate_semimajor_au",
            "system_priority_class",
        ]
    ] if not system_summary.empty else pd.DataFrame(
        columns=[
            "hostname",
            "n_observed_planets",
            "n_candidate_missing_planets",
            "max_gap_period_ratio",
            "max_candidate_priority_score",
            "mean_candidate_priority_score",
            "dominant_discoverymethod_system",
            "best_candidate_id",
            "best_candidate_period_days",
            "best_candidate_semimajor_au",
            "system_priority_class",
        ]
    )

    write_csv(candidates_out, output_tree["base"] / "candidate_missing_planets_by_system.csv")
    write_csv(system_summary_out, output_tree["base"] / "system_gap_summary.csv")
    write_csv(high_priority_systems, output_tree["base"] / "high_priority_systems.csv")
    write_csv(high_priority_candidates, output_tree["base"] / "high_priority_candidates.csv")
    write_csv(validation, output_tree["base"] / "system_missing_planets_validation.csv")
    write_json(validation_summary, output_tree["base"] / "system_missing_planets_validation_summary.json")
    write_json(
        {
            "config": config.to_dict(),
            "resolved_paths": {
                "catalog": repo_relative(catalog_path),
                **topology.paths,
            },
            "warnings": topology.warnings,
            "n_catalog_planets": int(len(catalog)),
            "n_target_systems": int(len(system_metadata)),
            "n_candidate_rows": int(len(candidates_out)),
        },
        output_tree["base"] / "system_missing_planets_config.json",
    )

    if config.make_figures:
        make_figures(system_summary_out, candidates_out, target_catalog, output_tree["figures"])
    if config.make_latex_summary:
        build_latex_summary(candidates_out, system_summary_out, validation_summary, output_tree["base"] / "system_missing_planets_summary.tex")

    LOGGER.info("wrote outputs to %s", output_tree["base"])
    return {
        "catalog_path": catalog_path,
        "output_dir": output_tree["base"],
        "candidates": candidates_out,
        "system_summary": system_summary_out,
        "validation": validation,
        "validation_summary": validation_summary,
        "topology_paths": topology.paths,
    }


def main() -> None:
    _configure_logging()
    args = parse_args()
    config = _build_config_from_args(args)
    run_pipeline(config)


if __name__ == "__main__":
    main()
