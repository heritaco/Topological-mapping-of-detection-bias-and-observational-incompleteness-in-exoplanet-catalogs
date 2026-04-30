from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import validate_prudent_text
from .io import load_catalog, safe_hostname, write_csv, write_json


LOGGER = logging.getLogger("system_missing_planets.case_studies")

BASE_PHRASE = (
    "El pipeline no detecta planetas reales; prioriza intervalos orbitales donde un candidato no detectado "
    "seria plausible bajo la arquitectura observada, el soporte topologico previo y la dificultad relativa de deteccion."
)

METHOD_COLORS = {
    "Transit": "#1f77b4",
    "Radial Velocity": "#d95f02",
    "Transit Timing Variations": "#2a9d8f",
    "Eclipse Timing Variations": "#6d597a",
    "Mixed": "#6b7280",
    "Unknown": "#6b7280",
}


@dataclass
class CaseSpec:
    case_rank: int
    case_type: str
    preferred_hosts: list[str]
    preferred_methods: list[str]


CASE_SPECS = [
    CaseSpec(1, "top_ranked_visual_case", ["WASP-132", "WASP-47", "TOI-4010", "KOI-142"], ["Transit"]),
    CaseSpec(2, "radial_velocity_like_case", ["HD 27894", "HIP 57274", "HD 153557", "55 Cnc"], ["Radial Velocity"]),
    CaseSpec(3, "mixed_or_fallback_case", ["HAT-P-17", "TOI-4010", "KOI-142", "WASP-47", "HIP 57274"], ["Mixed", "Transit"]),
]


def _canonical_method(value: object) -> str:
    text = str(value or "Unknown")
    if "radial velocity" in text.lower():
        return "Radial Velocity"
    if "transit timing variations" in text.lower():
        return "Transit Timing Variations"
    if "eclipse timing variations" in text.lower():
        return "Eclipse Timing Variations"
    if "transit" in text.lower():
        return "Transit"
    return text if text else "Unknown"


def _select_case(candidates: pd.DataFrame, spec: CaseSpec, used_hosts: set[str]) -> pd.Series:
    eligible = candidates[~candidates["hostname"].astype(str).isin(used_hosts)].copy()
    if eligible.empty:
        raise ValueError("No quedan candidatos elegibles para seleccionar casos.")
    if spec.case_type == "radial_velocity_like_case":
        eligible = eligible[eligible["likely_missing_due_to"].astype(str) == "weak_RV_signal"].copy()
    elif spec.case_type == "top_ranked_visual_case":
        eligible = eligible[eligible["dominant_discoverymethod_system"].astype(str).isin(spec.preferred_methods)].copy()
    elif spec.case_type == "mixed_or_fallback_case":
        mixed_exact = eligible[eligible["dominant_discoverymethod_system"].astype(str) == "Mixed"].copy()
        if not mixed_exact.empty:
            eligible = mixed_exact
        else:
            mixed = eligible[eligible["dominant_discoverymethod_system"].astype(str).isin(spec.preferred_methods)].copy()
            eligible = mixed if not mixed.empty else eligible

    preferred = eligible[eligible["hostname"].astype(str).isin(spec.preferred_hosts)].copy()
    pool = preferred if not preferred.empty else eligible
    pool = pool.sort_values(
        ["candidate_priority_score", "gap_period_ratio", "missing_detectability_score"],
        ascending=[False, False, False],
    )
    return pool.iloc[0]


def _format_range(median: float, q16: float, q84: float, unit: str) -> str:
    if np.isfinite(median) and np.isfinite(q16) and np.isfinite(q84):
        return f"{median:.2f} [{q16:.2f}, {q84:.2f}] {unit}".strip()
    if np.isfinite(median):
        return f"{median:.2f} {unit}".strip()
    return f"NaN {unit}".strip()


def _main_reason(row: pd.Series) -> str:
    method = str(row.get("dominant_discoverymethod_system", "Unknown"))
    likely = str(row.get("likely_missing_due_to", "unknown")).replace("_", " ")
    return (
        f"Gap orbital amplio ({float(row['gap_period_ratio']):.2f}), soporte topologico={float(row['topology_score']):.3f}, "
        f"score final={float(row['candidate_priority_score']):.3f}, con lectura {method} y razon probable de no deteccion: {likely}."
    )


def _followup_text(row: pd.Series, case_type: str) -> str:
    reason = str(row.get("likely_missing_due_to", "unknown"))
    host = str(row["hostname"])
    if case_type == "radial_velocity_like_case" or reason == "weak_RV_signal":
        return (
            f"Priorizar seguimiento RV mas sensible en {host}, junto con un analisis de completitud dinamica y re-evaluacion "
            "de series temporales de velocidad radial en el entorno del periodo candidato."
        )
    if reason in {"shallow_transit_depth", "low_transit_probability", "long_period"}:
        return (
            f"Priorizar fotometria adicional, revision de senales debiles y chequeo de ventanas de transito para {host}; "
            "si hay soporte independiente, combinar con RV de precision o TTV segun disponibilidad."
        )
    return (
        f"Usar {host} como caso para seguimiento mixto: fotometria, RV y analisis de completitud, sin tratar el candidato como deteccion."
    )


def _interpretation_text(row: pd.Series, case_type: str) -> str:
    host = str(row["hostname"])
    inner = str(row["inner_planet"])
    outer = str(row["outer_planet"])
    period = float(row["candidate_period_days"])
    semimajor = float(row["candidate_semimajor_au"])
    score = float(row["candidate_priority_score"])
    method = str(row.get("dominant_discoverymethod_system", "Unknown"))
    likely = str(row.get("likely_missing_due_to", "unknown")).replace("_", " ")

    if case_type == "radial_velocity_like_case":
        text = (
            f"En el sistema {host}, el candidato sintetico aparece dentro de un gap amplio entre {inner} y {outer}. "
            f"La posicion orbital plausible se ubica en P*~{period:.2f} dias y a*~{semimajor:.3f} AU. "
            f"La prioridad observacional (score={score:.3f}) se apoya en el tamano del gap, el soporte de analogos y una lectura "
            f"compatible con dificultad relativa de deteccion por velocidad radial ({likely})."
        )
    elif case_type == "mixed_or_fallback_case":
        text = (
            f"En el sistema {host}, el gap entre {inner} y {outer} es suficientemente amplio para proponer una posicion orbital plausible "
            f"en P*~{period:.2f} dias y a*~{semimajor:.3f} AU. La evidencia debe leerse como una convergencia entre arquitectura "
            f"intra-sistema y soporte topologico previo, con un contexto observacional {method.lower()} y razon probable de no deteccion: {likely}."
        )
    else:
        text = (
            f"En el sistema {host}, el pipeline identifica un gap orbital entre {inner} y {outer}. "
            f"La posicion sintetica prioritaria se ubica en P*~{period:.2f} dias y a*~{semimajor:.3f} AU. "
            f"La lectura es compatible con un candidato no detectado cuya prioridad observacional (score={score:.3f}) aumenta por el gap, "
            f"el soporte topologico previo y una dificultad relativa de deteccion coherente con {likely}."
        )
    validate_prudent_text(text)
    return text


def _plot_case_figure(hostname: str, catalog: pd.DataFrame, all_candidates: pd.DataFrame, selected_candidate: pd.Series, output_path: Path) -> None:
    observed = catalog[catalog["hostname"].astype(str) == hostname].sort_values("pl_orbper").copy()
    system_candidates = all_candidates[all_candidates["hostname"].astype(str) == hostname].sort_values("candidate_period_days").copy()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    observed_methods = observed["discoverymethod"].astype(str).map(_canonical_method)
    observed_colors = observed_methods.map(lambda value: METHOD_COLORS.get(value, METHOD_COLORS["Unknown"]))
    ax.scatter(
        pd.to_numeric(observed["pl_orbper"], errors="coerce"),
        np.ones(len(observed)),
        c=observed_colors.tolist(),
        s=90,
        edgecolor="black",
        linewidth=0.7,
        zorder=3,
    )
    for _, row in observed.iterrows():
        ax.annotate(str(row["pl_name"]), (float(row["pl_orbper"]), 1.03), fontsize=8, rotation=25)

    if not system_candidates.empty:
        ax.scatter(
            pd.to_numeric(system_candidates["candidate_period_days"], errors="coerce"),
            np.full(len(system_candidates), 1.10),
            facecolors="white",
            edgecolors="#6b7280",
            s=90 + 140 * pd.to_numeric(system_candidates["candidate_priority_score"], errors="coerce").fillna(0.0),
            linewidth=1.1,
            zorder=2,
        )
        for _, row in system_candidates.iterrows():
            ax.annotate(f"C{int(row['candidate_rank_in_gap'])}", (float(row["candidate_period_days"]), 1.115), fontsize=8)

    ax.scatter(
        [float(selected_candidate["candidate_period_days"])],
        [1.10],
        marker="*",
        s=320,
        color="#c1121f",
        edgecolor="black",
        linewidth=0.8,
        zorder=4,
        label="Selected synthetic candidate",
    )
    ax.axvspan(
        float(selected_candidate["candidate_period_days"]) / 1.12,
        float(selected_candidate["candidate_period_days"]) * 1.12,
        color="#f4a261",
        alpha=0.14,
        zorder=1,
    )
    ax.set_xscale("log")
    ax.set_yticks([1.0, 1.10])
    ax.set_yticklabels(["Observed", "Candidates"])
    ax.set_xlabel("Orbital period [days]")
    ax.set_title(f"System architecture: {hostname}")
    ax.grid(True, axis="x", alpha=0.25)
    ax.text(
        0.01,
        0.02,
        "Candidates are synthetic prioritization points, not confirmed planets.",
        transform=ax.transAxes,
        fontsize=8,
        color="#444444",
    )
    ax.legend(loc="upper left", fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_three_case_studies(
    *,
    output_dir: Path,
    catalog_hint: str = "data/PSCompPars_imputed_iterative.csv",
) -> tuple[pd.DataFrame, dict[str, object]]:
    candidates = pd.read_csv(output_dir / "high_priority_candidates.csv")
    system_summary = pd.read_csv(output_dir / "system_gap_summary.csv")
    _, catalog = load_catalog(catalog_hint, LOGGER)
    candidates = candidates[candidates["candidate_priority_class"].astype(str) == "high"].copy()
    candidates = candidates[pd.to_numeric(candidates["candidate_period_days"], errors="coerce").notna()].copy()

    selected_rows: list[pd.Series] = []
    used_hosts: set[str] = set()
    for spec in CASE_SPECS:
        row = _select_case(candidates, spec, used_hosts)
        used_hosts.add(str(row["hostname"]))
        row = row.copy()
        row["case_rank"] = spec.case_rank
        row["case_type"] = spec.case_type
        row["scientific_interpretation"] = _interpretation_text(row, spec.case_type)
        row["followup_recommendation"] = _followup_text(row, spec.case_type)
        row["caution_note"] = (
            "Candidate is not a confirmed planet. The interval is prioritized for follow-up, not interpreted as a detection."
        )
        selected_rows.append(row)

    final = pd.DataFrame(selected_rows)
    final = final.merge(
        system_summary[["hostname", "n_observed_planets"]],
        on="hostname",
        how="left",
    )
    keep_columns = [
        "case_rank",
        "hostname",
        "candidate_id",
        "inner_planet",
        "outer_planet",
        "candidate_period_days",
        "candidate_semimajor_au",
        "candidate_mass_median",
        "candidate_mass_q16",
        "candidate_mass_q84",
        "candidate_radius_median",
        "candidate_radius_q16",
        "candidate_radius_q84",
        "gap_period_ratio",
        "gap_log_width",
        "candidate_priority_score",
        "candidate_priority_class",
        "gap_score",
        "topology_score",
        "analog_support_score",
        "missing_detectability_score",
        "data_quality_score",
        "gap_shadow_score",
        "gap_TOI_score",
        "gap_ATI_score",
        "dominant_discoverymethod_system",
        "inner_discoverymethod",
        "outer_discoverymethod",
        "transit_probability_proxy",
        "transit_depth_proxy",
        "rv_K_proxy",
        "likely_missing_due_to",
        "case_type",
        "scientific_interpretation",
        "followup_recommendation",
        "caution_note",
        "n_observed_planets",
        "analog_support_count",
        "validation_error_if_available",
    ]
    final = final[keep_columns].sort_values("case_rank").reset_index(drop=True)

    for _, row in final.iterrows():
        _plot_case_figure(
            hostname=str(row["hostname"]),
            catalog=catalog,
            all_candidates=candidates,
            selected_candidate=row,
            output_path=output_dir / "figures" / f"final_case_{safe_hostname(str(row['hostname']))}.pdf",
        )

    summary_payload = {
        "selected_cases": [
            {
                "hostname": str(row["hostname"]),
                "candidate_id": str(row["candidate_id"]),
                "case_type": str(row["case_type"]),
                "main_reason": _main_reason(row),
                "followup_recommendation": str(row["followup_recommendation"]),
                "caution": "Candidate is not a confirmed planet.",
            }
            for _, row in final.iterrows()
        ],
        "selection_logic": (
            "Se priorizaron casos high con score alto, gap amplio, estimaciones fisicas disponibles, interpretacion de detectabilidad "
            "y variedad narrativa entre un caso visual top, un caso radial-velocity-like y un caso mixed/fallback."
        ),
        "global_caution": "The module prioritizes orbital intervals; it does not detect planets.",
        "base_phrase": BASE_PHRASE,
        "supporting_paths": {
            "toi_table": "outputs/topological_incompleteness_index/tables/regional_toi_scores.csv",
            "ati_table": "outputs/topological_incompleteness_index/tables/anchor_ati_scores.csv",
            "shadow_table": "outputs/observational_shadow/tables/node_observational_shadow_metrics.csv",
            "node_membership_table": "outputs/observational_shadow/metadata/membership_with_shadow_inputs_orbital_pca2_cubes10_overlap0p35.csv",
        },
    }
    return final, summary_payload


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_dir = Path("outputs/system_missing_planets")
    final, payload = build_three_case_studies(output_dir=output_dir)
    write_csv(final, output_dir / "final_three_case_studies.csv")
    write_json(payload, output_dir / "final_three_case_studies_summary.json")
    LOGGER.info("Wrote %s selected case studies to %s", len(final), output_dir)


if __name__ == "__main__":
    main()
