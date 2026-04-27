from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPER_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "mapper"
FINAL_LABELS = ("physical", "observational", "mixed", "weak")


def resolve_outputs_dir(outputs_dir: str | Path | None = None) -> Path:
    path = Path(outputs_dir) if outputs_dir is not None else DEFAULT_MAPPER_OUTPUTS_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required synthesis input is missing: {path}")
    return pd.read_csv(path, low_memory=False)


def _read_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def _num(value: Any, default: float = np.nan) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _max_numeric(row: pd.Series, columns: list[str]) -> float:
    values = [_num(row.get(column)) for column in columns]
    values = [value for value in values if np.isfinite(value)]
    return max(values) if values else np.nan


def _ensure_str_id(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    if column in frame.columns:
        frame = frame.copy()
        frame[column] = frame[column].astype(str)
    return frame


def _selected_config_ids(main_selection: pd.DataFrame) -> list[str]:
    if "config_id" not in main_selection.columns:
        raise ValueError("main_graph_selection.csv must contain `config_id`.")
    config_ids = main_selection["config_id"].dropna().astype(str).tolist()
    if not config_ids:
        raise ValueError("main_graph_selection.csv contains no selected config IDs.")
    return config_ids


def _p_value_score(p_value: float) -> float:
    if not np.isfinite(p_value):
        return 0.0
    if p_value <= 0.01:
        return 1.0
    if p_value <= 0.05:
        return 0.85
    if p_value <= 0.10:
        return 0.65
    return 0.0


def _physical_evidence_score(row: pd.Series) -> float:
    radius_purity = _num(row.get("radius_class_purity"))
    orbit_purity = _max_numeric(
        row,
        ["frac_orbit_short_period", "frac_orbit_intermediate_period", "frac_orbit_long_period"],
    )
    thermal_purity = _max_numeric(
        row,
        ["frac_thermal_very_hot", "frac_thermal_hot", "frac_thermal_warm", "frac_thermal_cool"],
    )
    population_purity = _max_numeric(
        row,
        [
            "frac_hot_jupiter_candidate",
            "frac_super_earth_candidate",
            "frac_sub_neptune_candidate",
            "frac_rocky_candidate",
            "frac_long_period_giant_candidate",
        ],
    )
    fallback_values: list[float] = []
    for column in ["radius_class_dominant", "orbit_class_dominant", "thermal_class_dominant"]:
        value = str(row.get(column, "") or "")
        if value and value.lower() not in {"unknown", "unknown_mixed", "nan"}:
            fallback_values.append(0.55)
    candidates = [
        radius_purity,
        orbit_purity,
        thermal_purity,
        population_purity,
        *fallback_values,
    ]
    candidates = [value for value in candidates if np.isfinite(value)]
    return _clip01(max(candidates) if candidates else 0.0)


def _observational_bias_score(row: pd.Series) -> float:
    method_fraction = _num(row.get("dominant_discoverymethod_fraction"), 0.0)
    method_entropy = _num(row.get("discoverymethod_entropy"), np.nan)
    facility_fraction = _num(row.get("dominant_disc_facility_fraction"), 0.0)
    year_iqr = _num(row.get("disc_year_iqr"), np.nan)
    z_score = _num(row.get("discoverymethod_enrichment_z"), np.nan)
    p_value = _num(row.get("discoverymethod_enrichment_p"), np.nan)
    js_divergence = _num(row.get("discoverymethod_js_divergence_vs_global"), 0.0)

    entropy_score = 0.0 if not np.isfinite(method_entropy) else _clip01(1.0 - method_entropy / 2.5)
    z_score_component = 0.0 if not np.isfinite(z_score) else _clip01(z_score / 5.0)
    p_component = _p_value_score(p_value)
    facility_component = facility_fraction if facility_fraction >= 0.80 else 0.0
    year_component = 0.0
    if np.isfinite(year_iqr):
        if year_iqr <= 2:
            year_component = 0.75
        elif year_iqr <= 5:
            year_component = 0.50
    js_component = _clip01(js_divergence / 0.35)

    return _clip01(
        max(
            method_fraction,
            entropy_score,
            z_score_component,
            p_component,
            facility_component,
            year_component,
            js_component,
        )
    )


def _imputation_risk_score(row: pd.Series) -> float:
    mean_imp = _num(row.get("mean_imputation_fraction"), 0.0)
    max_imp = _num(row.get("max_imputation_fraction"), mean_imp)
    n_members = _num(row.get("n_members"), 0.0)
    mean_component = _clip01(mean_imp / 0.30)
    max_component = _clip01(max_imp / 0.75)
    if n_members < 5:
        size_component = 0.85
    elif n_members < 10:
        size_component = 0.50
    elif n_members < 20:
        size_component = 0.25
    else:
        size_component = 0.0
    return _clip01(max(mean_component, max_component, size_component))


def _is_significant(row: pd.Series) -> bool:
    p_value = _num(row.get("discoverymethod_enrichment_p"), np.nan)
    z_score = _num(row.get("discoverymethod_enrichment_z"), np.nan)
    return (np.isfinite(p_value) and p_value <= 0.05) or (np.isfinite(z_score) and z_score >= 2.0)


def _classify(row: pd.Series) -> tuple[str, str]:
    physical = _num(row.get("physical_evidence_score"), 0.0)
    observational = _num(row.get("observational_bias_score"), 0.0)
    risk = _num(row.get("imputation_risk_score"), 0.0)
    mean_imp = _num(row.get("mean_imputation_fraction"), 0.0)
    max_imp = _num(row.get("max_imputation_fraction"), mean_imp)
    n_members = _num(row.get("n_members"), 0.0)
    facility_fraction = _num(row.get("dominant_disc_facility_fraction"), 0.0)
    significant = _is_significant(row)

    physical_coherent = physical >= 0.65
    high_bias = observational >= 0.75 or significant
    low_bias = observational < 0.60 and not significant and facility_fraction < 0.85
    high_imputation = mean_imp >= 0.30 or max_imp >= 0.75
    too_small = n_members < 5

    if high_imputation or too_small:
        if physical_coherent and high_bias and risk < 0.85:
            label = "mixed"
        elif high_bias and risk < 0.85:
            label = "observational"
        else:
            label = "weak"
    elif physical_coherent and high_bias:
        label = "mixed"
    elif high_bias and physical < 0.65:
        label = "observational"
    elif physical_coherent and low_bias and risk < 0.50:
        label = "physical"
    else:
        label = "weak"

    if label == "physical":
        confidence = "high" if physical >= 0.80 and observational < 0.45 and risk < 0.30 else "medium"
    elif label == "observational":
        confidence = "high" if observational >= 0.85 and risk < 0.70 else "medium"
    elif label == "mixed":
        confidence = "high" if physical >= 0.75 and observational >= 0.85 and risk < 0.70 else "medium"
    else:
        confidence = "high" if risk >= 0.85 or n_members < 5 else "low"
    return label, confidence


def _rationale(row: pd.Series) -> str:
    parts: list[str] = []
    parts.append(f"physical_score={_num(row.get('physical_evidence_score'), 0.0):.2f}")
    parts.append(f"bias_score={_num(row.get('observational_bias_score'), 0.0):.2f}")
    parts.append(f"imputation_risk={_num(row.get('imputation_risk_score'), 0.0):.2f}")
    method = str(row.get("dominant_discoverymethod", "") or "")
    method_fraction = _num(row.get("dominant_discoverymethod_fraction"), np.nan)
    if method and np.isfinite(method_fraction):
        parts.append(f"{method} fraction={method_fraction:.2f}")
    p_value = _num(row.get("discoverymethod_enrichment_p"), np.nan)
    z_score = _num(row.get("discoverymethod_enrichment_z"), np.nan)
    if np.isfinite(z_score):
        parts.append(f"enrichment_z={z_score:.2f}")
    if np.isfinite(p_value):
        parts.append(f"p={p_value:.3f}")
    radius = str(row.get("radius_class_dominant", "") or "")
    if radius:
        parts.append(f"radius={radius}")
    return "; ".join(parts)


def _report_language(row: pd.Series) -> str:
    label = str(row.get("final_label"))
    region = f"{row.get('region_type')} `{row.get('region_id')}` in `{row.get('config_id')}`"
    method = str(row.get("dominant_discoverymethod", "") or "unknown discovery method")
    radius = str(row.get("radius_class_dominant", "") or "no dominant radius class")
    orbit = str(row.get("orbit_class_dominant", "") or "no dominant orbit class")
    if label == "physical":
        return (
            f"{region} is best treated as physically interpretable: it is coherent in {radius}/{orbit}, "
            "has low imputation risk, and does not show strong discovery-method enrichment."
        )
    if label == "observational":
        return (
            f"{region} is observationally suspicious: it is dominated by {method} metadata and should not be "
            "read as an astrophysical population without additional controls."
        )
    if label == "mixed":
        return (
            f"{region} has both physical coherence ({radius}/{orbit}) and discovery-method enrichment ({method}); "
            "it should be discussed as mixed evidence rather than a clean physical structure."
        )
    return (
        f"{region} is weak evidence: size, imputation, or ambiguous physical/observational signals make it unsuitable "
        "for strong interpretation."
    )


def _canonical_region_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "config_id",
        "feature_space",
        "lens",
        "region_type",
        "region_id",
        "n_members",
        "final_label",
        "confidence",
        "physical_evidence_score",
        "observational_bias_score",
        "imputation_risk_score",
        "dominant_discoverymethod",
        "dominant_discoverymethod_fraction",
        "discoverymethod_enrichment_z",
        "discoverymethod_enrichment_p",
        "mean_imputation_fraction",
        "radius_class_dominant",
        "radius_class_purity",
        "orbit_class_dominant",
        "thermal_class_dominant",
        "rationale_short",
        "recommended_report_language",
    ]
    for column in required:
        if column not in frame.columns:
            frame[column] = np.nan
    extras = [column for column in frame.columns if column not in required]
    return frame[required + extras]


def _node_synthesis(
    selected_ids: list[str],
    node_physical: pd.DataFrame,
    node_bias: pd.DataFrame,
    enrichment: pd.DataFrame,
) -> pd.DataFrame:
    node_physical = _ensure_str_id(node_physical, "node_id")
    node_bias = _ensure_str_id(node_bias, "node_id")
    enrichment = _ensure_str_id(enrichment, "node_id")
    frame = node_bias[node_bias["config_id"].astype(str).isin(selected_ids)].copy()
    if frame.empty:
        return pd.DataFrame()
    physical_cols = [
        "config_id",
        "node_id",
        "frac_orbit_short_period",
        "frac_orbit_intermediate_period",
        "frac_orbit_long_period",
        "frac_thermal_very_hot",
        "frac_thermal_hot",
        "frac_thermal_warm",
        "frac_thermal_cool",
        "frac_hot_jupiter_candidate",
        "frac_super_earth_candidate",
        "frac_sub_neptune_candidate",
        "frac_rocky_candidate",
        "frac_long_period_giant_candidate",
        "candidate_population_top",
        "physically_derived_fraction",
        "frac_any_imputed",
    ]
    physical_cols = [column for column in physical_cols if column in node_physical.columns]
    frame = frame.merge(
        node_physical[physical_cols].drop_duplicates(["config_id", "node_id"]) if physical_cols else pd.DataFrame(),
        on=["config_id", "node_id"],
        how="left",
    )
    enrich_cols = [
        "config_id",
        "node_id",
        "enrichment_z",
        "empirical_p_value",
        "n_perm",
        "seed",
    ]
    enrich_cols = [column for column in enrich_cols if column in enrichment.columns]
    frame = frame.merge(
        enrichment[enrich_cols].drop_duplicates(["config_id", "node_id"]) if enrich_cols else pd.DataFrame(),
        on=["config_id", "node_id"],
        how="left",
    )
    frame["region_type"] = "node"
    frame["region_id"] = frame["node_id"]
    frame["discoverymethod_enrichment_z"] = pd.to_numeric(frame.get("enrichment_z"), errors="coerce")
    frame["discoverymethod_enrichment_p"] = pd.to_numeric(frame.get("empirical_p_value"), errors="coerce")
    return frame


def _component_synthesis(
    selected_ids: list[str],
    component_summary: pd.DataFrame,
    component_bias: pd.DataFrame,
) -> pd.DataFrame:
    component_summary = _ensure_str_id(component_summary, "component_id")
    component_bias = _ensure_str_id(component_bias, "component_id")
    frame = component_bias[component_bias["config_id"].astype(str).isin(selected_ids)].copy()
    if frame.empty:
        return pd.DataFrame()
    summary_cols = [
        "config_id",
        "component_id",
        "beta_1_component",
        "dominant_candidate_population",
        "physically_derived_fraction",
        "caution_level",
    ]
    summary_cols = [column for column in summary_cols if column in component_summary.columns]
    frame = frame.merge(
        component_summary[summary_cols].drop_duplicates(["config_id", "component_id"]) if summary_cols else pd.DataFrame(),
        on=["config_id", "component_id"],
        how="left",
    )
    frame["region_type"] = "component"
    frame["region_id"] = frame["component_id"]
    frame["discoverymethod_enrichment_z"] = np.nan
    frame["discoverymethod_enrichment_p"] = np.nan
    return frame


def _apply_classification(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    out["physical_evidence_score"] = out.apply(_physical_evidence_score, axis=1)
    out["observational_bias_score"] = out.apply(_observational_bias_score, axis=1)
    out["imputation_risk_score"] = out.apply(_imputation_risk_score, axis=1)
    labels = out.apply(_classify, axis=1)
    out["final_label"] = [item[0] for item in labels]
    out["confidence"] = [item[1] for item in labels]
    out["rationale_short"] = out.apply(_rationale, axis=1)
    out["recommended_report_language"] = out.apply(_report_language, axis=1)
    return _canonical_region_columns(out)


def _fmt(value: Any, digits: int = 3) -> str:
    number = _num(value)
    return "NA" if not np.isfinite(number) else f"{number:.{digits}f}"


def _table(frame: pd.DataFrame, columns: list[str], limit: int = 8) -> str:
    if frame.empty:
        return "None available."
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.head(limit).to_dict(orient="records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            values.append(_fmt(value) if isinstance(value, float) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _label_counts(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    counts = (
        frame.groupby(["config_id", "region_type", "final_label"])
        .size()
        .reset_index(name="n_regions")
        .sort_values(["config_id", "region_type", "final_label"])
    )
    return counts


def build_synthesis_markdown(synthesis: pd.DataFrame, permutation_null: pd.DataFrame) -> str:
    counts = _label_counts(synthesis)
    node_frame = synthesis[synthesis["region_type"] == "node"].copy()
    component_frame = synthesis[synthesis["region_type"] == "component"].copy()
    physically = synthesis[synthesis["final_label"] == "physical"].sort_values(
        ["confidence", "physical_evidence_score", "observational_bias_score"],
        ascending=[True, False, True],
    )
    suspicious = synthesis[synthesis["final_label"] == "observational"].sort_values(
        ["observational_bias_score", "discoverymethod_enrichment_z"],
        ascending=[False, False],
        na_position="last",
    )
    mixed = synthesis[synthesis["final_label"] == "mixed"].sort_values(
        ["physical_evidence_score", "observational_bias_score"],
        ascending=[False, False],
        na_position="last",
    )
    weak = synthesis[synthesis["final_label"] == "weak"].sort_values(
        ["imputation_risk_score", "n_members"],
        ascending=[False, True],
        na_position="last",
    )
    orbital = synthesis[synthesis["config_id"] == "orbital_pca2_cubes10_overlap0p35"]
    n_perm_values = sorted(set(pd.to_numeric(permutation_null.get("n_perm", pd.Series(dtype=float)), errors="coerce").dropna().astype(int).tolist()))
    n_perm_text = ", ".join(str(value) for value in n_perm_values) if n_perm_values else "not available"
    final_perm_note = (
        "The available permutation audit appears to be the final 1000-permutation run."
        if n_perm_values and max(n_perm_values) >= 1000
        else "The available permutation audit is below 1000 permutations; treat p-values as screening-level until the final 1000-permutation run is written."
    )

    conclusion = (
        "The final region synthesis classifies selected Mapper regions using four evidence streams: topology-derived regions, "
        "imputation confidence, heuristic physical coherence, and discovery-method enrichment. The strongest conclusion is not "
        "that Mapper has discovered causal astrophysical classes, but that some regions are physically interpretable, some are "
        "observationally suspicious, and several are mixed. This classification should be read as reproducible evidence triage "
        "for scientific follow-up rather than as proof of physical taxonomy."
    )

    lines = [
        "# Final Region Synthesis",
        "",
        "This file classifies selected `pca2` Mapper nodes and connected components as `physical`, `observational`, `mixed`, or `weak` using explicit rule-based evidence.",
        "",
        f"Permutation `n_perm` values detected: {n_perm_text}. {final_perm_note}",
        "",
        "## Overall Synthesis",
        "",
        _table(counts, ["config_id", "region_type", "final_label", "n_regions"], limit=80),
        "",
        "## Orbital Mapper Synthesis",
        "",
        _table(
            orbital.sort_values(["final_label", "observational_bias_score"], ascending=[True, False]),
            [
                "region_type",
                "region_id",
                "final_label",
                "confidence",
                "n_members",
                "dominant_discoverymethod",
                "dominant_discoverymethod_fraction",
                "physical_evidence_score",
                "observational_bias_score",
                "imputation_risk_score",
            ],
            limit=12,
        ),
        "",
        "## Most Physically Interpretable Regions",
        "",
        _table(
            physically,
            [
                "config_id",
                "region_type",
                "region_id",
                "confidence",
                "physical_evidence_score",
                "observational_bias_score",
                "mean_imputation_fraction",
                "radius_class_dominant",
                "orbit_class_dominant",
            ],
        ),
        "",
        "## Observationally Suspicious Regions",
        "",
        _table(
            suspicious,
            [
                "config_id",
                "region_type",
                "region_id",
                "confidence",
                "dominant_discoverymethod",
                "dominant_discoverymethod_fraction",
                "discoverymethod_enrichment_z",
                "discoverymethod_enrichment_p",
                "observational_bias_score",
            ],
        ),
        "",
        "## Mixed Regions",
        "",
        _table(
            mixed,
            [
                "config_id",
                "region_type",
                "region_id",
                "confidence",
                "physical_evidence_score",
                "observational_bias_score",
                "dominant_discoverymethod",
                "radius_class_dominant",
            ],
        ),
        "",
        "## Regions Not To Overinterpret",
        "",
        _table(
            weak,
            [
                "config_id",
                "region_type",
                "region_id",
                "confidence",
                "n_members",
                "imputation_risk_score",
                "mean_imputation_fraction",
                "dominant_discoverymethod",
                "physical_evidence_score",
            ],
        ),
        "",
        "## Interpretation Caution",
        "",
        "This is an evidence classification based on topology, imputation, physical coherence, and discovery-method enrichment. It is not causal proof that discovery method created a region, nor proof that a region is an astrophysical class.",
        "",
        "## LaTeX Conclusion Paragraph",
        "",
        conclusion,
        "",
    ]
    return "\n".join(lines)


def synthesize_regions(outputs_dir: str | Path | None = None) -> tuple[pd.DataFrame, Path, Path]:
    root = resolve_outputs_dir(outputs_dir)
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    main_selection = _read_required(tables_dir / "main_graph_selection.csv")
    node_physical = _read_required(tables_dir / "node_physical_interpretation.csv")
    component_summary = _read_required(tables_dir / "component_summary.csv")
    node_bias = _read_required(tables_dir / "node_discovery_bias.csv")
    component_bias = _read_required(tables_dir / "component_discovery_bias.csv")
    enrichment = _read_required(tables_dir / "discoverymethod_enrichment_summary.csv")
    permutation_null = _read_required(tables_dir / "discoverymethod_permutation_null.csv")
    _read_optional(tables_dir / "mapper_node_source_audit.csv")
    _read_optional(tables_dir / "mapper_graph_metrics_all_existing.csv")

    selected_ids = _selected_config_ids(main_selection)
    node_regions = _node_synthesis(selected_ids, node_physical, node_bias, enrichment)
    component_regions = _component_synthesis(selected_ids, component_summary, component_bias)
    synthesis = _apply_classification(pd.concat([node_regions, component_regions], ignore_index=True, sort=False))
    synthesis = synthesis.sort_values(
        ["config_id", "region_type", "final_label", "observational_bias_score", "physical_evidence_score"],
        ascending=[True, True, True, False, False],
        ignore_index=True,
    )

    csv_path = tables_dir / "final_region_synthesis.csv"
    md_path = tables_dir / "final_region_synthesis.md"
    synthesis.to_csv(csv_path, index=False)
    md_path.write_text(build_synthesis_markdown(synthesis, permutation_null), encoding="utf-8")
    return synthesis, csv_path, md_path
