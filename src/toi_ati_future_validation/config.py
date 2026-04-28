from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "toi_ati_future_validation", "random_state": 42},
    "inputs": {
        "regional_toi_scores": "outputs/topological_incompleteness_index/tables/regional_toi_scores.csv",
        "anchor_ati_scores": "outputs/topological_incompleteness_index/tables/anchor_ati_scores.csv",
        "anchor_neighbor_deficits": "outputs/topological_incompleteness_index/tables/anchor_neighbor_deficits.csv",
        "top_anchor_radius_deficit_tables": "outputs/toi_ati_case_anatomy/tables/top_anchor_radius_deficit_tables.csv",
        "top_anchor_radius_deficit_summary": "outputs/toi_ati_case_anatomy/tables/top_anchor_radius_deficit_summary.csv",
        "final_presentation_cases": "outputs/toi_ati_case_anatomy/tables/final_presentation_cases.csv",
        "top_regions_case_anatomy": "outputs/toi_ati_case_anatomy/tables/top_regions_case_anatomy.csv",
        "top_anchors_case_anatomy": "outputs/toi_ati_case_anatomy/tables/top_anchors_case_anatomy.csv",
        "r3_node_geometry": "outputs/topological_incompleteness_index/tables/r3_node_geometry.csv",
        "node_shadow_metrics": "outputs/observational_shadow/tables/node_observational_shadow_metrics.csv",
        "node_method_metrics": "outputs/observational_bias_audit/tables/node_method_bias_metrics.csv",
        "dataset": None,
        "membership": None,
        "edges": None,
    },
    "outputs": {
        "base_dir": "outputs/toi_ati_future_validation",
        "tables_dir": "outputs/toi_ati_future_validation/tables",
        "figures_dir": "outputs/toi_ati_future_validation/figures_pdf",
        "metadata_dir": "outputs/toi_ati_future_validation/metadata",
        "logs_dir": "outputs/toi_ati_future_validation/logs",
        "latex_dir": "latex/09_toi_ati_future_validation",
    },
    "analysis": {
        "config_id": "orbital_pca2_cubes10_overlap0p35",
        "epsilon": 1e-9,
        "top_n_regions": 10,
        "top_n_anchors": 10,
        "exposure_cases_n": 3,
        "candidate_cases_n": 10,
    },
    "deficit_stability": {
        "radii": ["r_kNN", "r_node_median", "r_node_q75"],
        "weak_threshold": 0.10,
        "moderate_threshold": 0.30,
        "strong_threshold": 0.60,
        "require_positive_all_radii_for_stable": True,
    },
    "robust_indices": {
        "make_radius_penalized_ati": True,
        "make_stability_adjusted_ati": True,
        "penalize_negative_large_radius": True,
        "penalize_imputation": True,
        "penalize_small_nodes": True,
        "weight_grid": {
            "toi": [0.5, 1.0, 1.5],
            "deficit": [0.5, 1.0, 1.5],
            "imputation": [0.5, 1.0, 1.5],
            "representativeness": [0.5, 1.0, 1.5],
        },
    },
    "report": {
        "make_figures": True,
        "make_latex": True,
        "make_summary": True,
        "compile_latex": False,
    },
}


@dataclass(frozen=True)
class ProjectConfig:
    raw: Dict[str, Any]
    path: Path | None = None

    def section(self, name: str) -> Dict[str, Any]:
        value = self.raw.get(name, {})
        if not isinstance(value, dict):
            raise TypeError(f"Config section '{name}' must be a mapping.")
        return value

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.section("inputs")

    @property
    def outputs(self) -> Dict[str, Any]:
        return self.section("outputs")

    @property
    def analysis(self) -> Dict[str, Any]:
        return self.section("analysis")

    @property
    def deficit_stability(self) -> Dict[str, Any]:
        return self.section("deficit_stability")

    @property
    def robust_indices(self) -> Dict[str, Any]:
        return self.section("robust_indices")

    @property
    def report(self) -> Dict[str, Any]:
        return self.section("report")


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: Dict[str, Any] | None = None) -> ProjectConfig:
    cfg = dict(DEFAULT_CONFIG)
    cfg_path = Path(path) if path else None
    if cfg_path:
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML config files.")
        with cfg_path.open("r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh) or {}
        if not isinstance(loaded, dict):
            raise TypeError("YAML config must contain a mapping at the top level.")
        cfg = _deep_update(DEFAULT_CONFIG, loaded)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return ProjectConfig(raw=cfg, path=cfg_path)

