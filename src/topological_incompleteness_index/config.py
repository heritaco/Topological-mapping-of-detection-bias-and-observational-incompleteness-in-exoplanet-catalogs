from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR, resolve_repo_path


@dataclass
class ProjectSection:
    name: str = "topological_incompleteness_index"
    random_state: int = 42


@dataclass
class InputsSection:
    dataset: str | None = None
    membership: str | None = None
    edges: str | None = None
    node_shadow_metrics: str = "outputs/observational_shadow/tables/node_observational_shadow_metrics.csv"
    top_shadow_candidates: str = "outputs/observational_shadow/tables/top_shadow_candidates.csv"
    node_method_metrics: str = "outputs/observational_bias_audit/tables/node_method_bias_metrics.csv"
    node_method_fraction_matrix: str = "outputs/observational_bias_audit/tables/node_method_fraction_matrix.csv"


@dataclass
class OutputsSection:
    base_dir: str = "outputs/topological_incompleteness_index"
    tables_dir: str = "outputs/topological_incompleteness_index/tables"
    figures_dir: str = "outputs/topological_incompleteness_index/figures_pdf"
    metadata_dir: str = "outputs/topological_incompleteness_index/metadata"
    logs_dir: str = "outputs/topological_incompleteness_index/logs"
    latex_dir: str = "latex/04_topological_incompleteness"


@dataclass
class AnalysisSection:
    config_id: str = "orbital_pca2_cubes10_overlap0p35"
    r3_variables: dict[str, str] = field(
        default_factory=lambda: {"mass": "pl_bmasse", "period": "pl_orbper", "semimajor": "pl_orbsmax"}
    )
    discovery_method_column: str = "discoverymethod"
    planet_name_column: str = "pl_name"
    epsilon: float = 1.0e-9
    min_r3_valid_fraction_for_analogs: float = 0.6


@dataclass
class TOISection:
    use_shadow_score: bool = True
    use_r3_imputation: bool = True
    use_physical_continuity: bool = True
    use_network_support: bool = True
    physical_sigma: float = 1.0
    min_node_members: int = 3
    high_priority_quantile: float = 0.90


@dataclass
class ATISection:
    max_anchors_per_node: int = 1
    prefer_method: str = "Radial Velocity"
    prefer_low_imputation: bool = True
    prefer_medoid: bool = True


@dataclass
class NeighborDeficitSection:
    radii: list[str] = field(default_factory=lambda: ["r_node_median", "r_node_q75", "r_kNN"])
    knn_min: int = 3
    knn_max: int = 10
    analog_min_nodes: int = 3
    analog_shadow_quantile_max: float = 0.50
    analog_physical_distance_quantile: float = 0.35
    reference_min_planets: int = 3


@dataclass
class ReportSection:
    make_latex: bool = True
    make_figures: bool = True
    make_summary: bool = True


@dataclass
class TopologicalIncompletenessConfig:
    project: ProjectSection = field(default_factory=ProjectSection)
    inputs: InputsSection = field(default_factory=InputsSection)
    outputs: OutputsSection = field(default_factory=OutputsSection)
    analysis: AnalysisSection = field(default_factory=AnalysisSection)
    toi: TOISection = field(default_factory=TOISection)
    ati: ATISection = field(default_factory=ATISection)
    neighbor_deficit: NeighborDeficitSection = field(default_factory=NeighborDeficitSection)
    report: ReportSection = field(default_factory=ReportSection)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config_path() -> Path:
    return CONFIGS_DIR / "topological_incompleteness_index.yaml"


def _parse_config_text(text: str, path: Path) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(f"No se pudo parsear {path}. Instala PyYAML o usa JSON.") from exc
        payload = yaml.safe_load(stripped)
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"El archivo de configuracion {path} debe contener un diccionario.")
        return payload


def _section(section_type: type[Any], payload: dict[str, Any] | None) -> Any:
    return section_type(**(payload or {}))


def _validate(config: TopologicalIncompletenessConfig) -> None:
    if config.toi.high_priority_quantile <= 0 or config.toi.high_priority_quantile >= 1:
        raise ValueError("toi.high_priority_quantile debe estar en (0,1).")
    if config.neighbor_deficit.analog_shadow_quantile_max <= 0 or config.neighbor_deficit.analog_shadow_quantile_max > 1:
        raise ValueError("neighbor_deficit.analog_shadow_quantile_max debe estar en (0,1].")
    if config.neighbor_deficit.analog_physical_distance_quantile <= 0 or config.neighbor_deficit.analog_physical_distance_quantile > 1:
        raise ValueError("neighbor_deficit.analog_physical_distance_quantile debe estar en (0,1].")
    if config.neighbor_deficit.knn_min <= 0 or config.neighbor_deficit.knn_max < config.neighbor_deficit.knn_min:
        raise ValueError("neighbor_deficit.knn_min/knn_max invalidos.")
    required_r3_keys = {"mass", "period", "semimajor"}
    if set(config.analysis.r3_variables) != required_r3_keys:
        raise ValueError("analysis.r3_variables debe contener exactamente: mass, period, semimajor.")


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TopologicalIncompletenessConfig:
    config_path = resolve_repo_path(str(path) if path is not None else None, default_config_path())
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    overrides = overrides or {}
    if "analysis" in overrides and isinstance(overrides["analysis"], dict):
        payload.setdefault("analysis", {}).update({k: v for k, v in overrides["analysis"].items() if v is not None})
    root_overrides = {k: v for k, v in overrides.items() if k != "analysis" and v is not None}
    payload.update(root_overrides)
    config = TopologicalIncompletenessConfig(
        project=_section(ProjectSection, payload.get("project")),
        inputs=_section(InputsSection, payload.get("inputs")),
        outputs=_section(OutputsSection, payload.get("outputs")),
        analysis=_section(AnalysisSection, payload.get("analysis")),
        toi=_section(TOISection, payload.get("toi")),
        ati=_section(ATISection, payload.get("ati")),
        neighbor_deficit=_section(NeighborDeficitSection, payload.get("neighbor_deficit")),
        report=_section(ReportSection, payload.get("report")),
    )
    _validate(config)
    return config
