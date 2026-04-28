from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR, LATEX_DIR, OUTPUTS_DIR, resolve_repo_path


DEFAULT_REQUESTED_NODES = ["cube17_cluster2", "cube26_cluster0", "cube24_cluster0"]
R3_VARIABLES = ["pl_bmasse", "pl_orbper", "pl_orbsmax"]


@dataclass
class LocalShadowCaseConfig:
    mapper_outputs_dir: str = "outputs/mapper"
    audit_outputs_dir: str = "outputs/observational_bias_audit"
    shadow_outputs_dir: str = "outputs/observational_shadow"
    local_outputs_dir: str = "outputs/local_shadow_case_studies"
    latex_dir: str = "latex/local_shadow_case_studies"
    physical_csv_path: str | None = None
    primary_config_id: str = "orbital_pca2_cubes10_overlap0p35"
    requested_node_ids: list[str] = field(default_factory=lambda: list(DEFAULT_REQUESTED_NODES))
    top_method_required: str = "Radial Velocity"
    n_cases: int = 3
    min_case_members: int = 5
    max_mean_imputation_fraction: float = 0.25
    peripheral_degree_threshold: int = 1
    peripheral_component_max_nodes: int = 3
    epsilon: float = 1e-9
    analog_shadow_quantile: float = 0.5
    analog_centroid_distance_tau: float = 1.5
    analog_min_nodes: int = 2
    analog_min_members: int = 5
    analog_min_r3_valid_fraction: float = 0.6
    neighbor_reference_min_size: int = 3
    analog_count_cap: int = 8
    seed: int = 42
    r3_variables: list[str] = field(default_factory=lambda: list(R3_VARIABLES))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config_path() -> Path:
    return CONFIGS_DIR / "local_shadow_case_studies.yaml"


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
            raise RuntimeError(f"No se pudo parsear {path}. Usa JSON o instala PyYAML.") from exc
        payload = yaml.safe_load(stripped)
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"El archivo de configuracion {path} debe contener un diccionario.")
        return payload


def load_local_case_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> LocalShadowCaseConfig:
    config_path = resolve_repo_path(path, default_config_path())
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})
    return LocalShadowCaseConfig(**payload)


def resolved_mapper_outputs_dir(config: LocalShadowCaseConfig) -> Path:
    return resolve_repo_path(config.mapper_outputs_dir, OUTPUTS_DIR / "mapper")


def resolved_audit_outputs_dir(config: LocalShadowCaseConfig) -> Path:
    return resolve_repo_path(config.audit_outputs_dir, OUTPUTS_DIR / "observational_bias_audit")


def resolved_shadow_outputs_dir(config: LocalShadowCaseConfig) -> Path:
    return resolve_repo_path(config.shadow_outputs_dir, OUTPUTS_DIR / "observational_shadow")


def resolved_local_outputs_dir(config: LocalShadowCaseConfig) -> Path:
    return resolve_repo_path(config.local_outputs_dir, OUTPUTS_DIR / "local_shadow_case_studies")


def resolved_latex_dir(config: LocalShadowCaseConfig) -> Path:
    return resolve_repo_path(config.latex_dir, LATEX_DIR / "local_shadow_case_studies")

