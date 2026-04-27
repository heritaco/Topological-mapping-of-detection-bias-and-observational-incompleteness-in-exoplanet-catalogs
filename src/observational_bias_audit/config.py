from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .paths import CONFIGS_DIR, LATEX_DIR, OUTPUTS_DIR, resolve_repo_path


DEFAULT_COMPARISON_CONFIGS = [
    "phys_min_pca2_cubes10_overlap0p35",
    "joint_no_density_pca2_cubes10_overlap0p35",
    "joint_pca2_cubes10_overlap0p35",
    "thermal_pca2_cubes10_overlap0p35",
]


@dataclass
class AuditConfig:
    mapper_outputs_dir: str = "outputs/mapper"
    audit_outputs_dir: str = "outputs/observational_bias_audit"
    latex_dir: str = "latex/observational_bias_audit"
    physical_csv_path: str | None = None
    primary_config_id: str = "orbital_pca2_cubes10_overlap0p35"
    comparison_config_ids: list[str] = field(default_factory=lambda: list(DEFAULT_COMPARISON_CONFIGS))
    n_permutations: int = 1000
    seed: int = 42
    heatmap_top_n_nodes: int = 40
    enrichment_top_n_nodes: int = 25
    enrichment_top_n_methods: int = 8
    peripheral_degree_threshold: int = 1
    peripheral_component_max_nodes: int = 3

    def all_config_ids(self) -> list[str]:
        values = [self.primary_config_id, *self.comparison_config_ids]
        ordered: list[str] = []
        seen: set[str] = set()
        for value in values:
            if value and value not in seen:
                ordered.append(value)
                seen.add(value)
        return ordered

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config_path() -> Path:
    return CONFIGS_DIR / "observational_bias_audit.yaml"


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
            raise RuntimeError(
                f"No se pudo parsear {path}. Usa YAML compatible con JSON o instala PyYAML."
            ) from exc
        payload = yaml.safe_load(stripped)
        if payload is None:
            return {}
        if not isinstance(payload, dict):
            raise ValueError(f"El archivo de configuracion {path} debe contener un objeto/diccionario.")
        return payload


def load_audit_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> AuditConfig:
    config_path = resolve_repo_path(path, default_config_path())
    payload: dict[str, Any] = {}
    if config_path.exists():
        payload = _parse_config_text(config_path.read_text(encoding="utf-8"), config_path)
    if overrides:
        payload.update({key: value for key, value in overrides.items() if value is not None})
    return AuditConfig(**payload)


def resolved_mapper_outputs_dir(config: AuditConfig) -> Path:
    return resolve_repo_path(config.mapper_outputs_dir, OUTPUTS_DIR / "mapper")


def resolved_audit_outputs_dir(config: AuditConfig) -> Path:
    return resolve_repo_path(config.audit_outputs_dir, OUTPUTS_DIR / "observational_bias_audit")


def resolved_latex_dir(config: AuditConfig) -> Path:
    return resolve_repo_path(config.latex_dir, LATEX_DIR / "observational_bias_audit")
