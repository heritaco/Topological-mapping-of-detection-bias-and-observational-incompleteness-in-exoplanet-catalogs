"""Configuration objects and default path conventions."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence
import json

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


CATALOG_CANDIDATE_PATHS = [
    "reports/imputation/PSCompPars_imputed_iterative.csv",
    "reports/imputation/outputs/PSCompPars_imputed_iterative.csv",
    "outputs/imputation/PSCompPars_imputed_iterative.csv",
    "data/PSCompPars_imputed_iterative.csv",
    "data/PSCompPars_2026.04.25_17.36.36.csv",
]

MISSING_PLANET_CANDIDATE_PATHS = [
    "outputs/system_missing_planets/candidate_missing_planets_by_system.csv",
    "outputs/system_missing_planets/high_priority_candidates.csv",
    "outputs/system_missing_planets/final_three_case_studies.csv",
    "outputs/system_missing_planets/candidate_characterization_input.csv",
    "outputs/system_missing_planets/system_candidates.csv",
    "outputs/system_missing_planets/candidates.csv",
    "outputs/system_missing_planets/synthetic_candidates.csv",
    "reports/system_missing_planets/outputs/tables/system_candidates.csv",
    "reports/system_missing_planets/outputs/tables/candidates.csv",
    "reports/system_missing_planets/outputs/tables/synthetic_candidates.csv",
]

TOI_REGION_PATHS = [
    "outputs/topological_incompleteness_index/tables/regional_toi_scores.csv",
    "outputs/topological_incompleteness_index/tables/toi_regions.csv",
    "outputs/topological_incompleteness_index/toi_regions.csv",
    "reports/topological_incompleteness_index/outputs/tables/toi_regions.csv",
    "reports/toi_ati_case_anatomy/outputs/tables/top_regions_toi.csv",
]

ATI_ANCHOR_PATHS = [
    "outputs/topological_incompleteness_index/tables/anchor_ati_scores.csv",
    "outputs/topological_incompleteness_index/tables/ati_anchors.csv",
    "outputs/topological_incompleteness_index/ati_anchors.csv",
    "reports/topological_incompleteness_index/outputs/tables/ati_anchors.csv",
    "reports/toi_ati_case_anatomy/outputs/tables/top_anchors_ati.csv",
]


@dataclass
class PathConfig:
    repo_root: str = "."
    catalog_csv: Optional[str] = None
    candidates_csv: Optional[str] = None
    toi_regions_csv: Optional[str] = None
    ati_anchors_csv: Optional[str] = None
    output_dir: str = "outputs/candidate_characterization"
    report_dir: str = "reports/candidate_characterization"
    model_dir: str = "outputs/candidate_characterization/models"


@dataclass
class ModelConfig:
    prefer_gpu: bool = True
    random_state: int = 42
    test_size: float = 0.2
    quantiles: List[float] = field(default_factory=lambda: [0.05, 0.5, 0.95])
    analog_k: int = 75
    analog_temperature: float = 1.0
    min_training_rows: int = 80
    use_xgboost_if_available: bool = True
    calibration_cv: int = 3


@dataclass
class PhysicalConfig:
    default_bond_albedo: float = 0.30
    density_earth_g_cm3: float = 5.514
    r_sun_au: float = 0.00465047
    min_positive_value: float = 1.0e-12


@dataclass
class CharacterizationConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    physical: PhysicalConfig = field(default_factory=PhysicalConfig)
    feature_set: str = "candidate_characterization_minimal"
    feature_registry: str = "configs/features/feature_registry.yaml"
    feature_sets: str = "configs/features/feature_sets.yaml"
    allow_audit_features: bool = False
    allow_observed_diagnostic: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")


def _deep_update(base: Dict, update: Dict) -> Dict:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: Optional[str] = None) -> CharacterizationConfig:
    cfg = CharacterizationConfig()
    if not path:
        return cfg
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    raw_text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is not installed. Install pyyaml or pass JSON config.")
        data = yaml.safe_load(raw_text) or {}
    else:
        data = json.loads(raw_text)
    merged = _deep_update(cfg.to_dict(), data)
    return CharacterizationConfig(
        paths=PathConfig(**merged.get("paths", {})),
        model=ModelConfig(**merged.get("model", {})),
        physical=PhysicalConfig(**merged.get("physical", {})),
        feature_set=str(merged.get("feature_set", "candidate_characterization_minimal")),
        feature_registry=str(merged.get("feature_registry", "configs/features/feature_registry.yaml")),
        feature_sets=str(merged.get("feature_sets", "configs/features/feature_sets.yaml")),
        allow_audit_features=bool(merged.get("allow_audit_features", False)),
        allow_observed_diagnostic=bool(merged.get("allow_observed_diagnostic", False)),
    )


def resolve_first_existing(repo_root: Path, explicit: Optional[str], candidates: Sequence[str]) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = repo_root / p
        return p if p.exists() else p
    for rel in candidates:
        p = repo_root / rel
        if p.exists():
            return p
    return None
