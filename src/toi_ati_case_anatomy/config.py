from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG: Dict[str, Any] = {
    "project": {"name": "toi_ati_case_anatomy", "random_state": 42},
    "inputs": {},
    "outputs": {
        "base_dir": "outputs/toi_ati_case_anatomy",
        "tables_dir": "outputs/toi_ati_case_anatomy/tables",
        "figures_dir": "outputs/toi_ati_case_anatomy/figures_pdf",
        "metadata_dir": "outputs/toi_ati_case_anatomy/metadata",
        "logs_dir": "outputs/toi_ati_case_anatomy/logs",
        "latex_dir": "latex/08_toi_ati_case_anatomy",
    },
    "analysis": {
        "config_id": "orbital_pca2_cubes10_overlap0p35",
        "top_n_regions": 10,
        "top_n_anchors": 10,
        "top_n_anchors_for_tables": 5,
        "detailed_case_count": 5,
        "required_columns_policy": "warn",
        "default_case_nodes": [],
        "default_anchor_names": [],
        "epsilon": 1e-9,
    },
    "figures": {"make_figures": True, "top_n_bars": 20, "save_format": "pdf"},
    "report": {"make_latex": True, "make_markdown_summary": True},
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
    def analysis(self) -> Dict[str, Any]:
        return self.section("analysis")

    @property
    def inputs(self) -> Dict[str, Any]:
        return self.section("inputs")

    @property
    def outputs(self) -> Dict[str, Any]:
        return self.section("outputs")


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> ProjectConfig:
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
    return ProjectConfig(raw=cfg, path=cfg_path)
