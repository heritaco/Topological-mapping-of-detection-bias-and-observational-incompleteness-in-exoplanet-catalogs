"""Unified config loading and run-directory preparation."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .io import deep_merge, ensure_dir, read_yaml, write_yaml
from .paths import find_repo_root, resolve_config_paths, resolve_repo_path

REQUIRED_SECTIONS = ("project", "paths", "run", "inputs", "stages")


def generate_run_id(prefix: str = "run") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def parse_override(value: str) -> tuple[list[str], Any]:
    key, raw = value.split("=", 1)
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        parsed: Any = lowered == "true"
    elif lowered in {"none", "null"}:
        parsed = None
    else:
        try:
            parsed = int(raw)
        except ValueError:
            try:
                parsed = float(raw)
            except ValueError:
                parsed = raw
    return key.split("."), parsed


def apply_overrides(config: dict[str, Any], overrides: dict[str, Any] | list[str] | None) -> dict[str, Any]:
    out = deepcopy(config)
    if not overrides:
        return out
    if isinstance(overrides, dict):
        return deep_merge(out, overrides)
    for override in overrides:
        keys, value = parse_override(override)
        cursor = out
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})
        cursor[keys[-1]] = value
    return out


def validate_config(config: dict[str, Any]) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ValueError(f"Missing required config sections: {', '.join(missing)}")


def load_pipeline_config(
    config_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base_path = Path(config_path or "configs/pipeline/default.yaml")
    root = Path(repo_root).resolve() if repo_root else find_repo_root(base_path.parent)
    base_path = resolve_repo_path(root, base_path) or base_path
    original = read_yaml(base_path)
    effective = deepcopy(original)

    if experiment_path:
        exp_path = resolve_repo_path(root, experiment_path)
        if exp_path is None:
            raise FileNotFoundError(experiment_path)
        experiment = read_yaml(exp_path)
        original = {"base_config": original, "experiment_config": experiment}
        effective = deep_merge(effective, experiment)

    effective = apply_overrides(effective, overrides)
    effective.setdefault("paths", {})["repo_root"] = str(root)
    effective.setdefault("run", {})
    if not effective["run"].get("run_id"):
        effective["run"]["run_id"] = generate_run_id()
    validate_config(effective)
    effective = resolve_config_paths(effective, root)
    return original, effective


def get_run_dir(config: dict[str, Any]) -> Path:
    repo_root = Path(config["paths"]["repo_root"]).resolve()
    run_id = str(config["run"]["run_id"])
    template = config.get("paths", {}).get("run_dir_template", "outputs/runs/{run_id}")
    return resolve_repo_path(repo_root, template.format(run_id=run_id)) or (repo_root / "outputs" / "runs" / run_id)


def prepare_run_config(
    config_path: str | Path | None = None,
    experiment_path: str | Path | None = None,
    overrides: dict[str, Any] | list[str] | None = None,
    repo_root: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    original, effective = load_pipeline_config(config_path, experiment_path, overrides, repo_root)
    run_dir = get_run_dir(effective)
    overwrite = bool(effective.get("run", {}).get("overwrite", False))
    if run_dir.exists() and not overwrite:
        # Reusing a run id is allowed for status/indexing/dry development; manifests append deterministically.
        pass
    ensure_dir(run_dir / "config")
    write_yaml(run_dir / "config" / "config_original.yaml", original)
    write_yaml(run_dir / "config" / "config_effective.yaml", effective)
    return original, effective, run_dir
