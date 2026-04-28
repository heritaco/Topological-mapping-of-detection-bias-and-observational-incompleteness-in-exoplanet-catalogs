from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from .config import ProjectConfig
from .paths import find_repo_root, resolve_path

OPTIONAL_INPUTS = {
    "top_anchor_radius_deficit_tables",
    "top_anchor_radius_deficit_summary",
    "final_presentation_cases",
    "top_regions_case_anatomy",
    "top_anchors_case_anatomy",
    "r3_node_geometry",
    "node_shadow_metrics",
    "node_method_metrics",
    "dataset",
    "membership",
    "edges",
}


def load_csv_if_exists(path: Path | None, *, required: bool = False) -> pd.DataFrame:
    if path is None or not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_input_tables(cfg: ProjectConfig, root: Path | None = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], Dict[str, str]]:
    repo_root = root or find_repo_root()
    warnings: Dict[str, str] = {}
    inputs_used: Dict[str, str] = {}
    tables: Dict[str, pd.DataFrame] = {}
    for name, value in cfg.inputs.items():
        path = resolve_path(repo_root, value)
        required = name not in OPTIONAL_INPUTS
        try:
            frame = load_csv_if_exists(path, required=required)
            tables[name] = frame
            inputs_used[name] = str(path) if path is not None else "null"
            if frame.empty and not required:
                warnings[name] = f"Optional input missing or empty: {path}"
        except FileNotFoundError as exc:
            if required:
                missing_hint = _missing_input_hint(name)
                raise FileNotFoundError(f"{exc}. {missing_hint}") from exc
            warnings[name] = str(exc)
            tables[name] = pd.DataFrame()
            inputs_used[name] = str(path) if path is not None else "null"
    return tables, warnings, inputs_used


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_log(warnings: Dict[str, str] | Iterable[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(warnings, dict):
        lines = [f"{key}: {value}" for key, value in warnings.items()]
    else:
        lines = [str(item) for item in warnings]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def git_commit_hash(root: Path | None = None) -> str | None:
    repo_root = root or find_repo_root()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {table_name}: {', '.join(missing)}")


def _missing_input_hint(name: str) -> str:
    hints = {
        "regional_toi_scores": "Run python -m src.topological_incompleteness_index.run_topological_incompleteness --config configs/topological_incompleteness_index.yaml first.",
        "anchor_ati_scores": "Run python -m src.topological_incompleteness_index.run_topological_incompleteness --config configs/topological_incompleteness_index.yaml first.",
        "anchor_neighbor_deficits": "Run python -m src.topological_incompleteness_index.run_topological_incompleteness --config configs/topological_incompleteness_index.yaml first.",
    }
    return hints.get(name, "Run the upstream pipeline that generates this table before future validation.")
