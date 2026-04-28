from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd

from .config import ProjectConfig
from .paths import find_repo_root, resolve_path


OPTIONAL_TABLES = {
    "r3_node_geometry",
    "node_shadow_metrics",
    "case_node_summary",
    "case_anchor_planets",
    "membership",
    "edges",
    "imputed_dataset",
}


def load_csv_if_exists(path: Path | None, *, required: bool = False) -> pd.DataFrame:
    if path is None or not path.exists():
        if required:
            raise FileNotFoundError(f"Required CSV not found: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def load_input_tables(cfg: ProjectConfig, root: Path | None = None) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    repo_root = root or find_repo_root()
    warnings: Dict[str, str] = {}
    tables: Dict[str, pd.DataFrame] = {}
    for name, value in cfg.inputs.items():
        path = resolve_path(repo_root, value)
        required = name not in OPTIONAL_TABLES
        try:
            tables[name] = load_csv_if_exists(path, required=required)
            if tables[name].empty and not required:
                warnings[name] = f"Optional input missing or empty: {path}"
        except FileNotFoundError as exc:
            if required:
                raise
            warnings[name] = str(exc)
            tables[name] = pd.DataFrame()
    return tables, warnings


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def require_columns(df: pd.DataFrame, columns: Iterable[str], table_name: str) -> list[str]:
    return [c for c in columns if c not in df.columns]
