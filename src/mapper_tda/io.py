from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_MAPPER_OUTPUTS_DIR = OUTPUTS_DIR / "mapper"
DEFAULT_IMPUTATION_OUTPUTS_DIR = OUTPUTS_DIR / "imputation"

MAPPER_GEOMETRY_CANDIDATES: dict[str, list[str]] = {
    "iterative": [
        "outputs/imputation/data/mapper_features_imputed_iterative.csv",
        "outputs/imputation/mapper_features_imputed_iterative.csv",
        "reports/imputation/mapper_features_imputed_iterative.csv",
        "outputs/imputation/data/mapper_features_imputed_knn.csv",
        "reports/imputation/mapper_features_imputed_knn.csv",
        "reports/imputation/mapper_features_complete_case.csv",
    ],
    "knn": [
        "outputs/imputation/data/mapper_features_imputed_knn.csv",
        "outputs/imputation/mapper_features_imputed_knn.csv",
        "reports/imputation/mapper_features_imputed_knn.csv",
        "reports/imputation/mapper_features_complete_case.csv",
    ],
    "median": [
        "outputs/imputation/data/mapper_features_imputed_median.csv",
        "outputs/imputation/mapper_features_imputed_median.csv",
        "reports/imputation/mapper_features_imputed_median.csv",
        "reports/imputation/mapper_features_complete_case.csv",
    ],
    "complete_case": [
        "outputs/imputation/data/mapper_features_complete_case.csv",
        "outputs/imputation/mapper_features_complete_case.csv",
        "reports/imputation/mapper_features_complete_case.csv",
    ],
    "raw": [],
}

PHYSICAL_CANDIDATES: dict[str, list[str]] = {
    "iterative": [
        "outputs/imputation/data/PSCompPars_imputed_iterative.csv",
        "outputs/imputation/PSCompPars_imputed_iterative.csv",
        "reports/imputation/PSCompPars_imputed_iterative.csv",
        "outputs/imputation/data/PSCompPars_imputed_knn.csv",
        "reports/imputation/PSCompPars_imputed_knn.csv",
    ],
    "knn": [
        "outputs/imputation/data/PSCompPars_imputed_knn.csv",
        "outputs/imputation/PSCompPars_imputed_knn.csv",
        "reports/imputation/PSCompPars_imputed_knn.csv",
    ],
    "median": [
        "outputs/imputation/data/PSCompPars_imputed_median.csv",
        "outputs/imputation/PSCompPars_imputed_median.csv",
        "reports/imputation/PSCompPars_imputed_median.csv",
    ],
    "complete_case": [
        "outputs/imputation/data/PSCompPars_imputed_iterative.csv",
        "outputs/imputation/PSCompPars_imputed_iterative.csv",
        "reports/imputation/PSCompPars_imputed_iterative.csv",
    ],
    "raw": [],
}


def _resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", low_memory=False)


def resolve_output_dir(value: str | None, default_path: Path) -> Path:
    path = _resolve_path(value or default_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_outputs_dir(value: str | None = None) -> Path:
    return resolve_output_dir(value, DEFAULT_MAPPER_OUTPUTS_DIR)


def resolve_imputation_outputs_dir(value: str | None = None) -> Path:
    return resolve_output_dir(value, DEFAULT_IMPUTATION_OUTPUTS_DIR)


def _first_existing_path(candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = _resolve_path(candidate)
        if path.exists():
            return path
    return None


def _latest_raw_pscomppars() -> Path:
    data_candidates = sorted(DATA_DIR.glob("PSCompPars_*.csv"))
    if not data_candidates:
        raise FileNotFoundError("No se encontro ningun data/PSCompPars_*.csv como fallback raw.")
    return data_candidates[-1]


def resolve_mapper_features_path(
    explicit_path: str | None = None,
    input_method: str = "iterative",
) -> Path:
    if explicit_path:
        path = _resolve_path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe mapper-features CSV: {path}")
        return path

    candidate = _first_existing_path(MAPPER_GEOMETRY_CANDIDATES.get(input_method, []))
    if candidate is not None:
        return candidate
    return _latest_raw_pscomppars()


def resolve_physical_csv_path(
    explicit_path: str | None = None,
    input_method: str = "iterative",
) -> Path:
    if explicit_path:
        path = _resolve_path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe physical CSV: {path}")
        return path

    candidate = _first_existing_path(PHYSICAL_CANDIDATES.get(input_method, []))
    if candidate is not None:
        return candidate
    return _latest_raw_pscomppars()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        return [_json_ready(item) for item in value.tolist()]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_mapper_output_tree(outputs_dir: Path) -> dict[str, Path]:
    tree = {
        "root": outputs_dir,
        "data": outputs_dir / "data",
        "graphs": outputs_dir / "graphs",
        "nodes": outputs_dir / "nodes",
        "edges": outputs_dir / "edges",
        "metrics": outputs_dir / "metrics",
        "distances": outputs_dir / "distances",
        "tables": outputs_dir / "tables",
        "figures_pdf": outputs_dir / "figures_pdf",
        "figures_png": outputs_dir / "figures_png",
        "figures_interpretation_pdf": outputs_dir / "figures_pdf" / "interpretation",
        "figures_interpretation_png": outputs_dir / "figures_png" / "interpretation",
        "latex_assets": outputs_dir / "latex_assets",
        "config": outputs_dir / "config",
        "bootstrap": outputs_dir / "bootstrap",
        "null_models": outputs_dir / "null_models",
        "logs": PROJECT_ROOT / "outputs" / "logs",
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    (tree["figures_pdf"] / "presentation").mkdir(parents=True, exist_ok=True)
    return tree


def alignment_keys_for_frames(mapper_df: pd.DataFrame, physical_df: pd.DataFrame) -> tuple[str | None, list[str]]:
    if "rowid" in mapper_df.columns and "rowid" in physical_df.columns:
        return "rowid", ["rowid"]
    if {"pl_name", "hostname"}.issubset(mapper_df.columns) and {"pl_name", "hostname"}.issubset(physical_df.columns):
        return "pl_name_hostname", ["pl_name", "hostname"]
    return None, []


def align_mapper_and_physical_inputs(
    mapper_df: pd.DataFrame,
    physical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    warnings: list[str] = []
    key_name, key_columns = alignment_keys_for_frames(mapper_df, physical_df)

    if key_columns:
        mapper_df = mapper_df.copy()
        physical_df = physical_df.copy()
        mapper_key = mapper_df.loc[:, key_columns].copy()
        physical_key = physical_df.loc[:, key_columns].copy()
        if mapper_key.duplicated().any():
            raise ValueError(f"Hay claves duplicadas en mapper_features para {key_columns}.")
        if physical_key.duplicated().any():
            raise ValueError(f"Hay claves duplicadas en physical_csv para {key_columns}.")

        merged = mapper_df.reset_index(drop=False).merge(
            physical_df.reset_index(drop=False),
            on=key_columns,
            how="outer",
            indicator=True,
            suffixes=("_mapper", "_physical"),
        )
        matched = merged[merged["_merge"] == "both"].copy()
        unmatched = merged[merged["_merge"] != "both"].copy()
        if matched.empty:
            raise ValueError("No fue posible alinear mapper_features con physical_csv.")
        if not unmatched.empty:
            warnings.append(f"Se detectaron {len(unmatched)} filas no emparejadas.")

        mapper_index_col = "index_mapper"
        physical_index_col = "index_physical"
        mapper_aligned = mapper_df.iloc[matched[mapper_index_col].astype(int).tolist()].reset_index(drop=True)
        physical_aligned = physical_df.iloc[matched[physical_index_col].astype(int).tolist()].reset_index(drop=True)
    else:
        if len(mapper_df) != len(physical_df):
            raise ValueError(
                "No hay claves de alineacion y las longitudes no coinciden entre mapper_features y physical_csv."
            )
        key_name = "preserved_index"
        mapper_aligned = mapper_df.reset_index(drop=True)
        physical_aligned = physical_df.reset_index(drop=True)

    if {"pl_name", "hostname"}.issubset(mapper_aligned.columns) and {"pl_name", "hostname"}.issubset(physical_aligned.columns):
        same_names = (
            mapper_aligned["pl_name"].astype("string").fillna("")
            == physical_aligned["pl_name"].astype("string").fillna("")
        )
        same_hosts = (
            mapper_aligned["hostname"].astype("string").fillna("")
            == physical_aligned["hostname"].astype("string").fillna("")
        )
        if not bool((same_names & same_hosts).all()):
            warnings.append("pl_name/hostname no coinciden completamente tras la alineacion.")

    summary = {
        "n_rows_mapper_features": int(len(mapper_df)),
        "n_rows_physical": int(len(physical_df)),
        "alignment_key_used": key_name,
        "n_matched_rows": int(len(mapper_aligned)),
        "n_unmatched_rows": int(max(len(mapper_df), len(physical_df)) - len(mapper_aligned)),
        "warnings": " | ".join(warnings),
    }
    return mapper_aligned, physical_aligned, summary
