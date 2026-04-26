from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
PROCESSED_DIR = DATA_DIR / "processed"


def find_csv(explicit_path: str | None = None) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV: {path}")
        return path

    candidates = sorted(PROJECT_ROOT.glob("PSCompPars_*.csv"))
    candidates += sorted(DATA_DIR.glob("PSCompPars_*.csv"))
    candidates += sorted((DATA_DIR / "raw").glob("PSCompPars_*.csv"))
    if not candidates:
        raise FileNotFoundError("No encontre PSCompPars_*.csv en la raiz, data/ ni data/raw.")
    return candidates[-1]


def load_pscomppars(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, comment="#", low_memory=False)


def resolve_output_dir(base: str | None, default_root: Path, csv_path: Path) -> Path:
    path = Path(base) if base else default_root / csv_path.stem / "imputation"
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

