"""General utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence
import json
import re
import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_column_name(name: str) -> str:
    name = str(name).strip()
    name = re.sub(r"[^0-9a-zA-Z_]+", "_", name)
    name = re.sub(r"_+", "_", name)
    return name.strip("_").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [clean_column_name(c) for c in out.columns]
    return out


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        key = clean_column_name(cand)
        if key in lower_map:
            return lower_map[key]
    return None


def coalesce_columns(df: pd.DataFrame, candidates: Sequence[str], default=np.nan) -> pd.Series:
    cols = [c for c in [first_existing_column(df, [x]) for x in candidates] if c is not None]
    if not cols:
        return pd.Series(default, index=df.index)
    result = df[cols[0]].copy()
    for c in cols[1:]:
        result = result.where(result.notna(), df[c])
    return result


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_log10(values, min_positive: float = 1.0e-12):
    arr = pd.to_numeric(values, errors="coerce")
    arr = np.asarray(arr, dtype=float)
    arr = np.where(np.isfinite(arr) & (arr > min_positive), arr, np.nan)
    return np.log10(arr)


def weighted_quantile(values, quantiles, sample_weight=None):
    values = np.asarray(values, dtype=float)
    quantiles = np.asarray(quantiles, dtype=float)
    mask = np.isfinite(values)
    values = values[mask]
    if sample_weight is None:
        sample_weight = np.ones_like(values)
    else:
        sample_weight = np.asarray(sample_weight, dtype=float)[mask]
    if values.size == 0:
        return np.full_like(quantiles, np.nan, dtype=float)
    sorter = np.argsort(values)
    values = values[sorter]
    weights = sample_weight[sorter]
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if weights.sum() <= 0:
        weights = np.ones_like(values)
    cumulative = np.cumsum(weights) - 0.5 * weights
    cumulative /= np.sum(weights)
    return np.interp(quantiles, cumulative, values)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str), encoding="utf-8")


def read_csv_if_exists(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None or not path.exists():
        return None
    return pd.read_csv(path)


def finite_median(series: pd.Series, default: float = np.nan) -> float:
    arr = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(np.nanmedian(arr))


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
