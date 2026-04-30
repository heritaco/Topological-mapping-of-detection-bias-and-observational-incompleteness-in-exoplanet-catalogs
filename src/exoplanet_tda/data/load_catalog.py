"""Catalog loading helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_catalog(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)
