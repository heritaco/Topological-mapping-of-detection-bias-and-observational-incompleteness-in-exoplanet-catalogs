"""Lightweight data quality checks."""

from __future__ import annotations

import pandas as pd


def row_column_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {"rows": int(len(frame)), "columns": int(len(frame.columns))}
