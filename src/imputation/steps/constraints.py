from __future__ import annotations

from typing import Mapping

import pandas as pd


Bounds = Mapping[str, tuple[float | None, float | None]]


def apply_feature_bounds(
    matrix: pd.DataFrame,
    bounds: Bounds,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clip physically bounded variables and return a clipping audit."""

    clipped = matrix.copy()
    rows: list[dict[str, object]] = []
    for column, (lower, upper) in bounds.items():
        if column not in clipped.columns:
            continue
        before = pd.to_numeric(clipped[column], errors="coerce")
        after = before.clip(lower=lower, upper=upper)
        rows.append(
            {
                "feature": column,
                "lower_bound": lower,
                "upper_bound": upper,
                "clipped_low": int(((before < lower) & before.notna()).sum()) if lower is not None else 0,
                "clipped_high": int(((before > upper) & before.notna()).sum()) if upper is not None else 0,
            }
        )
        clipped[column] = after
    return clipped, pd.DataFrame(rows)

