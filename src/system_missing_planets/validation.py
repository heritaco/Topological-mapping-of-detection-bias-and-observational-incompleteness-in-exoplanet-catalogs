from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .gap_model import GapStatistics, expand_gap_candidates, expected_missing_counts


def run_leave_one_out_validation(
    catalog: pd.DataFrame,
    system_metadata: pd.DataFrame,
    gap_stats: GapStatistics,
    *,
    min_gap_ratio: float,
    max_candidates_per_gap: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    metadata = system_metadata.set_index("hostname")
    valid = catalog[pd.to_numeric(catalog["pl_orbper"], errors="coerce") > 0].copy()
    rows: list[dict[str, object]] = []
    for hostname, group in valid.groupby("hostname", dropna=True):
        if hostname not in metadata.index:
            continue
        ordered = group.sort_values("pl_orbper").reset_index(drop=True)
        if len(ordered) < 3:
            continue
        for idx in range(1, len(ordered) - 1):
            hidden = ordered.iloc[idx]
            inner = ordered.iloc[idx - 1]
            outer = ordered.iloc[idx + 1]
            p_in = float(inner["pl_orbper"])
            p_out = float(outer["pl_orbper"])
            if not np.isfinite(p_in) or not np.isfinite(p_out) or p_in <= 0 or p_out <= 0:
                continue
            gap_ratio = p_out / p_in
            gap_log_width = math.log10(p_out) - math.log10(p_in)
            expected = expected_missing_counts(
                gap_ratio=gap_ratio,
                gap_log_width=gap_log_width,
                stats=gap_stats,
                min_gap_ratio=min_gap_ratio,
                max_candidates_per_gap=max_candidates_per_gap,
                method_group=str(metadata.loc[hostname, "dominant_method_group"]),
                mass_bin=str(metadata.loc[hostname, "stellar_mass_bin"]),
            )
            gap_frame = pd.DataFrame(
                [
                    {
                        "hostname": hostname,
                        "inner_planet": inner["pl_name"],
                        "outer_planet": outer["pl_name"],
                        "P_in": p_in,
                        "P_out": p_out,
                        "a_in": pd.to_numeric(pd.Series([inner.get("pl_orbsmax")]), errors="coerce").iloc[0],
                        "a_out": pd.to_numeric(pd.Series([outer.get("pl_orbsmax")]), errors="coerce").iloc[0],
                        "gap_ratio": gap_ratio,
                        "gap_log_width": gap_log_width,
                        "expected_missing_count": expected["expected_missing_count"],
                        "expected_missing_count_heuristic": expected["expected_missing_count_heuristic"],
                        "expected_missing_count_empirical": expected["expected_missing_count_empirical"],
                        "typical_gap_log_width": expected["typical_gap_log_width"],
                        "typical_gap_source": expected["typical_gap_source"],
                        "dominant_discoverymethod_system": metadata.loc[hostname, "dominant_discoverymethod_system"],
                        "dominant_method_group": metadata.loc[hostname, "dominant_method_group"],
                        "inner_discoverymethod": inner.get("discoverymethod", "Unknown"),
                        "outer_discoverymethod": outer.get("discoverymethod", "Unknown"),
                        "candidate_stellar_mass": metadata.loc[hostname, "system_st_mass_median"],
                        "candidate_stellar_radius": metadata.loc[hostname, "system_st_rad_median"],
                        "gap_score": gap_stats.gap_percentile(gap_log_width),
                    }
                ]
            )
            predictions = expand_gap_candidates(gap_frame)
            true_period = float(hidden["pl_orbper"])
            if predictions.empty:
                rows.append(
                    {
                        "hostname": hostname,
                        "removed_planet": hidden["pl_name"],
                        "true_period_days": true_period,
                        "predicted_period_best": np.nan,
                        "abs_logP_error": np.nan,
                        "recovered": False,
                        "within_0p1_dex": False,
                        "within_0p2_dex": False,
                        "within_factor_1p5": False,
                        "within_factor_2": False,
                    }
                )
                continue
            predictions = predictions.copy()
            predictions["abs_logP_error"] = np.abs(np.log10(pd.to_numeric(predictions["candidate_period_days"], errors="coerce")) - math.log10(true_period))
            best = predictions.sort_values("abs_logP_error").iloc[0]
            error = float(best["abs_logP_error"])
            predicted_period = float(best["candidate_period_days"])
            factor = max(predicted_period / true_period, true_period / predicted_period) if predicted_period > 0 else np.inf
            rows.append(
                {
                    "hostname": hostname,
                    "removed_planet": hidden["pl_name"],
                    "true_period_days": true_period,
                    "predicted_period_best": predicted_period,
                    "abs_logP_error": error,
                    "recovered": True,
                    "within_0p1_dex": bool(error <= 0.1),
                    "within_0p2_dex": bool(error <= 0.2),
                    "within_factor_1p5": bool(factor <= 1.5),
                    "within_factor_2": bool(factor <= 2.0),
                }
            )
    validation = pd.DataFrame(rows)
    summary: dict[str, Any] = {
        "n_holdout_tests": int(len(validation)),
        "median_abs_logP_error": float(pd.to_numeric(validation.get("abs_logP_error"), errors="coerce").median()) if not validation.empty else np.nan,
        "recall_within_0p1_dex": float(validation["within_0p1_dex"].mean()) if not validation.empty else np.nan,
        "recall_within_0p2_dex": float(validation["within_0p2_dex"].mean()) if not validation.empty else np.nan,
        "recall_within_factor_1p5": float(validation["within_factor_1p5"].mean()) if not validation.empty else np.nan,
        "recall_within_factor_2": float(validation["within_factor_2"].mean()) if not validation.empty else np.nan,
    }
    return validation, summary


def validation_error_by_system(validation: pd.DataFrame) -> pd.DataFrame:
    if validation.empty:
        return pd.DataFrame(columns=["hostname", "validation_error_if_available"])
    grouped = validation.groupby("hostname", dropna=True)["abs_logP_error"].median().reset_index()
    return grouped.rename(columns={"abs_logP_error": "validation_error_if_available"})
