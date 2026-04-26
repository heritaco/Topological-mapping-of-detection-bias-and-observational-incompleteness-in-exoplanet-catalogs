from __future__ import annotations

import numpy as np
import pandas as pd

from .physical_derivation import DensityDerivationAudit


def build_missingness_audit(
    numeric_before: pd.DataFrame,
    transformed_before: pd.DataFrame,
    imputed_by_method: dict[str, pd.DataFrame],
    density_audit: DensityDerivationAudit,
    log_audit: pd.DataFrame,
) -> pd.DataFrame:
    log_by_feature = log_audit.set_index("feature") if not log_audit.empty else pd.DataFrame()
    rows: list[dict[str, object]] = []
    total = len(numeric_before)

    for feature in numeric_before.columns:
        row = {
            "feature": feature,
            "rows": total,
            "observed_after_physical_derivation": int(numeric_before[feature].notna().sum()),
            "missing_after_physical_derivation": int(numeric_before[feature].isna().sum()),
            "missing_for_imputation": int(transformed_before[feature].isna().sum()),
            "missing_for_imputation_pct": round(float(transformed_before[feature].isna().mean() * 100), 4),
            "log10_applied": bool(log_by_feature.loc[feature, "log10_applied"]) if feature in log_by_feature.index else False,
            "nonpositive_set_missing": int(log_by_feature.loc[feature, "nonpositive_set_missing"])
            if feature in log_by_feature.index
            else 0,
            "derived_density_count": density_audit.derived_count if feature == "pl_dens" else 0,
        }
        for method, imputed in imputed_by_method.items():
            row[f"missing_after_{method}"] = int(imputed[feature].isna().sum())
        rows.append(row)

    return pd.DataFrame(rows)


def compare_to_complete_cases(
    numeric_before: pd.DataFrame,
    transformed_before: pd.DataFrame,
    imputed_by_method: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    complete_mask = transformed_before.notna().all(axis=1)
    rows: list[dict[str, object]] = []

    for method, imputed in imputed_by_method.items():
        for feature in numeric_before.columns:
            missing_feature = transformed_before[feature].isna()
            complete_values = numeric_before.loc[complete_mask, feature]
            imputed_all = imputed[feature]
            imputed_missing = imputed.loc[missing_feature, feature]

            complete_median = complete_values.median()
            imputed_median = imputed_all.median()
            denominator = abs(complete_median) if pd.notna(complete_median) and complete_median != 0 else np.nan
            rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "complete_case_rows": int(complete_mask.sum()),
                    "missing_cells_imputed": int(missing_feature.sum()),
                    "complete_case_median": complete_median,
                    "imputed_all_rows_median": imputed_median,
                    "imputed_missing_cells_median": imputed_missing.median() if not imputed_missing.empty else np.nan,
                    "median_delta_vs_complete_cases": imputed_median - complete_median
                    if pd.notna(imputed_median) and pd.notna(complete_median)
                    else np.nan,
                    "relative_median_delta_vs_complete_cases": (imputed_median - complete_median) / denominator
                    if pd.notna(denominator) and pd.notna(imputed_median)
                    else np.nan,
                    "complete_case_p05": complete_values.quantile(0.05),
                    "complete_case_p95": complete_values.quantile(0.95),
                    "imputed_all_rows_p05": imputed_all.quantile(0.05),
                    "imputed_all_rows_p95": imputed_all.quantile(0.95),
                }
            )

    return pd.DataFrame(rows)


def validation_metrics_by_feature(
    method: str,
    feature: str,
    truth: pd.Series,
    prediction: pd.Series,
) -> dict[str, object]:
    errors = prediction - truth
    abs_errors = errors.abs()
    truth_abs = truth.abs().replace(0, np.nan)
    rel_abs_errors = abs_errors / truth_abs
    return {
        "method": method,
        "feature": feature,
        "masked_cells": int(truth.notna().sum()),
        "mae": abs_errors.mean(),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else np.nan,
        "median_abs_error": abs_errors.median(),
        "median_relative_abs_error": rel_abs_errors.median(),
    }


def summarize_validation(validation_by_feature: pd.DataFrame) -> pd.DataFrame:
    if validation_by_feature.empty:
        return validation_by_feature
    summary = (
        validation_by_feature.groupby("method", as_index=False)
        .agg(
            masked_cells=("masked_cells", "sum"),
            mean_feature_mae=("mae", "mean"),
            mean_feature_rmse=("rmse", "mean"),
            median_feature_relative_abs_error=("median_relative_abs_error", "median"),
        )
        .sort_values("method")
    )
    return summary

