from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from feature_config import (
    IDENTIFIER_COLUMNS,
    IMPUTATION_VALUE_BOUNDS,
    LOG_CANDIDATE_COLUMNS,
    MAPPER_BIAS_COLUMNS,
)
from imputation.io import write_json
from imputation.steps.audit import (
    build_missingness_audit,
    compare_to_complete_cases,
    summarize_validation,
    validation_metrics_by_feature,
)
from imputation.steps.baseline_imputers import impute_with_iterative, impute_with_median
from imputation.steps.constraints import apply_feature_bounds
from imputation.steps.knn_imputer import impute_with_knn
from imputation.steps.log_transform import apply_log10_transform, invert_log10_transform, log_feature_subset
from imputation.steps.physical_derivation import DensityDerivationAudit, derive_planet_density
from imputation.steps.scaling import invert_robust_scale, robust_scale


VALID_METHODS = ("knn", "median", "iterative")


@dataclass(frozen=True)
class ImputationConfig:
    features: tuple[str, ...]
    log_features: tuple[str, ...] = ()
    methods: tuple[str, ...] = ("knn", "median", "iterative")
    primary_method: str = "knn"
    n_neighbors: int = 7
    knn_weights: str = "distance"
    iterative_max_iter: int = 20
    validation_mask_fraction: float = 0.10
    validation_max_complete_rows: int = 2000
    random_state: int = 42
    value_bounds: Mapping[str, tuple[float | None, float | None]] = field(
        default_factory=lambda: dict(IMPUTATION_VALUE_BOUNDS)
    )


@dataclass
class ImputationResult:
    prepared_df: pd.DataFrame
    numeric_before: pd.DataFrame
    transformed_before: pd.DataFrame
    imputed_by_method: dict[str, pd.DataFrame]
    output_by_method: dict[str, pd.DataFrame]
    missingness_audit: pd.DataFrame
    complete_case_comparison: pd.DataFrame
    validation_by_feature: pd.DataFrame
    validation_summary: pd.DataFrame
    log_transform_audit: pd.DataFrame
    constraint_audit: pd.DataFrame
    density_audit: DensityDerivationAudit


def default_log_features(features: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    return tuple(log_feature_subset(features, LOG_CANDIDATE_COLUMNS))


def validate_methods(methods: tuple[str, ...], primary_method: str) -> None:
    invalid = sorted(set(methods) - set(VALID_METHODS))
    if invalid:
        raise ValueError(f"Metodos de imputacion no soportados: {invalid}")
    if primary_method not in methods:
        raise ValueError("primary_method debe estar incluido en methods.")


def coerce_numeric_features(df: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    missing = [feature for feature in features if feature not in df.columns]
    if missing:
        raise KeyError(f"Faltan features para imputacion: {missing}")
    matrix = df.loc[:, list(features)].copy()
    for column in matrix.columns:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return matrix


def _impute_scaled_matrix(method: str, scaled: pd.DataFrame, config: ImputationConfig) -> pd.DataFrame:
    if method == "knn":
        return impute_with_knn(scaled, n_neighbors=config.n_neighbors, weights=config.knn_weights)
    if method == "median":
        return impute_with_median(scaled)
    if method == "iterative":
        return impute_with_iterative(
            scaled,
            random_state=config.random_state,
            max_iter=config.iterative_max_iter,
        )
    raise ValueError(f"Metodo no soportado: {method}")


def impute_numeric_matrix(
    numeric: pd.DataFrame,
    method: str,
    config: ImputationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transformed, log_audit = apply_log10_transform(numeric, config.log_features)
    scaled, scaler = robust_scale(transformed)
    scaled_imputed = _impute_scaled_matrix(method, scaled, config)
    transformed_imputed = invert_robust_scale(scaled_imputed, scaler)
    original_units = invert_log10_transform(transformed_imputed, config.log_features)
    bounded, constraint_audit = apply_feature_bounds(original_units, config.value_bounds)
    return bounded, log_audit.to_frame(), constraint_audit


def build_output_frame(
    prepared_df: pd.DataFrame,
    imputed: pd.DataFrame,
    transformed_before: pd.DataFrame,
    method: str,
) -> pd.DataFrame:
    feature_set = list(imputed.columns)
    preferred_context = [
        *IDENTIFIER_COLUMNS,
        *MAPPER_BIAS_COLUMNS,
        "pl_dens_source",
    ]
    context_cols = []
    for column in preferred_context:
        if column in prepared_df.columns and column not in feature_set and column not in context_cols:
            context_cols.append(column)

    output = prepared_df.loc[:, context_cols].copy()
    for feature in feature_set:
        output[feature] = imputed[feature]
    for feature in feature_set:
        output[f"{feature}_was_imputed"] = transformed_before[feature].isna().to_numpy()

    output["complete_case_before_imputation"] = transformed_before.notna().all(axis=1).to_numpy()
    output["any_feature_was_imputed"] = transformed_before.isna().any(axis=1).to_numpy()
    output["imputation_method"] = method
    return output


def run_masked_validation(
    numeric_before: pd.DataFrame,
    transformed_before: pd.DataFrame,
    config: ImputationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete_mask = transformed_before.notna().all(axis=1)
    complete_numeric = numeric_before.loc[complete_mask].copy()
    if len(complete_numeric) < 2 or config.validation_mask_fraction <= 0:
        empty = pd.DataFrame(
            columns=[
                "method",
                "feature",
                "masked_cells",
                "mae",
                "rmse",
                "median_abs_error",
                "median_relative_abs_error",
            ]
        )
        return empty, empty

    if len(complete_numeric) > config.validation_max_complete_rows:
        complete_numeric = complete_numeric.sample(
            n=config.validation_max_complete_rows,
            random_state=config.random_state,
        )

    rng = np.random.default_rng(config.random_state)
    mask = rng.random(complete_numeric.shape) < config.validation_mask_fraction
    if not mask.any():
        mask[rng.integers(0, complete_numeric.shape[0]), rng.integers(0, complete_numeric.shape[1])] = True
    for column_index in range(complete_numeric.shape[1]):
        if not mask[:, column_index].any():
            mask[rng.integers(0, complete_numeric.shape[0]), column_index] = True
        if mask[:, column_index].all():
            mask[rng.integers(0, complete_numeric.shape[0]), column_index] = False

    masked_numeric = complete_numeric.mask(mask)
    rows: list[dict[str, object]] = []
    for method in config.methods:
        imputed, _, _ = impute_numeric_matrix(masked_numeric, method, config)
        for column_index, feature in enumerate(complete_numeric.columns):
            feature_mask = mask[:, column_index]
            truth = complete_numeric.loc[feature_mask, feature]
            prediction = imputed.loc[feature_mask, feature]
            rows.append(validation_metrics_by_feature(method, feature, truth, prediction))

    by_feature = pd.DataFrame(rows)
    summary = summarize_validation(by_feature)
    return by_feature, summary


def run_imputation_pipeline(df: pd.DataFrame, config: ImputationConfig) -> ImputationResult:
    validate_methods(config.methods, config.primary_method)
    prepared_df, density_audit = derive_planet_density(df)
    numeric_before = coerce_numeric_features(prepared_df, config.features)
    transformed_before, log_audit_obj = apply_log10_transform(numeric_before, config.log_features)
    log_transform_audit = log_audit_obj.to_frame()
    scaled, scaler = robust_scale(transformed_before)

    imputed_by_method: dict[str, pd.DataFrame] = {}
    constraint_audits: list[pd.DataFrame] = []
    for method in config.methods:
        scaled_imputed = _impute_scaled_matrix(method, scaled, config)
        transformed_imputed = invert_robust_scale(scaled_imputed, scaler)
        original_units = invert_log10_transform(transformed_imputed, config.log_features)
        bounded, constraint_audit = apply_feature_bounds(original_units, config.value_bounds)
        constraint_audit.insert(0, "method", method)
        constraint_audits.append(constraint_audit)
        imputed_by_method[method] = bounded

    output_by_method = {
        method: build_output_frame(prepared_df, imputed, transformed_before, method)
        for method, imputed in imputed_by_method.items()
    }

    missingness_audit = build_missingness_audit(
        numeric_before=numeric_before,
        transformed_before=transformed_before,
        imputed_by_method=imputed_by_method,
        density_audit=density_audit,
        log_audit=log_transform_audit,
    )
    complete_case_comparison = compare_to_complete_cases(
        numeric_before=numeric_before,
        transformed_before=transformed_before,
        imputed_by_method=imputed_by_method,
    )
    validation_by_feature, validation_summary = run_masked_validation(
        numeric_before=numeric_before,
        transformed_before=transformed_before,
        config=config,
    )

    constraint_audit_all = pd.concat(constraint_audits, ignore_index=True) if constraint_audits else pd.DataFrame()
    return ImputationResult(
        prepared_df=prepared_df,
        numeric_before=numeric_before,
        transformed_before=transformed_before,
        imputed_by_method=imputed_by_method,
        output_by_method=output_by_method,
        missingness_audit=missingness_audit,
        complete_case_comparison=complete_case_comparison,
        validation_by_feature=validation_by_feature,
        validation_summary=validation_summary,
        log_transform_audit=log_transform_audit,
        constraint_audit=constraint_audit_all,
        density_audit=density_audit,
    )


def write_imputation_outputs(
    result: ImputationResult,
    config: ImputationConfig,
    csv_path: Path,
    output_dir: Path,
    reports_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for method, output in result.output_by_method.items():
        path = output_dir / f"mapper_features_{method}_imputed.csv"
        output.to_csv(path, index=False)
        paths[f"data_{method}"] = path

    audit_paths = {
        "missingness_audit": reports_dir / "imputation_missingness.csv",
        "complete_case_comparison": reports_dir / "imputation_complete_case_comparison.csv",
        "validation_by_feature": reports_dir / "imputation_validation_by_feature.csv",
        "validation_summary": reports_dir / "imputation_validation_summary.csv",
        "log_transform_audit": reports_dir / "imputation_log_transform.csv",
        "constraint_audit": reports_dir / "imputation_constraint_clipping.csv",
        "density_audit": reports_dir / "imputation_density_derivation.csv",
    }

    result.missingness_audit.to_csv(audit_paths["missingness_audit"], index=False)
    result.complete_case_comparison.to_csv(audit_paths["complete_case_comparison"], index=False)
    result.validation_by_feature.to_csv(audit_paths["validation_by_feature"], index=False)
    result.validation_summary.to_csv(audit_paths["validation_summary"], index=False)
    result.log_transform_audit.to_csv(audit_paths["log_transform_audit"], index=False)
    result.constraint_audit.to_csv(audit_paths["constraint_audit"], index=False)
    pd.DataFrame([asdict(result.density_audit)]).to_csv(audit_paths["density_audit"], index=False)
    paths.update(audit_paths)

    config_path = reports_dir / "imputation_config.json"
    write_json(
        config_path,
        {
            "csv_path": str(csv_path),
            "features": list(config.features),
            "log_features": list(config.log_features),
            "methods": list(config.methods),
            "primary_method": config.primary_method,
            "n_neighbors": config.n_neighbors,
            "knn_weights": config.knn_weights,
            "iterative_max_iter": config.iterative_max_iter,
            "validation_mask_fraction": config.validation_mask_fraction,
            "validation_max_complete_rows": config.validation_max_complete_rows,
            "random_state": config.random_state,
            "value_bounds": dict(config.value_bounds),
            "physical_derivation": asdict(result.density_audit),
        },
    )
    paths["config"] = config_path
    return paths
