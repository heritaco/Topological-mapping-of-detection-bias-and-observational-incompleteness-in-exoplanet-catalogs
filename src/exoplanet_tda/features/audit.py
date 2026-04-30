"""Feature availability and missingness audit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.exoplanet_tda.core.io import ensure_dir
from src.exoplanet_tda.core.paths import resolve_repo_path

from .derived import add_derived_features
from .leakage import AUDIT_ONLY_FEATURES
from .registry import FeatureRegistry, load_feature_registry


def _load_csv(path: Path | None, warnings: list[str]) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Could not read {path}: {exc}")
        return None


def _available_frame(frames: list[tuple[str, pd.DataFrame]], feature: str, aliases: tuple[str, ...]) -> tuple[str | None, pd.Series | None, int]:
    names = (feature, *aliases)
    max_rows = max((len(frame) for _, frame in frames), default=0)
    for frame_name, frame in frames:
        for name in names:
            if name in frame.columns:
                return frame_name, frame[name], len(frame)
    return None, None, max_rows


def build_feature_audit_tables(
    frames: list[tuple[str, pd.DataFrame]],
    registry: FeatureRegistry,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for spec in registry.all_specs():
        frame_name, series, row_count = _available_frame(frames, spec.name, spec.aliases)
        available = series is not None
        missing_count = int(series.isna().sum()) if series is not None else int(row_count)
        missing_pct = float(missing_count / row_count) if row_count else float("nan")
        action = spec.recommended_action
        if spec.name in AUDIT_ONLY_FEATURES or spec.role == "audit":
            action = "audit-only by default; use for observational audit, stratified validation, or sensitivity analysis"
        elif available and missing_pct > 0.8:
            action = "high missingness; use only in ablation or sensitivity analysis unless imputation is justified"
        elif not available:
            action = "missing from current inputs; do not fail pipeline, report absence"
        rows.append(
            {
                "feature_name": spec.name,
                "feature_group": spec.group,
                "available": bool(available),
                "source_table": frame_name or "",
                "missing_count": missing_count,
                "missing_percentage": missing_pct,
                "role": spec.role,
                "leakage_risk": spec.leakage_risk,
                "recommended_action": action,
            }
        )
    availability = pd.DataFrame(rows)
    missingness = availability[
        [
            "feature_name",
            "feature_group",
            "source_table",
            "missing_count",
            "missing_percentage",
            "available",
            "role",
            "leakage_risk",
            "recommended_action",
        ]
    ].copy()
    summary = (
        availability.groupby("feature_group", as_index=False)
        .agg(
            n_features=("feature_name", "count"),
            n_available=("available", "sum"),
            mean_missing_percentage=("missing_percentage", "mean"),
            max_leakage_risk=("leakage_risk", lambda values: "high" if "high" in set(values) else "medium" if "medium" in set(values) else "low"),
        )
        .sort_values("feature_group")
    )
    return availability, missingness, summary


def write_feature_audit(
    availability: pd.DataFrame,
    missingness: pd.DataFrame,
    summary: pd.DataFrame,
    table_dir: Path,
    report_dir: Path,
    run_id: str,
) -> dict[str, Path]:
    ensure_dir(table_dir)
    ensure_dir(report_dir)
    availability_path = table_dir / "feature_availability.csv"
    missingness_path = table_dir / "feature_missingness.csv"
    summary_path = table_dir / "feature_set_summary.csv"
    report_path = report_dir / "feature_audit.md"
    availability.to_csv(availability_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    summary.to_csv(summary_path, index=False)

    n_features = int(len(availability))
    n_available = int(availability["available"].sum()) if n_features else 0
    high_missing = availability[(availability["available"]) & (availability["missing_percentage"] > 0.8)]["feature_name"].tolist()
    audit_only = availability[availability["role"].eq("audit")]["feature_name"].tolist()
    lines = [
        f"# Feature Audit: {run_id}",
        "",
        "- Scope: feature governance, leakage-safe features, ablation, observational audit, and sensitivity analysis.",
        f"- Features registered: {n_features}",
        f"- Features available in current inputs: {n_available}",
        f"- High-missingness available features: {', '.join(high_missing) if high_missing else 'none'}",
        f"- Audit-only variables: {', '.join(audit_only) if audit_only else 'none'}",
        "",
        "This audit does not claim that additional variables improve probabilistic characterization or candidate prioritization.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "availability": availability_path,
        "missingness": missingness_path,
        "summary": summary_path,
        "report": report_path,
    }


def run_feature_audit(ctx) -> tuple[dict[str, Path], dict[str, Any], list[str]]:
    cfg = ctx.config.get("stages", {}).get("feature_audit", {}) or {}
    registry_path = resolve_repo_path(ctx.repo_root, cfg.get("feature_registry", "configs/features/feature_registry.yaml"))
    feature_sets_path = resolve_repo_path(ctx.repo_root, cfg.get("feature_sets", "configs/features/feature_sets.yaml"))
    registry = load_feature_registry(registry_path, feature_sets_path)
    warnings: list[str] = []

    frames: list[tuple[str, pd.DataFrame]] = []
    inputs = ctx.config.get("inputs", {}) or {}
    for label, key in [("raw_catalog", "raw_catalog"), ("imputed_catalog", "imputed_catalog"), ("system_candidates", "system_candidates_csv")]:
        path_value = inputs.get(key)
        path = resolve_repo_path(ctx.repo_root, path_value) if path_value else None
        frame = None if ctx.dry_run else _load_csv(path, warnings)
        if frame is not None:
            frames.append((label, add_derived_features(frame, logger=ctx.logger)))
    if ctx.dry_run:
        frames = []

    availability, missingness, summary = build_feature_audit_tables(frames, registry)
    paths = write_feature_audit(availability, missingness, summary, ctx.table_dir("feature_audit"), ctx.report_dir("feature_audit"), ctx.run_id)
    metrics = {
        "features_registered": int(len(availability)),
        "features_available": int(availability["available"].sum()) if len(availability) else 0,
        "dry_run": bool(ctx.dry_run),
    }
    return paths, metrics, warnings
