"""Lightweight feature ablation runner for candidate characterization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd

from src.candidate_characterization.config import CharacterizationConfig, load_config
from src.candidate_characterization.io import load_catalog
from src.candidate_characterization.validation import validate_property_models
from src.exoplanet_tda.core.io import ensure_dir

from .registry import load_feature_registry


def _experiment_rows(feature_sets_path: str | Path | None = None, registry_path: str | Path | None = None) -> list[dict[str, str]]:
    registry = load_feature_registry(registry_path, feature_sets_path)
    doc = registry.feature_sets_doc.get("feature_ablation_plan", {}) or {}
    return [{"name": str(row["name"]), "feature_set": str(row["feature_set"])} for row in doc.get("experiments", []) or []]


def run_ablation(
    cfg: CharacterizationConfig,
    run_id: str,
    output_root: Path,
    registry_path: str | Path = "configs/features/feature_registry.yaml",
    feature_sets_path: str | Path = "configs/features/feature_sets.yaml",
) -> tuple[pd.DataFrame, Path]:
    catalog, _ = load_catalog(cfg)
    repo_root = Path(cfg.paths.repo_root).resolve()
    registry_path = Path(registry_path)
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path
    feature_sets_path = Path(feature_sets_path)
    if not feature_sets_path.is_absolute():
        feature_sets_path = repo_root / feature_sets_path
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []
    for experiment in _experiment_rows(feature_sets_path, registry_path):
        feature_set = experiment["feature_set"]
        try:
            metrics, _ = validate_property_models(
                catalog,
                quantiles=cfg.model.quantiles,
                prefer_gpu=cfg.model.prefer_gpu,
                random_state=cfg.model.random_state,
                mode="multiplanet",
                test_size=cfg.model.test_size,
                feature_set=feature_set,
                registry_path=str(registry_path),
                feature_sets_path=str(feature_sets_path),
            )
            for _, metric in metrics.iterrows():
                row = metric.to_dict()
                row.update(
                    {
                        "experiment": experiment["name"],
                        "feature_set": feature_set,
                        "n_features_available": int(metric.get("n_features_available", 0)),
                        "n_rows_train": int(metric.get("n_rows_train", 0)),
                        "warnings": "",
                    }
                )
                rows.append(row)
        except Exception as exc:  # noqa: BLE001
            warning = f"{experiment['name']} failed: {exc}"
            warnings.append(warning)
            rows.append(
                {
                    "experiment": experiment["name"],
                    "target": "",
                    "feature_set": feature_set,
                    "n_features_available": 0,
                    "n_rows_train": 0,
                    "warnings": warning,
                }
            )

    table_dir = ensure_dir(output_root / "runs" / run_id / "tables" / "feature_ablation")
    report_dir = ensure_dir(output_root / "runs" / run_id / "reports" / "stage_summaries")
    metrics_df = pd.DataFrame(rows)
    metrics_path = table_dir / "ablation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    report_path = report_dir / "feature_ablation.md"
    report_path.write_text(
        "# Feature Ablation\n\n"
        f"- Experiments evaluated: {len(rows)}\n"
        f"- Warnings: {'; '.join(warnings) if warnings else 'none'}\n"
        "- Interpretation: ablation supports sensitivity analysis; it does not prove that more variables improve the model.\n",
        encoding="utf-8",
    )
    return metrics_df, metrics_path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run lightweight candidate-characterization feature ablations.")
    parser.add_argument("--config", default="configs/candidate_characterization/default.yaml")
    parser.add_argument("--run-id", default="feature_ablation")
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--feature-registry", default="configs/features/feature_registry.yaml")
    parser.add_argument("--feature-sets", default="configs/features/feature_sets.yaml")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.cpu:
        cfg.model.prefer_gpu = False
    _, metrics_path = run_ablation(
        cfg,
        run_id=args.run_id,
        output_root=Path(args.output_root),
        registry_path=args.feature_registry,
        feature_sets_path=args.feature_sets,
    )
    print(f"[feature_ablation] wrote metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
