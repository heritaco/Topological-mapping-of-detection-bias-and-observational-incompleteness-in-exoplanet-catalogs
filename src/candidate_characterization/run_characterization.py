"""End-to-end CLI for topological candidate characterization.

Example:
    python -m src.candidate_characterization.run_characterization --repo-root . --train --predict --validate
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback
import pandas as pd

from .config import CharacterizationConfig, load_config
from .gpu import detect_accelerator
from .io import load_catalog, load_candidates
from .models import train_models, TrainedCharacterizationModels
from .predict import predict_candidates, write_prediction_outputs
from .summarize import build_markdown_summary
from .utils import ensure_dir, write_json
from .validation import validate_property_models, write_validation_outputs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Probabilistically characterize topologically prioritized missing-planet candidates.")
    p.add_argument("--config", default=None, help="Optional YAML/JSON config file.")
    p.add_argument("--repo-root", default=None, help="Repository root. Defaults to config or current directory.")
    p.add_argument("--catalog-csv", default=None, help="Observed/imputed PSCompPars catalog CSV.")
    p.add_argument("--candidates-csv", default=None, help="Candidate missing-planets CSV from system_missing_planets or custom file.")
    p.add_argument("--output-dir", default=None, help="Output directory. Default outputs/candidate_characterization.")
    p.add_argument("--report-dir", default=None, help="Report directory. Default reports/candidate_characterization.")
    p.add_argument("--model-dir", default=None, help="Model directory. Default outputs/candidate_characterization/models.")
    p.add_argument("--cpu", action="store_true", help="Disable GPU attempts even if CUDA is available.")
    p.add_argument("--train", action="store_true", help="Train and save models.")
    p.add_argument("--predict", action="store_true", help="Predict candidate properties.")
    p.add_argument("--validate", action="store_true", help="Run holdout validation.")
    p.add_argument("--validation-mode", choices=["random", "temporal", "multiplanet"], default="multiplanet")
    p.add_argument("--force-retrain", action="store_true", help="Retrain even if saved models exist.")
    p.add_argument("--analog-k", type=int, default=None, help="Number of analog planets per candidate.")
    p.add_argument("--random-state", type=int, default=None)
    return p.parse_args(argv)


def apply_arg_overrides(cfg: CharacterizationConfig, args) -> CharacterizationConfig:
    if args.repo_root is not None:
        cfg.paths.repo_root = args.repo_root
    if args.catalog_csv is not None:
        cfg.paths.catalog_csv = args.catalog_csv
    if args.candidates_csv is not None:
        cfg.paths.candidates_csv = args.candidates_csv
    if args.output_dir is not None:
        cfg.paths.output_dir = args.output_dir
    if args.report_dir is not None:
        cfg.paths.report_dir = args.report_dir
    if args.model_dir is not None:
        cfg.paths.model_dir = args.model_dir
    if args.cpu:
        cfg.model.prefer_gpu = False
    if args.analog_k is not None:
        cfg.model.analog_k = args.analog_k
    if args.random_state is not None:
        cfg.model.random_state = args.random_state
    return cfg


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = apply_arg_overrides(load_config(args.config), args)
    repo_root = Path(cfg.paths.repo_root).resolve()
    output_dir = repo_root / cfg.paths.output_dir
    report_dir = repo_root / cfg.paths.report_dir
    model_dir = repo_root / cfg.paths.model_dir
    ensure_dir(output_dir)
    ensure_dir(report_dir)
    ensure_dir(model_dir)

    print("[candidate_characterization] accelerator:", detect_accelerator(cfg.model.prefer_gpu).to_dict())
    cfg.save_json(output_dir / "config_used.json")
    run_notes: list[str] = []

    catalog, catalog_path = load_catalog(cfg)
    print(f"[candidate_characterization] loaded catalog: {catalog_path} rows={len(catalog)}")
    run_notes.append(f"Catalog source: {catalog_path}")

    do_train = args.train or not (model_dir / "candidate_characterization_models.joblib").exists() or args.force_retrain
    models = None
    if do_train:
        print("[candidate_characterization] training models...")
        models = train_models(
            catalog,
            quantiles=cfg.model.quantiles,
            prefer_gpu=cfg.model.prefer_gpu,
            random_state=cfg.model.random_state,
            calibration_cv=cfg.model.calibration_cv,
        )
        models.save(model_dir)
        print(f"[candidate_characterization] saved models: {model_dir}")
    else:
        models = TrainedCharacterizationModels.load(model_dir)
        print(f"[candidate_characterization] loaded models: {model_dir}")

    validation_metrics = None
    if args.validate:
        print(f"[candidate_characterization] validating mode={args.validation_mode}...")
        validation_metrics, validation_details = validate_property_models(
            catalog,
            quantiles=cfg.model.quantiles,
            prefer_gpu=cfg.model.prefer_gpu,
            random_state=cfg.model.random_state,
            mode=args.validation_mode,
            test_size=cfg.model.test_size,
        )
        paths = write_validation_outputs(validation_metrics, validation_details, output_dir)
        print(f"[candidate_characterization] wrote validation: {paths}")
        run_notes.append(f"Validation completed in {args.validation_mode} mode.")

    predictions = None
    if args.predict or not args.train:
        try:
            candidates, candidates_path = load_candidates(cfg, catalog=catalog)
            print(f"[candidate_characterization] loaded candidates: {candidates_path} rows={len(candidates)}")
            run_notes.append(f"Candidate source: {candidates_path}")
            predictions, analog_neighbors = predict_candidates(
                candidates,
                catalog,
                models,
                quantiles=cfg.model.quantiles,
                analog_k=cfg.model.analog_k,
                analog_temperature=cfg.model.analog_temperature,
            )
            paths = write_prediction_outputs(predictions, analog_neighbors, output_dir)
            print(f"[candidate_characterization] wrote predictions: {paths}")
            run_notes.append(f"Predictions generated for {len(predictions)} prioritized candidates.")
        except FileNotFoundError as exc:
            warning = f"Prediction step skipped because candidate inputs were not found: {exc}"
            print(f"[candidate_characterization] WARNING: {warning}")
            run_notes.append(warning)
            predictions = None

    summary_path = build_markdown_summary(
        predictions if predictions is not None else pd.DataFrame(),
        report_dir,
        validation_metrics,
        notes=run_notes,
    )
    print(f"[candidate_characterization] wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # command-line readability
        print("[candidate_characterization] ERROR:", exc, file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(1)
