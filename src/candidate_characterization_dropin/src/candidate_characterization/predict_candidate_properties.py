"""Standalone candidate prediction CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from .config import load_config
from .io import load_catalog, load_candidates
from .models import TrainedCharacterizationModels, train_models
from .predict import predict_candidates, write_prediction_outputs
from .summarize import build_markdown_summary
from .utils import ensure_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict probabilistic properties for candidate missing planets.")
    p.add_argument("--config", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--catalog-csv", default=None)
    p.add_argument("--candidates-csv", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--report-dir", default=None)
    p.add_argument("--train-if-missing", action="store_true")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.repo_root:
        cfg.paths.repo_root = args.repo_root
    if args.catalog_csv:
        cfg.paths.catalog_csv = args.catalog_csv
    if args.candidates_csv:
        cfg.paths.candidates_csv = args.candidates_csv
    if args.model_dir:
        cfg.paths.model_dir = args.model_dir
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.report_dir:
        cfg.paths.report_dir = args.report_dir
    if args.cpu:
        cfg.model.prefer_gpu = False
    repo_root = Path(cfg.paths.repo_root).resolve()
    output_dir = ensure_dir(repo_root / cfg.paths.output_dir)
    report_dir = ensure_dir(repo_root / cfg.paths.report_dir)
    model_dir = ensure_dir(repo_root / cfg.paths.model_dir)
    catalog, _ = load_catalog(cfg)
    if (model_dir / "candidate_characterization_models.joblib").exists():
        models = TrainedCharacterizationModels.load(model_dir)
    elif args.train_if_missing:
        models = train_models(catalog, cfg.model.quantiles, cfg.model.prefer_gpu, cfg.model.random_state, cfg.model.calibration_cv)
        models.save(model_dir)
    else:
        raise FileNotFoundError(f"No models at {model_dir}. Run train_property_models or pass --train-if-missing.")
    candidates, _ = load_candidates(cfg, catalog)
    predictions, analog_neighbors = predict_candidates(candidates, catalog, models, cfg.model.quantiles, cfg.model.analog_k, cfg.model.analog_temperature)
    write_prediction_outputs(predictions, analog_neighbors, output_dir)
    build_markdown_summary(predictions, report_dir)
    print(f"Wrote candidate characterization outputs to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
