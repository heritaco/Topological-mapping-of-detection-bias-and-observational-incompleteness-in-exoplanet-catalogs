"""Standalone model training CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from .config import load_config
from .io import load_catalog
from .models import train_models
from .utils import ensure_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train candidate property models.")
    p.add_argument("--config", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--catalog-csv", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.repo_root:
        cfg.paths.repo_root = args.repo_root
    if args.catalog_csv:
        cfg.paths.catalog_csv = args.catalog_csv
    if args.model_dir:
        cfg.paths.model_dir = args.model_dir
    if args.cpu:
        cfg.model.prefer_gpu = False
    repo_root = Path(cfg.paths.repo_root).resolve()
    model_dir = ensure_dir(repo_root / cfg.paths.model_dir)
    catalog, path = load_catalog(cfg)
    print(f"Loaded catalog {path} rows={len(catalog)}")
    models = train_models(catalog, cfg.model.quantiles, cfg.model.prefer_gpu, cfg.model.random_state, cfg.model.calibration_cv)
    models.save(model_dir)
    print(f"Saved models to {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
