"""Standalone validation CLI."""
from __future__ import annotations

import argparse
from pathlib import Path
from .config import load_config
from .io import load_catalog
from .validation import validate_property_models, write_validation_outputs
from .utils import ensure_dir


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Validate candidate property models with random, temporal, or multiplanet split.")
    p.add_argument("--config", default=None)
    p.add_argument("--repo-root", default=None)
    p.add_argument("--catalog-csv", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--mode", choices=["random", "temporal", "multiplanet"], default="multiplanet")
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.repo_root:
        cfg.paths.repo_root = args.repo_root
    if args.catalog_csv:
        cfg.paths.catalog_csv = args.catalog_csv
    if args.output_dir:
        cfg.paths.output_dir = args.output_dir
    if args.cpu:
        cfg.model.prefer_gpu = False
    repo_root = Path(cfg.paths.repo_root).resolve()
    output_dir = ensure_dir(repo_root / cfg.paths.output_dir)
    catalog, _ = load_catalog(cfg)
    metrics, details = validate_property_models(catalog, cfg.model.quantiles, cfg.model.prefer_gpu, cfg.model.random_state, mode=args.mode, test_size=cfg.model.test_size)
    write_validation_outputs(metrics, details, output_dir)
    print(metrics.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
