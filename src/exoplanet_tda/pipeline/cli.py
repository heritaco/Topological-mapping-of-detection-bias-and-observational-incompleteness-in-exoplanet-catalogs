"""Command-line interface for unified exoplanet TDA runs."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.exoplanet_tda.core.manifest import ArtifactRegistry
from src.exoplanet_tda.reporting.summaries import index_legacy_outputs, status_summary

from .orchestrator import PipelineOrchestrator, create_context


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified exoplanet TDA experiment pipeline.")
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--stages", default="all")
    parser.add_argument("--only-stage", default=None)
    parser.add_argument("--skip-stage", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--index-legacy-outputs", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-gpu", choices=["auto", "true", "false"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.status:
        run_dir = Path(args.run_dir) if args.run_dir else Path("outputs") / "runs" / str(args.run_id)
        manifest_path = run_dir / "manifests" / "run_manifest.json"
        print(status_summary(ArtifactRegistry.load(manifest_path)))
        return 0

    ctx = create_context(
        args.config,
        run_id=args.run_id,
        dry_run=args.dry_run,
        fail_fast=args.fail_fast,
        use_gpu=args.use_gpu,
        overwrite=args.overwrite,
    )
    if args.index_legacy_outputs:
        index_legacy_outputs(ctx)
        print(f"Indexed legacy outputs into {ctx.repo_relative(ctx.run_dir / 'artifacts' / 'legacy_links.json')}")
        return 0

    stages = args.only_stage or args.stages
    skip = [item.strip() for item in args.skip_stage.split(",")] if args.skip_stage else None
    results = PipelineOrchestrator().run(ctx, stages=stages, skip=skip)
    print(f"Run {ctx.run_id} completed in {ctx.repo_relative(ctx.run_dir)}")
    for result in results:
        print(f"- {result.name}: {result.status}")
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
