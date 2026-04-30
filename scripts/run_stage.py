from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exoplanet_tda.pipeline.orchestrator import PipelineOrchestrator, create_context


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one unified exoplanet TDA stage.")
    parser.add_argument("--stage", required=True)
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--use-gpu", choices=["auto", "true", "false"], default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ctx = create_context(args.config, args.run_id, args.dry_run, args.fail_fast, args.use_gpu, args.overwrite)
    results = PipelineOrchestrator().run(ctx, stages=args.stage)
    for result in results:
        print(f"{result.name}: {result.status}")
    return 1 if any(result.status == "failed" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
