from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.exoplanet_tda.pipeline.orchestrator import create_context
from src.exoplanet_tda.reporting.summaries import index_legacy_outputs


def main() -> int:
    parser = argparse.ArgumentParser(description="Index legacy repository outputs into a unified run manifest.")
    parser.add_argument("--config", default="configs/pipeline/default.yaml")
    parser.add_argument("--run-id", default="legacy_index")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    ctx = create_context(args.config, run_id=args.run_id, overwrite=args.overwrite)
    path = index_legacy_outputs(ctx)
    print(f"Legacy index written to {ctx.repo_relative(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
