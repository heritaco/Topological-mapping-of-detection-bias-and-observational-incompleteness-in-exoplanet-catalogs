"""Pipeline orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.exoplanet_tda.core.config import prepare_run_config
from src.exoplanet_tda.core.logging_utils import setup_run_logger
from src.exoplanet_tda.core.manifest import ArtifactRegistry
from src.exoplanet_tda.core.run_context import RunContext

from .stage_registry import build_stage_registry
from .stages import StageResult


def create_context(
    config_path: str | Path | None,
    run_id: str | None = None,
    dry_run: bool | None = None,
    fail_fast: bool | None = None,
    use_gpu: str | None = None,
    overwrite: bool | None = None,
    extra_overrides: dict | None = None,
) -> RunContext:
    overrides = extra_overrides.copy() if extra_overrides else {}
    overrides.setdefault("run", {})
    if run_id is not None:
        overrides["run"]["run_id"] = run_id
    if dry_run is not None:
        overrides["run"]["dry_run"] = dry_run
    if fail_fast is not None:
        overrides["run"]["fail_fast"] = fail_fast
    if use_gpu is not None:
        overrides["run"]["use_gpu"] = use_gpu
    if overwrite is not None:
        overrides["run"]["overwrite"] = overwrite
    _, config, run_dir = prepare_run_config(config_path, overrides=overrides)
    repo_root = Path(config["paths"]["repo_root"]).resolve()
    logger = setup_run_logger(f"exoplanet_tda.{config['run']['run_id']}", run_dir / "logs" / "pipeline.log")
    registry = ArtifactRegistry(repo_root=repo_root, run_dir=run_dir, run_id=str(config["run"]["run_id"]))
    ctx = RunContext(
        repo_root=repo_root,
        run_id=str(config["run"]["run_id"]),
        run_dir=run_dir,
        config=config,
        logger=logger,
        registry=registry,
        random_state=int(config.get("project", {}).get("random_state", 42)),
        use_gpu=str(config.get("run", {}).get("use_gpu", "auto")).lower(),
        fail_fast=bool(config.get("run", {}).get("fail_fast", False)),
        dry_run=bool(config.get("run", {}).get("dry_run", False)),
    )
    ctx.log_artifact("config", run_dir / "config" / "config_original.yaml", "config", "Original submitted pipeline configuration")
    ctx.log_artifact("config", run_dir / "config" / "config_effective.yaml", "config", "Effective resolved pipeline configuration")
    ctx.log_artifact("logging", run_dir / "logs" / "pipeline.log", "log", "Unified pipeline log")
    return ctx


class PipelineOrchestrator:
    def __init__(self, stages: dict | None = None) -> None:
        self.stages = stages or build_stage_registry()

    def select_stage_names(self, requested: str | Iterable[str] | None = None, skip: Iterable[str] | None = None) -> list[str]:
        if requested is None or requested == "all":
            names = list(self.stages)
        elif isinstance(requested, str):
            names = [name.strip() for name in requested.split(",") if name.strip()]
        else:
            names = list(requested)
        skip_set = set(skip or [])
        return [name for name in names if name in self.stages and name not in skip_set]

    def run(self, ctx: RunContext, stages: str | Iterable[str] | None = None, skip: Iterable[str] | None = None) -> list[StageResult]:
        results: list[StageResult] = []
        selected = self.select_stage_names(stages, skip)
        if ctx.dry_run:
            ctx.write_json(
                "manifests/dry_run_plan.json",
                {
                    "run_id": ctx.run_id,
                    "stages": selected,
                    "inputs": ctx.config.get("inputs", {}),
                    "expected_stage_outputs": {name: ctx.config.get("stages", {}).get(name, {}).get("outputs_dir") for name in selected},
                },
            )
            ctx.log_artifact("dry_run", ctx.run_dir / "manifests" / "dry_run_plan.json", "manifest", "Dry-run execution plan")
        for name in selected:
            stage = self.stages[name]
            if not stage.should_run(ctx):
                result = StageResult(name=name, status="skipped", warnings=["Stage disabled by config"])
                ctx.finish_stage(name, result)
                results.append(result)
                continue
            try:
                ctx.logger.info("Starting stage: %s", name)
                result = stage.run(ctx)
            except Exception as exc:  # noqa: BLE001
                ctx.logger.exception("Stage failed: %s", name)
                result = StageResult(name=name, status="failed", error=str(exc))
            ctx.finish_stage(name, result)
            results.append(result)
            if result.status == "failed" and ctx.fail_fast:
                break
        ctx.registry.save()
        return results
