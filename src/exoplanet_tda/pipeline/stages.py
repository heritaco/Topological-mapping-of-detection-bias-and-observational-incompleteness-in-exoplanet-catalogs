"""Pipeline stage definitions and thin legacy adapters."""

from __future__ import annotations

import shutil
import sys
import time
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.exoplanet_tda.core.io import ensure_dir, write_json
from src.exoplanet_tda.core.paths import resolve_repo_path
from src.exoplanet_tda.core.run_context import RunContext
from src.exoplanet_tda.core.subprocess_utils import run_command


@dataclass
class StageResult:
    name: str
    status: str
    outputs: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "outputs": self.outputs,
            "metrics": self.metrics,
            "warnings": self.warnings,
            "error": self.error,
            "elapsed_seconds": self.elapsed_seconds,
        }


class PipelineStage:
    name: str = "stage"

    def should_run(self, ctx: RunContext) -> bool:
        return bool(ctx.config.get("stages", {}).get(self.name, {}).get("enabled", False))

    def run(self, ctx: RunContext) -> StageResult:
        raise NotImplementedError


def artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv", ".parquet", ".xlsx"}:
        return "table"
    if suffix in {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".html"}:
        return "figure"
    if suffix in {".joblib", ".pkl", ".pickle"}:
        return "model"
    if suffix in {".md", ".tex"}:
        return "report"
    if suffix in {".yaml", ".yml", ".json"}:
        return "config" if "config" in path.name else "manifest"
    return "other"


def register_tree(ctx: RunContext, stage: str, root: Path, description: str) -> tuple[list[dict[str, Any]], list[str]]:
    outputs: list[dict[str, Any]] = []
    warnings: list[str] = []
    if not root.exists():
        warnings.append(f"Output directory not found: {ctx.repo_relative(root)}")
        return outputs, warnings
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        entry = ctx.log_artifact(stage, path, artifact_kind(path), description, {"source": "legacy_index"})
        outputs.append(entry)
    return outputs, warnings


def copy_if_exists(ctx: RunContext, stage: str, source: Path, destination: Path, kind: str, description: str) -> dict[str, Any] | None:
    if not source.exists():
        return None
    ensure_dir(destination.parent)
    shutil.copy2(source, destination)
    return ctx.log_artifact(stage, destination, kind, description, {"source_path": ctx.repo_relative(source)})


class DataAuditStage(PipelineStage):
    name = "data_audit"

    def run(self, ctx: RunContext) -> StageResult:
        start = time.perf_counter()
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        inputs = ctx.config.get("inputs", {})
        for key, value in inputs.items():
            if not isinstance(value, str):
                continue
            path = resolve_repo_path(ctx.repo_root, value)
            exists = bool(path and path.exists())
            row: dict[str, Any] = {
                "name": key,
                "path": path.as_posix() if path else value,
                "relative_path": ctx.repo_relative(path) if path else value,
                "exists": exists,
                "kind": "input",
                "rows": None,
                "columns": None,
            }
            if exists and path and path.suffix.lower() == ".csv":
                try:
                    if ctx.dry_run:
                        raise RuntimeError("CSV shape inspection skipped in dry-run mode")
                    import pandas as pd

                    frame = pd.read_csv(path)
                    row["rows"] = int(len(frame))
                    row["columns"] = int(len(frame.columns))
                except Exception as exc:  # noqa: BLE001
                    if ctx.dry_run:
                        row["rows"] = "dry_run"
                        row["columns"] = "dry_run"
                    else:
                        warning = f"Could not read CSV input {key}: {exc}"
                        warnings.append(warning)
                        ctx.warn(self.name, warning)
            elif not exists:
                warnings.append(f"Missing configured input: {key}")
            rows.append(row)

        for stage_name, stage_cfg in (ctx.config.get("stages") or {}).items():
            out_value = stage_cfg.get("outputs_dir") if isinstance(stage_cfg, dict) else None
            if out_value:
                out_path = resolve_repo_path(ctx.repo_root, out_value)
                rows.append(
                    {
                        "name": f"{stage_name}.outputs_dir",
                        "path": out_path.as_posix() if out_path else out_value,
                        "relative_path": ctx.repo_relative(out_path) if out_path else out_value,
                        "exists": bool(out_path and out_path.exists()),
                        "kind": "legacy_output_dir",
                        "rows": None,
                        "columns": None,
                    }
                )

        table_path = ctx.table_dir(self.name) / "input_inventory.csv"
        with table_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["name", "path", "relative_path", "exists", "kind", "rows", "columns"])
            writer.writeheader()
            writer.writerows(rows)
        summary_path = ctx.report_dir(self.name) / "data_audit.md"
        summary_path.write_text(
            "# Data Audit\n\n"
            f"- Inputs inventoried: {len(rows)}\n"
            f"- Missing items: {sum(not bool(row['exists']) for row in rows)}\n"
            "- Interpretation: this audit verifies availability and shape; it does not validate scientific conclusions.\n",
            encoding="utf-8",
        )
        outputs = [
            ctx.log_artifact(self.name, table_path, "table", "Input and legacy-output inventory"),
            ctx.log_artifact(self.name, summary_path, "report", "Data audit stage summary"),
        ]
        return StageResult(self.name, "success", outputs, {"items": len(rows)}, warnings, elapsed_seconds=time.perf_counter() - start)


class LegacyOutputStage(PipelineStage):
    module: str | None = None
    script: str | None = None
    default_config_key = "config"

    def __init__(self, name: str, module: str | None = None, script: str | None = None) -> None:
        self.name = name
        self.module = module
        self.script = script

    def command(self, ctx: RunContext) -> list[str]:
        cfg = ctx.config["stages"].get(self.name, {})
        if self.module:
            command = [sys.executable, "-m", self.module]
        elif self.script:
            command = [sys.executable, self.script]
        else:
            command = []
        if cfg.get(self.default_config_key):
            command.extend(["--config", str(cfg[self.default_config_key])])
        return command

    def run(self, ctx: RunContext) -> StageResult:
        start = time.perf_counter()
        cfg = ctx.config.get("stages", {}).get(self.name, {})
        outputs_dir = resolve_repo_path(ctx.repo_root, cfg.get("outputs_dir")) if cfg.get("outputs_dir") else None
        warnings: list[str] = []
        error: str | None = None

        if ctx.dry_run:
            expected = []
            if outputs_dir:
                expected.append(ctx.log_artifact(self.name, outputs_dir, "other", "Expected legacy output directory", {"dry_run": True}))
            return StageResult(self.name, "skipped", expected, {"dry_run": True}, elapsed_seconds=time.perf_counter() - start)

        rerun = bool(cfg.get("rerun", False))
        if rerun:
            command = self.command(ctx)
            if command:
                proc = run_command(command, ctx.repo_root, ctx.run_dir / "logs" / "pipeline.log")
                if proc.returncode != 0:
                    error = f"Legacy command failed with exit code {proc.returncode}: {' '.join(command)}"
                    warnings.append(error)
            else:
                warnings.append("No command configured for legacy stage rerun.")

        outputs: list[dict[str, Any]] = []
        if outputs_dir:
            indexed, index_warnings = register_tree(ctx, self.name, outputs_dir, f"Legacy outputs for {self.name}")
            outputs.extend(indexed)
            warnings.extend(index_warnings)
        status = "success" if outputs and not error else "partial" if outputs or warnings else "skipped"
        if error:
            status = "failed" if ctx.fail_fast else "partial"
        return StageResult(self.name, status, outputs, {"artifact_count": len(outputs)}, warnings, error, time.perf_counter() - start)


class ImputationStage(LegacyOutputStage):
    def __init__(self) -> None:
        super().__init__("imputation", script="src/impute_exodata.py")

    def run(self, ctx: RunContext) -> StageResult:
        start = time.perf_counter()
        cfg = ctx.config.get("stages", {}).get(self.name, {})
        imputed = resolve_repo_path(ctx.repo_root, ctx.config.get("inputs", {}).get("imputed_catalog"))
        outputs: list[dict[str, Any]] = []
        warnings: list[str] = []
        if imputed and imputed.exists():
            outputs.append(ctx.log_artifact(self.name, imputed, "table", "Existing imputed catalog"))
            return StageResult(self.name, "success", outputs, {"artifact_count": 1}, elapsed_seconds=time.perf_counter() - start)
        if not cfg.get("enabled", False) or ctx.dry_run:
            warnings.append("Imputed catalog not found; heavy imputation was not run.")
            return StageResult(self.name, "skipped", outputs, {}, warnings, elapsed_seconds=time.perf_counter() - start)
        return super().run(ctx)


class MapperStage(LegacyOutputStage):
    def __init__(self) -> None:
        super().__init__("mapper", script="src/mapper_exodata.py")

    def command(self, ctx: RunContext) -> list[str]:
        cfg = ctx.config["stages"].get(self.name, {})
        spaces = cfg.get("spaces") or ["orbital"]
        lens = (cfg.get("lenses") or ["pca2"])[0]
        command = [
            sys.executable,
            "src/mapper_exodata.py",
            "--outputs-dir",
            str(cfg.get("outputs_dir", "outputs/mapper")),
            "--space",
            spaces[0] if len(spaces) == 1 else "all",
            "--lens",
            lens,
            "--n-cubes",
            str(cfg.get("n_cubes", 10)),
            "--overlap",
            str(cfg.get("overlap", 0.35)),
            "--fast",
        ]
        return command


class SystemMissingPlanetsStage(LegacyOutputStage):
    def __init__(self) -> None:
        super().__init__("system_missing_planets", module="src.system_missing_planets.run_system_missing_planets")

    def command(self, ctx: RunContext) -> list[str]:
        cfg = ctx.config["stages"].get(self.name, {})
        inputs = ctx.config.get("inputs", {})
        return [
            sys.executable,
            "-m",
            "src.system_missing_planets.run_system_missing_planets",
            "--catalog",
            str(inputs.get("imputed_catalog") or inputs.get("raw_catalog")),
            "--output-dir",
            str(cfg.get("outputs_dir", "outputs/system_missing_planets")),
            "--mode",
            str(cfg.get("mode", "all")),
            "--random-state",
            str(ctx.random_state),
            "--toi-table",
            str(inputs.get("toi_regions_csv", "")),
            "--ati-table",
            str(inputs.get("ati_anchors_csv", "")),
        ]


class CandidateCharacterizationStage(LegacyOutputStage):
    def __init__(self) -> None:
        super().__init__("candidate_characterization", module="src.candidate_characterization.run_characterization")

    def command(self, ctx: RunContext) -> list[str]:
        cfg = ctx.config["stages"].get(self.name, {})
        command = [
            sys.executable,
            "-m",
            str(cfg.get("module", "src.candidate_characterization.run_characterization")),
            "--config",
            str(cfg.get("config", "configs/candidate_characterization/default.yaml")),
            "--repo-root",
            ".",
        ]
        if cfg.get("train", True):
            command.append("--train")
        if cfg.get("predict", True):
            command.append("--predict")
        if cfg.get("validate", True):
            command.append("--validate")
        command.extend(["--validation-mode", str(cfg.get("validation_mode", "multiplanet"))])
        if ctx.use_gpu == "false":
            command.append("--cpu")
        return command

    def run(self, ctx: RunContext) -> StageResult:
        result = super().run(ctx)
        if ctx.dry_run:
            return result
        cfg = ctx.config.get("stages", {}).get(self.name, {})
        output_root = resolve_repo_path(ctx.repo_root, cfg.get("outputs_dir", "outputs/candidate_characterization"))
        report_root = resolve_repo_path(ctx.repo_root, cfg.get("report_dir", "reports/candidate_characterization"))
        mirrored: list[dict[str, Any]] = []
        known = [
            (output_root / "tables" / "candidate_property_predictions.csv", ctx.table_dir(self.name) / "candidate_property_predictions.csv", "table"),
            (output_root / "tables" / "candidate_analog_neighbors.csv", ctx.table_dir(self.name) / "candidate_analog_neighbors.csv", "table"),
            (output_root / "tables" / "validation_metrics.csv", ctx.table_dir(self.name) / "validation_metrics.csv", "table"),
            (output_root / "tables" / "validation_predictions.csv", ctx.table_dir(self.name) / "validation_predictions.csv", "table"),
            (output_root / "models" / "candidate_characterization_models.joblib", ctx.model_dir(self.name) / "candidate_characterization_models.joblib", "model"),
            (output_root / "models" / "model_metadata.json", ctx.model_dir(self.name) / "model_metadata.json", "model"),
            (report_root / "candidate_characterization_summary.md", ctx.report_dir(self.name) / "candidate_characterization_summary.md", "report"),
        ]
        for source, dest, kind in known:
            copied = copy_if_exists(ctx, self.name, source, dest, kind, "Mirrored candidate characterization artifact")
            if copied:
                mirrored.append(copied)
        result.outputs.extend(mirrored)
        result.metrics["mirrored_artifact_count"] = len(mirrored)
        if not result.outputs and not result.warnings:
            result.status = "skipped"
        return result


class ReportingStage(PipelineStage):
    name = "reporting"

    def run(self, ctx: RunContext) -> StageResult:
        start = time.perf_counter()
        manifest = ctx.registry.to_dict()
        lines = [
            f"# Unified Run Summary: {ctx.run_id}",
            "",
            f"- Run directory: `{ctx.repo_relative(ctx.run_dir)}`",
            f"- Timestamp: {manifest['updated_at']}",
            "",
            "## Stage Status",
            "",
        ]
        for stage_name, result in manifest.get("stages", {}).items():
            lines.append(f"- `{stage_name}`: {result.get('status', 'unknown')}")
        lines.extend(["", "## Key Artifacts", ""])
        for artifact in manifest.get("artifacts", [])[:40]:
            lines.append(f"- `{artifact['kind']}` `{artifact['relative_path']}`")
        if manifest.get("warnings"):
            lines.extend(["", "## Warnings", ""])
            for warning in manifest["warnings"]:
                lines.append(f"- `{warning['stage']}`: {warning['message']}")
        if manifest.get("errors"):
            lines.extend(["", "## Errors", ""])
            for error in manifest["errors"]:
                lines.append(f"- `{error['stage']}`: {error['message']}")
        lines.extend(
            [
                "",
                "## Next Steps",
                "",
                "- Review partial or skipped stages before comparing experiments.",
                "- Treat candidate regions and candidate orbital locations as exploratory prioritization, not confirmations.",
                "",
                "## Scientific Interpretation Constraints",
                "",
                "This run supports topological prioritization and probabilistic characterization of observational incompleteness. Results require validation and sensitivity analysis before scientific claims are strengthened.",
            ]
        )
        summary = ctx.report_dir("run") / "run_summary.md"
        summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        outputs = [ctx.log_artifact(self.name, summary, "report", "Unified run summary")]
        return StageResult(self.name, "success", outputs, {"artifact_count": len(outputs)}, elapsed_seconds=time.perf_counter() - start)
