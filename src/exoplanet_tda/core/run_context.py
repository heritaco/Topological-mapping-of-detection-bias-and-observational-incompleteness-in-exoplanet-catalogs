"""RunContext object shared by pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .constants import STAGE_DIR_NAMES
from .io import ensure_dir, safe_relative, write_json, write_yaml
from .manifest import ArtifactRegistry


@dataclass
class RunContext:
    repo_root: Path
    run_id: str
    run_dir: Path
    config: dict[str, Any]
    logger: Any
    registry: ArtifactRegistry
    random_state: int = 42
    use_gpu: str = "auto"
    fail_fast: bool = False
    dry_run: bool = False

    def path(self, *parts: str | Path) -> Path:
        return self.run_dir.joinpath(*parts)

    def ensure_dir(self, path: Path) -> Path:
        return ensure_dir(path)

    def stage_dir(self, stage_name: str, kind: str) -> Path:
        if kind == "reports":
            return self.report_dir(stage_name)
        if kind == "models":
            return self.model_dir(stage_name)
        directory = self.run_dir / kind / STAGE_DIR_NAMES.get(stage_name, stage_name)
        return ensure_dir(directory)

    def table_dir(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name, "tables")

    def figure_dir(self, stage_name: str) -> Path:
        return self.stage_dir(stage_name, "figures")

    def report_dir(self, stage_name: str) -> Path:
        if stage_name == "run":
            return ensure_dir(self.run_dir / "reports")
        return ensure_dir(self.run_dir / "reports" / "stage_summaries")

    def model_dir(self, stage_name: str) -> Path:
        return ensure_dir(self.run_dir / "models" / STAGE_DIR_NAMES.get(stage_name, stage_name))

    def log_artifact(
        self,
        stage: str,
        path: Path,
        kind: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        entry = self.registry.add_artifact(stage, path, kind, description, metadata)
        self.registry.save()
        return entry.to_dict()

    def warn(self, stage: str, message: str) -> None:
        self.logger.warning("[%s] %s", stage, message)
        self.registry.add_warning(stage, message)
        self.registry.save()

    def write_json(self, relative_path: str | Path, obj: Any) -> Path:
        path = self.run_dir / relative_path
        write_json(path, obj)
        return path

    def write_yaml(self, relative_path: str | Path, obj: Any) -> Path:
        path = self.run_dir / relative_path
        write_yaml(path, obj)
        return path

    def write_csv(self, relative_path: str | Path, dataframe: Any) -> Path:
        path = self.run_dir / relative_path
        ensure_dir(path.parent)
        dataframe.to_csv(path, index=False)
        return path

    def finish_stage(self, stage: str, result: Any) -> None:
        self.registry.record_stage(result)
        manifest_path = self.run_dir / "manifests" / f"stage_manifest_{stage}.json"
        write_json(manifest_path, result.to_dict() if hasattr(result, "to_dict") else result)
        self.registry.add_artifact(stage, manifest_path, "manifest", f"Stage manifest for {stage}")
        if getattr(result, "error", None):
            self.registry.add_error(stage, str(result.error))
        for warning in getattr(result, "warnings", []) or []:
            self.registry.add_warning(stage, str(warning))
        self.registry.save()

    def repo_relative(self, path: Path) -> str:
        return safe_relative(path, self.repo_root)
