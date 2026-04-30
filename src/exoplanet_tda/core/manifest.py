"""Artifact and run manifest registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import ensure_dir, read_json, safe_relative, write_json


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ArtifactEntry:
    stage: str
    path: str
    relative_path: str
    kind: str
    description: str
    created_at: str
    exists: bool
    size_bytes: int | None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_path(
        cls,
        *,
        stage: str,
        path: Path,
        repo_root: Path,
        kind: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactEntry":
        exists = path.exists()
        return cls(
            stage=stage,
            path=path.as_posix(),
            relative_path=safe_relative(path, repo_root),
            kind=kind,
            description=description,
            created_at=utc_now(),
            exists=exists,
            size_bytes=path.stat().st_size if exists and path.is_file() else None,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "path": self.path,
            "relative_path": self.relative_path,
            "kind": self.kind,
            "description": self.description,
            "created_at": self.created_at,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "metadata": self.metadata,
        }


class ArtifactRegistry:
    """Collects artifacts and writes run_manifest.json."""

    def __init__(self, repo_root: Path, run_dir: Path, run_id: str) -> None:
        self.repo_root = repo_root
        self.run_dir = run_dir
        self.run_id = run_id
        self.created_at = utc_now()
        self.artifacts: list[ArtifactEntry] = []
        self.stages: dict[str, dict[str, Any]] = {}
        self.warnings: list[dict[str, str]] = []
        self.errors: list[dict[str, str]] = []

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifests" / "run_manifest.json"

    def add_artifact(
        self,
        stage: str,
        path: Path,
        kind: str,
        description: str,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactEntry:
        entry = ArtifactEntry.from_path(
            stage=stage,
            path=path,
            repo_root=self.repo_root,
            kind=kind,
            description=description,
            metadata=metadata,
        )
        self.artifacts.append(entry)
        return entry

    def add_warning(self, stage: str, message: str) -> None:
        self.warnings.append({"stage": stage, "message": message, "created_at": utc_now()})

    def add_error(self, stage: str, message: str) -> None:
        self.errors.append({"stage": stage, "message": message, "created_at": utc_now()})

    def record_stage(self, result: Any) -> None:
        if hasattr(result, "to_dict"):
            self.stages[result.name] = result.to_dict()
        else:
            self.stages[str(result.get("name"))] = dict(result)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_at": self.created_at,
            "updated_at": utc_now(),
            "run_dir": self.run_dir.as_posix(),
            "stages": self.stages,
            "artifacts": [entry.to_dict() for entry in self.artifacts],
            "warnings": self.warnings,
            "errors": self.errors,
        }

    def save(self) -> Path:
        ensure_dir(self.manifest_path.parent)
        return write_json(self.manifest_path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> dict[str, Any]:
        return read_json(path)
