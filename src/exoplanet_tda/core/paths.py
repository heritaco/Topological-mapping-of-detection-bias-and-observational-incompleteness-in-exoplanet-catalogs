"""Path resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "src").exists() and (candidate / "configs").exists():
            return candidate
    return current


def resolve_repo_path(repo_root: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def resolve_config_paths(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Add a non-invasive resolved_paths section for common path settings."""
    resolved: dict[str, str] = {}
    for section in ("paths", "inputs"):
        for key, value in (config.get(section) or {}).items():
            if isinstance(value, str) and ("/" in value or "\\" in value or value in {".", ".."}):
                path = resolve_repo_path(repo_root, value)
                if path is not None:
                    resolved[f"{section}.{key}"] = path.as_posix()
    out = dict(config)
    out["resolved_paths"] = resolved
    return out
