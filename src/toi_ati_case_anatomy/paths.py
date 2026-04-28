from __future__ import annotations

from pathlib import Path
from typing import Dict

from .config import ProjectConfig


def find_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path.cwd()).resolve()
    for path in [cur, *cur.parents]:
        if (path / "src").exists() and (path / "configs").exists():
            return path
    return cur


def resolve_path(root: Path, maybe_path: str | None) -> Path | None:
    if maybe_path is None or str(maybe_path).lower() == "null":
        return None
    p = Path(maybe_path)
    return p if p.is_absolute() else (root / p)


def output_paths(cfg: ProjectConfig, root: Path | None = None) -> Dict[str, Path]:
    repo_root = root or find_repo_root()
    outputs = cfg.outputs
    paths = {
        "base": resolve_path(repo_root, outputs.get("base_dir")),
        "tables": resolve_path(repo_root, outputs.get("tables_dir")),
        "figures": resolve_path(repo_root, outputs.get("figures_dir")),
        "metadata": resolve_path(repo_root, outputs.get("metadata_dir")),
        "logs": resolve_path(repo_root, outputs.get("logs_dir")),
        "latex": resolve_path(repo_root, outputs.get("latex_dir")),
    }
    for key, path in paths.items():
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)
    return {k: v for k, v in paths.items() if v is not None}
