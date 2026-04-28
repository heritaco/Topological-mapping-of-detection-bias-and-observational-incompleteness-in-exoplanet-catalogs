from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml

@dataclass(frozen=True)
class ProjectConfig:
    root: Path
    raw: dict[str, Any]

    @property
    def inputs(self) -> dict[str, Any]:
        return self.raw.get("inputs", {})

    @property
    def analysis(self) -> dict[str, Any]:
        return self.raw.get("analysis", {})

    @property
    def outputs(self) -> dict[str, Any]:
        return self.raw.get("outputs", {})

    @property
    def figures(self) -> dict[str, Any]:
        return self.raw.get("figures", {})

def find_repo_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / ".git").exists() or (candidate / "README.md").exists():
            return candidate
    return current

def load_config(path: str | Path) -> ProjectConfig:
    cfg_path = Path(path)
    root = find_repo_root(Path.cwd())
    if not cfg_path.is_absolute():
        cfg_path = root / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ProjectConfig(root=root, raw=raw)
