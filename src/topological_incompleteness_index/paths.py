from __future__ import annotations
from pathlib import Path
from typing import Iterable

def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

def resolve_output_dirs(root: Path, outputs_cfg: dict) -> dict[str, Path]:
    out = {}
    for key, value in outputs_cfg.items():
        if value is None:
            continue
        p = Path(value)
        if not p.is_absolute():
            p = root / p
        out[key] = ensure_dir(p)
    return out

def first_existing(root: Path, explicit: str | None, patterns: Iterable[str], label: str) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = root / p
        if p.exists():
            return p
        raise FileNotFoundError(f"Configured {label} does not exist: {p}")

    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(sorted(root.glob(pattern)))
    matches = [m for m in matches if m.is_file()]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {label}. Set it explicitly in configs/topological_incompleteness_index.yaml "
            f"or run the upstream pipeline first. Patterns checked: {list(patterns)}"
        )
    return matches[0]
