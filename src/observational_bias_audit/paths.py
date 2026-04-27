from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
LATEX_DIR = PROJECT_ROOT / "latex"
CONFIGS_DIR = PROJECT_ROOT / "configs"


def resolve_repo_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_audit_output_tree(base_dir: Path) -> dict[str, Path]:
    tree = {
        "root": base_dir,
        "figures": base_dir / "figures",
        "tables": base_dir / "tables",
        "logs": base_dir / "logs",
        "metadata": base_dir / "metadata",
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def ensure_latex_dir(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    return base_dir


def relative_posix_path(target: Path, start: Path) -> str:
    return Path(target).resolve().relative_to(PROJECT_ROOT.resolve()).as_posix() if start == PROJECT_ROOT else Path(
        Path(__import__("os").path.relpath(target.resolve(), start.resolve()))
    ).as_posix()
