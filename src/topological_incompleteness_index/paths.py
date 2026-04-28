from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CONFIGS_DIR = PROJECT_ROOT / "configs"
LATEX_DIR = PROJECT_ROOT / "latex"


def resolve_repo_path(value: str | None, default: Path) -> Path:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_output_tree(base_dir: Path, figures_dir: Path, tables_dir: Path, metadata_dir: Path, logs_dir: Path) -> dict[str, Path]:
    tree = {
        "base": ensure_dir(base_dir),
        "figures": ensure_dir(figures_dir),
        "tables": ensure_dir(tables_dir),
        "metadata": ensure_dir(metadata_dir),
        "logs": ensure_dir(logs_dir),
    }
    return tree


def ensure_latex_dir(base_dir: Path) -> Path:
    return ensure_dir(base_dir)


def repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(path)
