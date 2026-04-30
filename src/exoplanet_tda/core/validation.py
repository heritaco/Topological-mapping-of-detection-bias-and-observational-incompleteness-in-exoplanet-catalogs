"""Validation helpers for unified runs."""

from __future__ import annotations

from pathlib import Path


def existing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def missing_paths(paths: list[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]
