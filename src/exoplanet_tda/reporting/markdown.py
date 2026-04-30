"""Markdown helpers."""

from __future__ import annotations


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)
