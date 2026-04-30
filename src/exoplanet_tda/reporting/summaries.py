"""Run status and legacy indexing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.exoplanet_tda.core.io import ensure_dir, safe_relative, write_json
from src.exoplanet_tda.core.manifest import utc_now
from src.exoplanet_tda.core.run_context import RunContext
from src.exoplanet_tda.pipeline.stages import artifact_kind


def status_summary(manifest: dict[str, Any]) -> str:
    lines = [f"Run: {manifest.get('run_id', 'unknown')}", "Stages:"]
    for name, result in (manifest.get("stages") or {}).items():
        lines.append(f"- {name}: {result.get('status', 'unknown')}")
    warnings = manifest.get("warnings") or []
    errors = manifest.get("errors") or []
    if warnings:
        lines.append("Warnings:")
        lines.extend(f"- {item.get('stage')}: {item.get('message')}" for item in warnings)
    if errors:
        lines.append("Errors:")
        lines.extend(f"- {item.get('stage')}: {item.get('message')}" for item in errors)
    return "\n".join(lines)


def index_legacy_outputs(ctx: RunContext) -> Path:
    roots = [ctx.repo_root / name for name in ("outputs", "reports", "configs", "tex")]
    entries: list[dict[str, Any]] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            if ctx.run_dir in path.parents:
                continue
            entries.append(
                {
                    "path": path.as_posix(),
                    "relative_path": safe_relative(path, ctx.repo_root),
                    "kind": artifact_kind(path),
                    "exists": path.exists(),
                    "size_bytes": path.stat().st_size if path.exists() else None,
                    "indexed_at": utc_now(),
                }
            )
    out = ctx.run_dir / "artifacts" / "legacy_links.json"
    ensure_dir(out.parent)
    write_json(out, {"run_id": ctx.run_id, "artifact_count": len(entries), "artifacts": entries})
    ctx.log_artifact("legacy_index", out, "manifest", "Index of legacy outputs, reports, configs, and tex files")
    return out
