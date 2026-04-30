from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    runs_root = Path("outputs") / "runs"
    if not runs_root.exists():
        print("No unified runs found.")
        return 0
    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        manifest = run_dir / "manifests" / "run_manifest.json"
        marker = "manifest" if manifest.exists() else "no manifest"
        print(f"{run_dir.name}\t{marker}\t{run_dir.as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
