"""Subprocess helpers that tee output into the run log."""

from __future__ import annotations

import subprocess
from pathlib import Path


def run_command(command: list[str], cwd: Path, log_path: Path) -> subprocess.CompletedProcess[str]:
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write("\n$ " + " ".join(command) + "\n")
        proc = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            check=False,
        )
        if proc.stdout:
            log_file.write(proc.stdout)
        if proc.stderr:
            log_file.write(proc.stderr)
        log_file.write(f"\n[exit_code] {proc.returncode}\n")
    return proc
