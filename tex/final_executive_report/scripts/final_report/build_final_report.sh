#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python3 "$ROOT/scripts/final_report/make_final_report_figures.py"
cd "$ROOT/latex/final_report"
if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf final_executive_report.tex
else
  pdflatex final_executive_report.tex
  pdflatex final_executive_report.tex
fi
