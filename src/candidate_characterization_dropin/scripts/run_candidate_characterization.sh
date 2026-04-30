#!/usr/bin/env bash
set -euo pipefail
# Run from repository root.
python -m src.candidate_characterization.run_characterization \
  --config configs/candidate_characterization/default.yaml \
  --repo-root . \
  --train \
  --predict \
  --validate \
  --validation-mode multiplanet
