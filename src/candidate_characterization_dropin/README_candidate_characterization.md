# Candidate Characterization Module

This drop-in module adds probabilistic characterization of topologically prioritized candidate missing planets to the existing exoplanet Mapper/TDA repository.

It is designed to be copied into the repository root so that the package lives at:

```text
src/candidate_characterization/
configs/candidate_characterization/default.yaml
scripts/run_candidate_characterization.sh
scripts/run_candidate_characterization.ps1
```

## What it does

Given:

1. an observed or imputed PSCompPars-like catalog;
2. a candidate table produced by `system_missing_planets` or a custom CSV with `hostname` and candidate period/semimajor axis;
3. optional TOI/ATI tables;

it produces:

```text
outputs/candidate_characterization/tables/candidate_property_predictions.csv
outputs/candidate_characterization/tables/candidate_analog_neighbors.csv
outputs/candidate_characterization/tables/validation_metrics.csv
outputs/candidate_characterization/tables/validation_predictions.csv
outputs/candidate_characterization/models/candidate_characterization_models.joblib
outputs/candidate_characterization/models/model_metadata.json
reports/candidate_characterization/candidate_characterization_summary.md
```

The predictions include probabilistic ranges for radius, mass, density, planet-size class, thermal/orbital class, transit probability proxy, RV signal proxy, and analog support.

## Install optional GPU dependencies

The module runs on CPU with `pandas`, `numpy`, `scikit-learn`, `joblib`, and `pyyaml`.

For GPU-accelerated tree models, install XGBoost with CUDA support in your environment. The code automatically attempts:

```python
tree_method="hist", device="cuda"
```

and falls back to CPU if the local build does not support CUDA.

Recommended extras:

```bash
pip install pandas numpy scikit-learn joblib pyyaml xgboost torch
```

`torch` is only used for CUDA detection; the main model path uses XGBoost when available.

## Input candidate CSV

Minimum useful columns:

```csv
candidate_id,hostname,pl_orbper,pl_orbsmax,node_id,toi,ati,shadow_score
```

Accepted aliases include:

- period: `p_star`, `p_candidate`, `p_candidate_days`, `period_candidate`, `period_days`, `pl_orbper`
- semimajor axis: `a_star`, `a_candidate`, `a_candidate_au`, `a_au`, `pl_orbsmax`
- host: `hostname`, `host`, `host_name`, `star_name`
- node: `node_id`, `node`, `mapper_node`, `region_id`

If `pl_orbsmax` is missing, the code derives it from period and stellar mass when available:

```math
a = \left[M_\star (P/365.25)^2\right]^{1/3}.
```

If `pl_insol` or `pl_eqt` are missing, the code attempts physical derivation from stellar properties.

## Default path discovery

Catalog paths searched:

```text
reports/imputation/PSCompPars_imputed_iterative.csv
reports/imputation/outputs/PSCompPars_imputed_iterative.csv
outputs/imputation/PSCompPars_imputed_iterative.csv
data/PSCompPars_imputed_iterative.csv
data/PSCompPars_2026.04.25_17.36.36.csv
```

Candidate paths searched:

```text
outputs/system_missing_planets/candidate_characterization_input.csv
outputs/system_missing_planets/system_candidates.csv
outputs/system_missing_planets/candidates.csv
outputs/system_missing_planets/synthetic_candidates.csv
reports/system_missing_planets/outputs/tables/system_candidates.csv
reports/system_missing_planets/outputs/tables/candidates.csv
reports/system_missing_planets/outputs/tables/synthetic_candidates.csv
```

## Run end-to-end

From repository root:

```bash
python -m src.candidate_characterization.run_characterization \
  --config configs/candidate_characterization/default.yaml \
  --repo-root . \
  --train \
  --predict \
  --validate \
  --validation-mode multiplanet
```

CPU-only:

```bash
python -m src.candidate_characterization.run_characterization \
  --repo-root . \
  --train --predict --validate --cpu
```

With explicit inputs:

```bash
python -m src.candidate_characterization.run_characterization \
  --repo-root . \
  --catalog-csv reports/imputation/PSCompPars_imputed_iterative.csv \
  --candidates-csv outputs/system_missing_planets/candidate_characterization_input.csv \
  --train --predict --validate
```

## Important interpretation constraint

The output is a conditional probabilistic characterization, not a planet confirmation. A row means:

> If a planet exists at this topologically prioritized candidate location, the observed catalog and model support this range of properties.

It does not mean the object exists, nor does it estimate an absolute number of missing planets.
