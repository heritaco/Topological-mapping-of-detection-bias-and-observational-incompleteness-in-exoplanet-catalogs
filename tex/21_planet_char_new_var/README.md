# Report 21: Planet Characterization and Feature Governance

This directory contains a standalone LaTeX report:

`Caracterizacion probabilistica de candidatos planetarios y nueva seleccion de variables`

The report summarizes the candidate characterization module in `src/candidate_characterization/` and the feature governance layer in `src/exoplanet_tda/features/` plus `configs/features/`.

## Source Data

The asset builder reads, when available:

- `outputs/candidate_characterization/tables/candidate_property_predictions.csv`
- `outputs/candidate_characterization/tables/candidate_analog_neighbors.csv`
- `outputs/candidate_characterization/tables/validation_metrics.csv`
- `outputs/candidate_characterization/tables/validation_predictions.csv`
- `reports/candidate_characterization/candidate_characterization_summary.md`
- `configs/features/feature_registry.yaml`
- `configs/features/feature_sets.yaml`
- newest `outputs/runs/*/tables/feature_audit/*`
- newest `outputs/runs/*/tables/feature_ablation/ablation_metrics.csv`

Missing inputs are handled gracefully with placeholder figures or table notes.

## Build Figures And Tables

Run from the repository root:

```bash
python tex/21_planet_char_new_var/scripts/build_report_assets.py
```

This generates PDF figures in `figures/` and LaTeX table snippets in `tables/`.

## Compile Later

This machine was not used to compile the report because Strawberry Perl / `latexmk` is not installed. No TeX installation, Perl installation, or `latexmk` command is required for asset generation.

When a TeX distribution is available, compile manually from this directory:

```bash
pdflatex main.tex
pdflatex main.tex
```

If bibliography support is added later, run `biber main` between LaTeX passes.

## Expected Outputs

- `main.tex`
- `figures/*.pdf`
- `tables/*.tex`
- `asset_build_summary.json`
- `scripts/build_report_assets.log`

The report uses conservative language: probabilistic characterization, candidate orbital locations, topological prioritization, observational incompleteness, leakage-safe feature sets, feature governance, ablation and sensitivity, exploratory evidence, and validation by held-out planets.
