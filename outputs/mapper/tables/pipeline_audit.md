 # Pipeline Audit

Audit date: 2026-04-27

Scope: repository inspection only. No full grid searches, bootstrap runs, null models, or expensive experiments were rerun. Existing scripts were not modified.

## A. Repository Structure

### Relevant scripts

- `src/impute_exodata.py`: CLI entry point for imputation. Produces full imputed PSCompPars tables, Mapper feature tables, validation tables, and imputation report assets.
- `src/imputation/pipeline.py`: main imputation implementation. Handles feature selection, physical derivations, source flags, validation by masking, and export of imputation tables/figures.
- `src/imputation/steps/physical_derivation.py`: derives `pl_dens` and `pl_orbsmax` when possible.
- `src/mapper_exodata.py`: CLI entry point for Mapper/TDA static outputs.
- `src/mapper_tda/feature_sets.py`: current authoritative Mapper feature-space definitions.
- `src/mapper_tda/pipeline.py`: Mapper config expansion, graph construction, metric collection, and pairwise graph metric distance construction.
- `src/mapper_tda/metrics.py`: graph metrics, node tables, edge tables, imputation/source traceability, and basic discovery metadata summaries.
- `src/mapper_tda/node_selection.py`: hard-coded main graph selection and node/component highlighting logic.
- `src/mapper_tda/interpretation.py`: text summary generation for current Mapper results.
- `src/mapper_tda/static_outputs.py`: CSV/JSON/PDF/LaTeX output writers.
- `src/mapper_tda/validation.py`: optional bootstrap, column-shuffle null models, and multi-imputation Mapper comparison.
- `tests/test_impute_exodata.py`: tests imputation defaults, physical derivations, output writing, and feature exclusions.
- `tests/test_mapper_tda.py`: tests Mapper defaults, metrics, interpretation tables/figures, optional bootstrap/null output generation, and no HTML-by-default behavior.

### Relevant data files

- `data/PSCompPars_2026.04.25_14.43.08.csv`: raw wide NASA table. After ignoring comment lines: 6273 rows, 320 columns. Contains `pl_dens`, `pl_orbeccen`, `discoverymethod`, `disc_year`, and `disc_facility`.
- `data/PSCompPars_2026.04.25_17.36.36.csv`: raw compact NASA table. After ignoring comment lines: 6273 rows, 84 columns. Contains `pl_orbeccen`, `discoverymethod`, `disc_year`, and `disc_facility`; does not contain `pl_dens`, which the imputation pipeline can derive.
- `reports/imputation/PSCompPars_imputed_iterative.csv`: current full imputed physical table available in this checkout.
- `reports/imputation/mapper_features_imputed_iterative.csv`: current Mapper feature table available in this checkout.
- `data/processed/PSCompPars_2026.04.25_17.36.36/imputation/*.csv`: processed Mapper feature tables also exist, but the documented resolver primarily looks in `outputs/imputation` and then `reports/imputation`.

### Relevant output folders

- `reports/imputation/`: current imputation outputs in this checkout. Contains median, KNN, iterative, complete-case Mapper feature tables, validation metrics, source/audit files, and `imputation_report.html`.
- `reports/imputation/outputs/`: imputation summary tables and PDF figures.
- `outputs/mapper/`: current static Mapper output root.
- `outputs/mapper/graphs/`: 21 graph JSON files.
- `outputs/mapper/nodes/`: 21 node CSV files.
- `outputs/mapper/edges/`: 21 edge CSV files.
- `outputs/mapper/config/`: 21 config JSON files.
- `outputs/mapper/metrics/`: aggregate Mapper metrics CSV.
- `outputs/mapper/tables/`: interpretive tables and summary tables.
- `outputs/mapper/figures_pdf/` and `outputs/mapper/figures_png/`: static figures.
- `outputs/mapper/figures_pdf/interpretation/`: interpretation figures, including placeholder diagnostics for bootstrap/null models when not run.
- `outputs/mapper/figures_pdf/presentation/`: presentation-style figures.
- `reports/mapper/`: legacy interactive Mapper outputs. These are useful for comparison but are not the current primary output path.

### Relevant LaTeX/report files

- `latex/03_mapper/mapper_report.tex`: generated Mapper report source.
- `latex/03_mapper/mapper_report.pdf`: compiled Mapper report.
- `latex/03_mapper/sections/*.tex`: generated report sections.
- `latex/03_mapper/tables/*.tex`: generated table extracts.
- `latex/03_mapper/figures/*.pdf`: copied report figures.
- `latex/02_imputation/imputation_pipeline_explanation.pdf`: imputation explanation/report artifact.
- `latex/01_proyecto_mapper_exoplanetas/atlas_topologico_exoplanetas.pdf`: older/proposal-style methodological document.

## B. Current Pipeline Status

| Step | Status | Evidence | Gaps / uncertainty |
| --- | --- | --- | --- |
| 0. data cleaning / imputation | DONE | `src/impute_exodata.py`, `src/imputation/pipeline.py`, and `reports/imputation/` contain full outputs for median, KNN, and iterative. `iterative` is the documented default and current main method. | Main README says the preferred output root is `outputs/imputation`, but that folder is absent in this checkout. Current data are under `reports/imputation`. |
| 1. Mapper grid search | PARTIAL | `outputs/mapper` contains 21 graph/node/edge/config files: 7 spaces x 3 lenses at `n_cubes=10`, `overlap=0.35`. `src/mapper_tda/pipeline.py` has `N_CUBES_GRID = [6, 8, 10, 12, 15]` and `OVERLAP_GRID = [0.20, 0.30, 0.35, 0.40, 0.50]`. | No current output evidence for a full `n_cubes`/`overlap` grid. Aggregate metrics tables currently list only 7 `pca2` rows, despite 21 graph artifacts. Some artifacts may be stale from an earlier all-lens run. |
| 2. graph metric extraction | PARTIAL | `outputs/mapper/metrics/mapper_graph_metrics.csv` contains `n_nodes`, `n_edges`, `beta_0`, `beta_1`, density, degree, clustering, component size, node size distribution, and imputation/derived fractions. Graph JSON files also embed `graph_metrics`. | Current aggregate metrics CSV covers only `pca2` for 7 spaces. There is no explicit graph coverage percentage column, though `rows_used=6273` is present. Metrics for non-`pca2` graphs are not in the current aggregate CSV. |
| 3. graph candidate selection | PARTIAL | `outputs/mapper/tables/main_graph_selection.csv` exists and selects six `pca2` graphs with metrics and caution levels. | Selection is hard-coded in `src/mapper_tda/node_selection.py`, then annotated with metrics. It is reproducible but not a fully metric-ranked selection across the whole grid/lens set. |
| 4. physical Mapper interpretation | PARTIAL | `phys_min` and `phys_density` graphs/tables exist. `node_physical_interpretation.csv`, `highlighted_nodes.csv`, and `component_summary.csv` include physical labels and imputation/source cautions. | Interpretation is heuristic. There is no final physical/observational/mixed/weak region classification. |
| 5. orbital Mapper interpretation | PARTIAL | `orbital_pca2_cubes10_overlap0p35` is selected as a primary low-imputation graph. It has node tables, figures, and report text. | Current `orbital` space uses only `pl_orbper` and `pl_orbsmax`; it does not include `pl_orbeccen`. No discovery-method enrichment/null audit is attached. |
| 6. joint Mapper interpretation | PARTIAL | `joint_no_density` and `joint` are selected/compared. Density sensitivity table exists. Joint interpretation figures and tables exist. | Region-level classification into physical/observational/mixed/weak is missing. |
| 7. comparison between physical/orbital/joint Mappers | PARTIAL | `mapper_space_comparison.csv`, `mapper_density_feature_sensitivity.csv`, and `mapper_graph_distances_metric_l2.csv` exist. Report summarizes orbital vs thermal vs density effect. | Current aggregate comparison is `pca2`-only. Lens artifacts exist but the current aggregate lens table does not include `domain` or `density`. |
| 8. observational bias audit using `discoverymethod` | PARTIAL | `node_physical_interpretation.csv` includes `discoverymethod_top` and `discoverymethod_entropy`; legacy `reports/mapper/*nodes*.csv` includes `discoverymethod_dominant` and percentage. | No dedicated bias audit table. No global expected distribution, enrichment score, p-value, or `disc_facility` node summary in current tables. |
| 9. permutation null test for `discoverymethod` enrichment | MISSING | `src/mapper_tda/validation.py` has column-shuffle null models for topology metrics like `beta_1`. | No permutation test for node/component discovery-method enrichment was found. Existing null models are topology nulls, not discovery-label enrichment nulls, and they were not run in this execution. |
| 10. robustness of bias metrics across grid search | MISSING | No bias metric table was found. | Requires both bias metrics and a real parameter-grid output set. |
| 11. imputation confidence audit by node | DONE | `node_physical_interpretation.csv`, `mapper_node_source_audit.csv`, `main_graph_selection.csv`, and figures such as `05_mapper_imputation_audit_by_config.pdf` exist. | This is one of the strongest implemented pieces. |
| 12. final synthesis table classifying regions as physical / observational / mixed / weak | MISSING | No table with these four final region classes was found. | Existing labels are physical heuristics and caution levels, not the intended final evidence classification. |

## C. Existing Feature Spaces

Authoritative source: `src/mapper_tda/feature_sets.py`.

| Feature-space name | Variables used | Maps to intended methodology? | Notes / mismatch |
| --- | --- | --- | --- |
| `phys_min` | `pl_rade`, `pl_bmasse` | Yes, maps to "physical" without density. | This is the clean mass-radius control. |
| `phys_density` | `pl_rade`, `pl_bmasse`, `pl_dens` | Yes, maps to "physical+density". | `pl_dens` is often physically derived from mass and radius, so it is not independent evidence. |
| `orbital` | `pl_orbper`, `pl_orbsmax` | Yes, but narrow. | Important mismatch: current orbital Mapper does not include eccentricity. It is only period + semimajor axis. |
| `thermal` | `pl_insol`, `pl_eqt` | Yes, maps to "thermal". | Current results show high imputation dependence for this space. |
| `orb_thermal` | `pl_orbper`, `pl_orbsmax`, `pl_insol`, `pl_eqt` | Yes, maps to "orbital+thermal". | In older `src/feature_config.py`, `MAPPER_ORB_FEATURES` includes the thermal variables too. The current Mapper code splits `orbital` and `orb_thermal` more cleanly. |
| `joint_no_density` | `pl_rade`, `pl_bmasse`, `pl_orbper`, `pl_orbsmax`, `pl_insol`, `pl_eqt` | Yes, maps to "joint without density". | Excludes `pl_dens` to avoid algebraic redundancy. |
| `joint` | `pl_rade`, `pl_bmasse`, `pl_dens`, `pl_orbper`, `pl_orbsmax`, `pl_insol`, `pl_eqt` | Yes, maps to "joint with density". | Used to evaluate density sensitivity. |

Aliases:

- `phys -> phys_density`
- `orb -> orb_thermal`
- `all -> all current spaces`

Eccentricity note:

- Full imputed physical data retains `pl_orbeccen`.
- Current `reports/imputation/mapper_features_imputed_iterative.csv` does not contain `pl_orbeccen`.
- Current `src/mapper_tda/feature_sets.py` does not include `pl_orbeccen` in any Mapper feature space.
- `src/impute_exodata.py` has an `--include-orbital-eccentricity` option, and `src/feature_config.py` has `MAPPER_ORB_OPTIONAL_FEATURES = ["pl_orbeccen"]`, but that optional feature is not reflected in the current Mapper feature-space definitions or current Mapper outputs.

## D. Existing Mapper Outputs

### Available graph outputs

`outputs/mapper/` currently contains:

- 21 graph JSON files in `outputs/mapper/graphs/`.
- 21 node CSV files in `outputs/mapper/nodes/`.
- 21 edge CSV files in `outputs/mapper/edges/`.
- 21 config JSON files in `outputs/mapper/config/`.

These correspond to:

- 7 spaces: `phys_min`, `phys_density`, `orbital`, `thermal`, `orb_thermal`, `joint_no_density`, `joint`.
- 3 lenses: `pca2`, `domain`, `density`.
- 1 cover setting in current filenames: `cubes10_overlap0p35`.

Important consistency warning:

- Graph/node/edge/config artifacts exist for all 21 space/lens combinations.
- `outputs/mapper/metrics/mapper_graph_metrics.csv` currently contains only 7 rows, all `lens=pca2`.
- `outputs/mapper/tables/mapper_space_comparison.csv` also contains 7 `pca2` rows.
- `outputs/mapper/tables/mapper_lens_sensitivity.csv` currently also contains only 7 `pca2` rows, despite its name.
- Therefore, current aggregate tables should be treated as the active `pca2` summary, while non-`pca2` graph files may be useful but not fully synchronized with the active summary tables.

### Available tables

Current files in `outputs/mapper/tables/`:

- `component_summary.csv`
- `highlighted_nodes.csv`
- `imputation_method_mapper_comparison.csv`
- `main_graph_selection.csv`
- `mapper_density_feature_sensitivity.csv`
- `mapper_edges_all.csv`
- `mapper_input_alignment_summary.csv`
- `mapper_input_availability.csv`
- `mapper_interpretive_summary.md`
- `mapper_interpretive_summary.tex`
- `mapper_lens_sensitivity.csv`
- `mapper_node_source_audit.csv`
- `mapper_space_comparison.csv`
- `node_physical_interpretation.csv`

### Available interpretive summaries

`outputs/mapper/tables/mapper_interpretive_summary.md` exists. Its main contents are:

- Orbital Mapper shows high structural complexity with low imputation dependence.
- Thermal Mapper shows high complexity but high imputation dependence.
- Adding derived density slightly modifies, and in this run reduces, Mapper complexity.
- The orbital PCA Mapper is the high-priority graph for scientific inspection.
- Thermal Mapper is cautionary because more than half its nodes exceed the high-imputation threshold.
- Density appears to regularize the joint space rather than add independent branching.
- Results are lens-sensitive; `pca2` remains the primary interpretation layer.
- Bootstrap and null models were not run in this execution.
- Multi-method Mapper comparison was skipped or partial because not all imputation inputs were available.

### Available selected graphs

`outputs/mapper/tables/main_graph_selection.csv` exists and contains six selected `pca2` graphs:

| config_id | priority | caution | n_nodes | n_edges | beta_1 | mean node imputation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `phys_min_pca2_cubes10_overlap0p35` | primary | low | 66 | 105 | 54 | 0.017 |
| `phys_density_pca2_cubes10_overlap0p35` | control | moderate | 67 | 102 | 52 | 0.002 |
| `orbital_pca2_cubes10_overlap0p35` | primary | low | 124 | 196 | 96 | 0.011 |
| `joint_no_density_pca2_cubes10_overlap0p35` | primary | moderate | 62 | 134 | 81 | 0.176 |
| `joint_pca2_cubes10_overlap0p35` | control | moderate | 56 | 126 | 77 | 0.152 |
| `thermal_pca2_cubes10_overlap0p35` | cautionary | high | 121 | 150 | 67 | 0.588 |

`orb_thermal_pca2_cubes10_overlap0p35` is present in metrics but is not selected in `main_graph_selection.csv`.

## E. Metadata Availability

### File-level availability

| File | Status |
| --- | --- |
| `reports/imputation/PSCompPars_imputed_iterative.csv` | Contains planet identifier, `discoverymethod`, `disc_year`, `disc_facility`, all requested physical/orbital/thermal variables including `pl_orbeccen`, source columns, `*_was_missing`, `*_was_observed`, `*_was_physically_derived`, `*_was_imputed`, and `imputation_method`. |
| `reports/imputation/mapper_features_imputed_iterative.csv` | Contains `pl_name`, `hostname`, `discoverymethod`, `disc_year`, `disc_facility`, `sy_dist`, the seven main Mapper variables, original-value columns, `*_was_missing`, and `*_source`. It does not contain `pl_orbeccen`, `*_was_imputed`, or `imputation_method`. |
| `outputs/mapper/data/planet_physical_labels.csv` | Contains identifiers, discovery metadata, all requested variables including `pl_orbeccen`, imputation flags, `imputation_method`, and added labels: `radius_class`, `orbit_class`, `thermal_class`, `candidate_population`. |
| `outputs/mapper/tables/node_physical_interpretation.csv` | Contains node-level `discoverymethod_top`, `discoverymethod_entropy`, `disc_year_median`, source fractions, imputation fractions, and physical heuristic labels. It does not include a node-level `disc_facility` summary. |

### Requested metadata checklist

| Field | Full imputed table | Mapper feature table | Mapper physical labels | Node interpretation |
| --- | --- | --- | --- | --- |
| planet identifier | YES (`pl_name`, `hostname`) | YES | YES | YES, examples/member indices |
| `discoverymethod` | YES | YES | YES | PARTIAL, summarized as top/entropy |
| `disc_year` | YES | YES | YES | PARTIAL, summarized as median/min/max |
| `disc_facility` | YES | YES | YES | NO dedicated node summary found |
| `pl_rade` | YES | YES | YES | YES, node summaries |
| `pl_bmasse` | YES | YES | YES | YES, node summaries |
| `pl_dens` | YES | YES | YES | YES, node summaries |
| `pl_orbper` | YES | YES | YES | YES, node summaries |
| `pl_orbsmax` | YES | YES | YES | YES, node summaries |
| `pl_orbeccen` | YES | NO | YES | NO current Mapper-space use |
| `pl_insol` | YES | YES | YES | YES, node summaries |
| `pl_eqt` | YES | YES | YES | YES, node summaries |
| imputation indicators / fraction | YES, flags and source columns | PARTIAL, source/missing columns but no `*_was_imputed` flags | YES | YES, node fractions |

Conclusion: metadata needed for a bias audit is retained in the full imputed/physical tables, but current Mapper outputs only partially summarize it. `discoverymethod` and `disc_year` are summarized at node level; `disc_facility` is retained upstream but not summarized in the current node interpretation table.

## F. Missing Pieces

Do not implement these yet. These are the minimal scripts/functions needed to complete the intended methodology.

### 1. Discovery-method bias audit

Proposed file: `src/mapper_tda/bias_audit.py`

Required inputs:

- Selected graph outputs or node tables from `outputs/mapper/nodes/`.
- `outputs/mapper/data/planet_physical_labels.csv` or `reports/imputation/PSCompPars_imputed_iterative.csv`.
- `outputs/mapper/tables/main_graph_selection.csv`.

Outputs:

- `outputs/mapper/tables/node_discovery_bias.csv`
- `outputs/mapper/tables/component_discovery_bias.csv`
- optional figure: `outputs/mapper/figures_pdf/interpretation/discovery_bias_by_node.pdf`

Metrics:

- node/component member count
- dominant `discoverymethod`
- dominant `discoverymethod` fraction
- discovery-method entropy
- global-vs-node enrichment ratio
- standardized residuals per method
- Jensen-Shannon or KL divergence versus global discovery-method distribution
- `disc_year` median/IQR
- dominant `disc_facility` and facility entropy

### 2. Discovery-method permutation null test

Proposed file: `src/mapper_tda/bias_nulls.py`

Required inputs:

- Node memberships for selected graphs.
- `discoverymethod` labels from physical/imputed table.
- Bias metrics from `node_discovery_bias.csv`.
- Configurable `n_perm`, e.g. 1000 for final run, smaller for smoke tests.

Outputs:

- `outputs/mapper/tables/discoverymethod_permutation_null.csv`
- `outputs/mapper/tables/discoverymethod_enrichment_summary.csv`

Metrics:

- observed enrichment statistic per node/component
- null mean/std
- z-score versus label permutation null
- empirical p-value
- optional FDR-adjusted q-value

Important distinction: existing `src/mapper_tda/validation.py` null models shuffle feature columns and test topology metrics like `beta_1`. They do not test discovery-method enrichment.

### 3. Bias robustness across Mapper grid

Proposed file: `src/mapper_tda/bias_grid_robustness.py`

Required inputs:

- A real grid of Mapper outputs across `n_cubes`, `overlap`, lenses, and selected spaces.
- Node discovery-bias tables per graph.
- Main graph selection table.

Outputs:

- `outputs/mapper/tables/bias_grid_robustness.csv`
- `outputs/mapper/tables/bias_grid_stable_regions.csv`

Metrics:

- mean/std/CV of node or component bias score across grid settings
- persistence count of enriched regions
- rank stability of high-bias nodes/components
- optional co-membership/Jaccard stability for regions

### 4. Region synthesis classification

Proposed file: `src/mapper_tda/region_synthesis.py`

Required inputs:

- `node_physical_interpretation.csv`
- `component_summary.csv`
- `node_discovery_bias.csv`
- `discoverymethod_enrichment_summary.csv`
- imputation/source audit fields
- optional bootstrap/null summaries when available

Outputs:

- `outputs/mapper/tables/final_region_synthesis.csv`
- `outputs/mapper/tables/final_region_synthesis.md`

Classification target:

- `physical`
- `observational`
- `mixed`
- `weak`

Suggested evidence fields:

- physical coherence score
- discovery-method enrichment score and p-value
- imputation confidence
- derived-feature dominance
- stability score
- final label
- human-readable rationale

### 5. Output manifest / stale-artifact check

Proposed file: `src/mapper_tda/output_manifest.py`

Required inputs:

- `outputs/mapper/graphs/`
- `outputs/mapper/nodes/`
- `outputs/mapper/edges/`
- `outputs/mapper/config/`
- aggregate metrics/tables

Outputs:

- `outputs/mapper/tables/output_manifest.csv`
- `outputs/mapper/tables/output_consistency_warnings.md`

Checks:

- every graph has a matching node, edge, and config file
- every graph has a row in aggregate metrics
- aggregate metrics include all lenses expected by the current run
- stale files from earlier runs are flagged

This is needed because the current checkout has 21 graph artifacts but only 7 active aggregate metric rows.

## G. Reproducibility Notes

### Environment

Documented setup:

```powershell
conda env create -f .\environment.yml
conda activate planetas
```

Manual setup documented in `README.md`:

```powershell
conda create -y -n planetas --override-channels -c conda-forge python=3.12 pandas numpy matplotlib scikit-learn scipy networkx pytest pip
conda activate planetas
python -m pip install "kmapper>=2.1"
```

### Imputation

Documented recommended command:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative --outputs-dir outputs/imputation
```

Observed current state:

- `outputs/imputation/` does not exist in this checkout.
- Current imputation artifacts are in `reports/imputation/` and `reports/imputation/outputs/`.
- The exact command used to produce the current `reports/imputation/` files is not logged in a machine-readable run manifest.
- Based on filenames and config, it was a compare-style run with `visualized_method=iterative`.

A command consistent with the current output location would be:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative
```

or explicitly:

```powershell
python .\src\impute_exodata.py --method compare --visualized-method iterative --reports-dir reports/imputation --outputs-dir reports/imputation/outputs
```

### Mapper generation

Documented commands:

```powershell
python .\src\mapper_exodata.py --space all --lens pca2 --input-method iterative --outputs-dir outputs/mapper --interpret-nodes --presentation --make-latex
python .\src\mapper_exodata.py --space all --lens all --input-method iterative --outputs-dir outputs/mapper --full-report
python .\src\mapper_exodata.py --space all --lens all --input-method iterative --outputs-dir outputs/mapper --full-validation --n-bootstrap 30 --n-null 30
```

Observed current state:

- `outputs/mapper/graphs`, `nodes`, `edges`, and `config` contain 21 files, consistent with `--space all --lens all` at default `n_cubes=10`, `overlap=0.35`.
- `outputs/mapper/metrics/mapper_graph_metrics.csv` and the active comparison tables contain only 7 `pca2` rows, consistent with a later `--lens pca2` aggregate run or stale all-lens artifacts left in place.
- `outputs/mapper/bootstrap/` and `outputs/mapper/null_models/` do not exist.
- `latex/03_mapper/sections/08_stability_diagnostics.tex` says bootstrap was not run.
- `latex/03_mapper/sections/09_null_model_diagnostics.tex` says null models were not run.
- Therefore, the exact command used for the current combined output state is UNCLEAR.

Most likely active summary command:

```powershell
python .\src\mapper_exodata.py --space all --lens pca2 --input-method iterative --outputs-dir outputs/mapper --interpret-nodes --presentation --make-latex
```

Earlier/stale artifacts may have come from:

```powershell
python .\src\mapper_exodata.py --space all --lens all --input-method iterative --outputs-dir outputs/mapper --full-report
```

### Report generation

The Mapper pipeline can generate LaTeX assets using `--make-latex`, `--presentation`, `--full-report`, or `--full-validation`.

Documented compile command:

```powershell
cd latex/03_mapper
latexmk -pdf -interaction=nonstopmode -halt-on-error mapper_report.tex
```

Equivalent Makefile commands:

```powershell
cd latex/03_mapper
make
```

or from `latex/`:

```powershell
cd latex
make mapper
```

### Tests

Documented command:

```powershell
python -m pytest
```

Tests were not run for this audit because the user requested inspection only and no new experiments.

