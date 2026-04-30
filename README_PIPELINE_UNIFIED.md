# Unified Exoplanet TDA Pipeline

This repository historically grew as several related subprojects: imputation, Mapper/TDA graphs, observational-bias audits, observational-shadow analyses, TOI/ATI scoring, system-level candidate regions, and probabilistic candidate characterization. The unified layer in `src/exoplanet_tda/` keeps those modules intact while adding a reproducible experiment runner around them.

New unified runs write to:

```text
outputs/runs/<run_id>/
```

Each run contains the submitted and effective configs, logs, manifests, stage summaries, mirrored tables/models where appropriate, and links to legacy artifacts. Existing outputs under `outputs/`, `reports/`, `configs/`, and `tex/` are not moved or deleted.

## Legacy vs Unified Structure

Legacy scientific code remains in modules such as `src/mapper_tda`, `src/observational_shadow`, `src/topological_incompleteness_index`, `src/system_missing_planets`, and `src/candidate_characterization`. The unified layer adds:

- `src/exoplanet_tda/core/`: config, paths, run context, logging, manifests.
- `src/exoplanet_tda/pipeline/`: stages, registry, orchestrator, CLI.
- `src/exoplanet_tda/<stage>/`: thin adapters and future extension points.
- `src/exoplanet_tda/reporting/`: run summaries and legacy indexing.

## Dry Run

```bash
python -m src.exoplanet_tda.pipeline.cli --config configs/pipeline/default.yaml --run-id dry_test --dry-run
```

Dry-run mode creates the run directory, writes config snapshots and a manifest, lists stages and expected inputs/outputs, and avoids heavy legacy execution.

## Full Pipeline

```bash
python scripts/run_pipeline.py --config configs/pipeline/default.yaml --run-id full_001
```

By default, heavy stages do not rerun when legacy outputs already exist; they index those outputs into the run manifest. Set a stage `rerun: true` in an experiment YAML when you intentionally want to execute a legacy stage.

## One Stage

```bash
python scripts/run_stage.py --stage candidate_characterization --config configs/pipeline/default.yaml --run-id char_001
```

Equivalent CLI form:

```bash
python -m src.exoplanet_tda.pipeline.cli --config configs/pipeline/default.yaml --run-id mapper_only --stages mapper
```

## Experiments

Create a YAML file under `configs/experiments/` with only the values you want to override. The config loader can merge a base config with an experiment config from Python, and the default templates show the intended shape. Keep experiment-specific choices in YAML rather than editing stage code.

## Adding a Stage

Add a `PipelineStage` subclass with:

- `name`
- `should_run(ctx)`
- `run(ctx) -> StageResult`

Register it in `src/exoplanet_tda/pipeline/stage_registry.py`. The stage should detect inputs, call existing code by import or subprocess, register outputs with `ctx.log_artifact`, and return `success`, `skipped`, `partial`, or `failed`.

## Manifests

The main manifest is:

```text
outputs/runs/<run_id>/manifests/run_manifest.json
```

Each artifact entry records stage, path, relative path, kind, description, timestamp, existence, size, and metadata. Each stage also writes `stage_manifest_<stage>.json`.

## Legacy Indexing

```bash
python scripts/index_legacy_outputs.py --run-id legacy_index
```

or:

```bash
python -m src.exoplanet_tda.pipeline.cli --index-legacy-outputs --run-id legacy_index
```

This scans `outputs/`, `reports/`, `configs/`, and `tex/`, then writes `outputs/runs/<run_id>/artifacts/legacy_links.json` without moving or deleting anything.

## GPU Settings

Unified `run.use_gpu` accepts `auto`, `true`, or `false`. For `candidate_characterization`, `false` passes `--cpu` to:

```bash
python -m src.candidate_characterization.run_characterization
```

The pipeline does not require a GPU.

## Scientific Interpretation

Outputs should be interpreted as exploratory evidence for topological prioritization, observational incompleteness, probabilistic characterization, candidate regions, candidate orbital locations, and local support. They are not confirmations. Sensitivity analysis, validation, and external observational context remain necessary before strengthening scientific claims.
