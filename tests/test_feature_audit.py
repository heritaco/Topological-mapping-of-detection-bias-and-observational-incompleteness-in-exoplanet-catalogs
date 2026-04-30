from pathlib import Path

import pandas as pd

from src.exoplanet_tda.features.audit import build_feature_audit_tables, run_feature_audit
from src.exoplanet_tda.features.registry import load_feature_registry
from src.exoplanet_tda.pipeline.orchestrator import create_context


def test_missing_columns_are_reported_not_crashing():
    registry = load_feature_registry("configs/features/feature_registry.yaml", "configs/features/feature_sets.yaml")
    frame = pd.DataFrame({"pl_orbper": [1.0, None], "st_mass": [1.0, 0.9]})
    availability, missingness, summary = build_feature_audit_tables([("catalog", frame)], registry)
    row = availability.loc[availability["feature_name"].eq("pl_orbeccen")].iloc[0]
    assert row["available"] is False or row["available"] == False
    assert "missing from current inputs" in row["recommended_action"]
    assert not summary.empty


def test_feature_audit_writes_expected_csvs(tmp_path):
    catalog = tmp_path / "catalog.csv"
    candidates = tmp_path / "candidates.csv"
    pd.DataFrame(
        {
            "pl_orbper": [10.0, 20.0],
            "pl_orbsmax": [0.1, 0.2],
            "st_mass": [1.0, 0.9],
            "st_rad": [1.0, 0.8],
            "st_teff": [5800, 5100],
            "st_lum": [1.0, 0.5],
        }
    ).to_csv(catalog, index=False)
    pd.DataFrame({"hostname": ["H1"], "pl_orbper": [15.0], "pl_orbsmax": [0.15]}).to_csv(candidates, index=False)

    ctx = create_context(
        "configs/pipeline/default.yaml",
        run_id="pytest_feature_audit",
        dry_run=False,
        overwrite=True,
        extra_overrides={
            "inputs": {
                "raw_catalog": catalog.as_posix(),
                "imputed_catalog": catalog.as_posix(),
                "system_candidates_csv": candidates.as_posix(),
            }
        },
    )
    paths, metrics, warnings = run_feature_audit(ctx)
    assert paths["availability"].exists()
    assert paths["missingness"].exists()
    assert paths["summary"].exists()
    assert paths["report"].exists()
    assert metrics["features_registered"] > 0
