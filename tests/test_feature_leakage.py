import pandas as pd

from src.exoplanet_tda.features.derived import add_derived_features
from src.exoplanet_tda.features.leakage import apply_leakage_rules
from src.exoplanet_tda.features.registry import load_feature_registry


def test_leakage_rules_remove_target_columns():
    registry = load_feature_registry("configs/features/feature_registry.yaml", "configs/features/feature_sets.yaml")
    report = apply_leakage_rules(
        ["pl_orbper", "pl_bmasse", "log_pl_bmasse", "st_mass", "discoverymethod"],
        target="pl_bmasse",
        registry=registry,
    )
    assert "pl_bmasse" not in report.features
    assert "log_pl_bmasse" not in report.features
    assert "discoverymethod" not in report.features
    assert "st_mass" in report.features


def test_leakage_rules_remove_radius_class_definers():
    registry = load_feature_registry("configs/features/feature_registry.yaml", "configs/features/feature_sets.yaml")
    report = apply_leakage_rules(["pl_rade", "pl_bmasse", "st_teff"], target="radius_class", registry=registry)
    assert "pl_rade" not in report.features
    assert "pl_bmasse" not in report.features
    assert "st_teff" in report.features


def test_derived_features_do_not_overwrite_raw_columns():
    df = pd.DataFrame(
        {
            "pl_orbper": [10.0],
            "pl_orbsmax": [0.2],
            "st_rad": [1.0],
            "st_mass": [1.0],
            "st_lum": [1.0],
            "st_teff": [5800.0],
        }
    )
    out = add_derived_features(df)
    assert out.loc[0, "pl_orbper"] == 10.0
    assert out.loc[0, "pl_orbsmax"] == 0.2
    assert "candidate_pl_orbper" in out.columns
    assert "transit_probability_proxy" in out.columns
