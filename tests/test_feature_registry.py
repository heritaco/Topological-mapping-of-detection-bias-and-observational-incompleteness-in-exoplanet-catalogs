import pandas as pd

from src.exoplanet_tda.features.registry import load_feature_registry
from src.candidate_characterization.run_characterization import parse_args


def test_feature_registry_loads_and_resolves_minimal_set():
    registry = load_feature_registry("configs/features/feature_registry.yaml", "configs/features/feature_sets.yaml")
    resolved = registry.resolve("candidate_characterization_minimal")
    assert resolved.leakage_safe is True
    assert "candidate_orbital" in resolved.groups
    assert "observational_audit" not in resolved.groups
    assert "candidate_pl_orbper" in resolved.features
    assert "st_mass" in resolved.features


def test_registry_expression_resolution():
    registry = load_feature_registry("configs/features/feature_registry.yaml", "configs/features/feature_sets.yaml")
    resolved = registry.resolve("candidate_orbital + stellar_base")
    assert "candidate_orbital" in resolved.groups
    assert "stellar_base" in resolved.groups
    assert "candidate_pl_orbsmax" in resolved.features
    assert "st_teff" in resolved.features


def test_candidate_characterization_accepts_feature_set_argument():
    args = parse_args(["--feature-set", "candidate_characterization_minimal"])
    assert args.feature_set == "candidate_characterization_minimal"
