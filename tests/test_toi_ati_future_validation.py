from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.toi_ati_future_validation.candidate_ranking import build_final_future_work_cases
from src.toi_ati_future_validation.deficit_stability import classify_deficit_profile
from src.toi_ati_future_validation.reporting import contains_forbidden_claim, write_latex_report
from src.toi_ati_future_validation.robust_indices import compute_robust_anchor_indices


def test_stable_deficit_classification():
    assert classify_deficit_profile(0.1, 0.2, 0.3) in {"stable_positive_deficit", "small_but_stable_deficit"}
    assert classify_deficit_profile(0.1, -0.2, -0.1) in {"radius_sensitive_deficit", "unstable_due_to_large_radius"}
    assert classify_deficit_profile(-0.1, -0.2, -0.3) == "no_deficit_or_overpopulated"


def test_radius_penalized_ati():
    anchors = pd.DataFrame({
        "anchor_pl_name": ["a"],
        "node_id": ["n1"],
        "ATI": [1.0],
        "TOI": [1.0],
        "delta_rel_neighbors_best": [0.5],
        "r3_imputation_score": [0.0],
        "anchor_representativeness": [1.0],
    })
    stability = pd.DataFrame({
        "anchor_pl_name": ["a"],
        "node_id": ["n1"],
        "delta_rel_mean": [0.05],
        "delta_rel_median": [0.05],
        "delta_rel_min": [-0.1],
        "delta_rel_max": [0.5],
        "delta_rel_std": [0.2],
        "n_positive_radii": [1],
        "all_radii_positive": [False],
        "any_large_radius_negative": [True],
        "radius_sensitivity_score": [0.6],
        "stable_deficit_score": [0.02],
        "deficit_stability_class": ["radius_sensitive_deficit"],
        "n_members": [8],
        "I_R3": [0.0],
        "S_net": [0.3],
    })
    out = compute_robust_anchor_indices(anchors, stability, pd.DataFrame())
    assert out.loc[0, "ATI_radius_penalized"] < out.loc[0, "ATI_original"]


def test_conservative_ati_penalizes_large_radius_negative():
    anchors = pd.DataFrame({
        "anchor_pl_name": ["a"],
        "node_id": ["n1"],
        "ATI": [1.0],
        "TOI": [1.0],
        "delta_rel_neighbors_best": [0.4],
        "r3_imputation_score": [0.0],
        "anchor_representativeness": [1.0],
    })
    stability = pd.DataFrame({
        "anchor_pl_name": ["a"],
        "node_id": ["n1"],
        "delta_rel_mean": [0.1],
        "delta_rel_median": [0.1],
        "delta_rel_min": [-0.2],
        "delta_rel_max": [0.4],
        "delta_rel_std": [0.25],
        "n_positive_radii": [1],
        "all_radii_positive": [False],
        "any_large_radius_negative": [True],
        "radius_sensitivity_score": [0.6],
        "stable_deficit_score": [0.03],
        "deficit_stability_class": ["unstable_due_to_large_radius"],
        "n_members": [8],
        "I_R3": [0.0],
        "S_net": [0.3],
    })
    out = compute_robust_anchor_indices(anchors, stability, pd.DataFrame())
    assert out.loc[0, "ATI_conservative"] < out.loc[0, "ATI_original"]


def test_rank_shift_computed():
    anchors = pd.DataFrame({
        "anchor_pl_name": ["a", "b"],
        "node_id": ["n1", "n2"],
        "ATI": [1.0, 0.8],
        "TOI": [1.0, 0.8],
        "delta_rel_neighbors_best": [0.4, 0.2],
        "r3_imputation_score": [0.0, 0.0],
        "anchor_representativeness": [1.0, 1.0],
    })
    stability = pd.DataFrame({
        "anchor_pl_name": ["a", "b"],
        "node_id": ["n1", "n2"],
        "delta_rel_mean": [0.1, 0.2],
        "delta_rel_median": [0.1, 0.2],
        "delta_rel_min": [-0.2, 0.1],
        "delta_rel_max": [0.4, 0.2],
        "delta_rel_std": [0.25, 0.05],
        "n_positive_radii": [1, 3],
        "all_radii_positive": [False, True],
        "any_large_radius_negative": [True, False],
        "radius_sensitivity_score": [0.6, 0.1],
        "stable_deficit_score": [0.03, 0.2],
        "deficit_stability_class": ["unstable_due_to_large_radius", "stable_positive_deficit"],
        "n_members": [8, 20],
        "I_R3": [0.0, 0.0],
        "S_net": [0.3, 0.8],
    })
    out = compute_robust_anchor_indices(anchors, stability, pd.DataFrame())
    assert "rank_ATI_original" in out.columns
    assert "rank_ATI_conservative" in out.columns
    assert "rank_shift" in out.columns


def test_final_future_work_cases():
    regions = pd.DataFrame({
        "node_id": ["r1", "r2"],
        "TOI_original": [0.5, 0.3],
        "TOI_rank_mean_sensitivity": [1.0, 2.0],
    })
    anchors = pd.DataFrame({
        "anchor_pl_name": ["a", "a", "b", "c"],
        "node_id": ["n1", "n2", "n3", "n4"],
        "ATI_original": [0.9, 0.7, 0.8, 0.6],
        "ATI_conservative": [0.3, 0.5, 0.8, 0.4],
        "TOI": [0.5, 0.4, 0.3, 0.2],
        "delta_rel_mean": [0.1, 0.2, 0.3, 0.05],
        "delta_rel_neighbors_best": [0.4, 0.3, 0.3, 0.1],
        "deficit_stability_class": ["radius_sensitive_deficit", "stable_positive_deficit", "stable_positive_deficit", "small_but_stable_deficit"],
        "stable_deficit_score": [0.03, 0.2, 0.25, 0.04],
        "rank_shift": [2, -1, 0, 1],
        "future_observation_direction": ["dir"] * 4,
    })
    out = build_final_future_work_cases(regions, anchors, candidate_n=5)
    expected = {
        "top_toi_region",
        "top_ati_original_anchor",
        "top_ati_conservative_anchor",
        "repeated_anchor_transition_case",
        "stable_deficit_anchor",
    }
    assert set(out["case_type"].tolist()) == expected


def test_no_forbidden_claims():
    text = "TOI/ATI no detecta planetas ausentes; prioriza regiones y anclas donde el catalogo parece observacionalmente incompleto."
    assert not contains_forbidden_claim(text)
    assert contains_forbidden_claim("descubrimos planetas") == ["descubrimos planetas"]


def test_latex_created_but_not_compiled(tmp_path: Path):
    tex_path = tmp_path / "toi_ati_future_validation.tex"
    write_latex_report(tex_path)
    assert tex_path.exists()
    assert not (tmp_path / "toi_ati_future_validation.pdf").exists()
