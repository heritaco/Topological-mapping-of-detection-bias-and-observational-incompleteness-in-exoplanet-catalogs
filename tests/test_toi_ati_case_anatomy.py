from __future__ import annotations

import numpy as np
import pandas as pd

from src.toi_ati_case_anatomy.decomposition import (
    add_ati_decomposition,
    add_toi_decomposition,
    summarize_deficit_by_radius,
)
from src.toi_ati_case_anatomy.validation import contains_forbidden_claim, safe_ratio


def test_safe_ratio():
    assert np.isclose(safe_ratio(1, 9, epsilon=0), 1 / 9)


def test_toi_decomposition_recomputes_product():
    df = pd.DataFrame({
        "node_id": ["a"],
        "shadow_score": [0.5],
        "I_R3": [0.2],
        "C_phys": [0.8],
        "S_net": [0.5],
        "TOI": [0.5 * 0.8 * 0.8 * 0.5],
    })
    out = add_toi_decomposition(df)
    assert np.isclose(out.loc[0, "TOI_recomputed"], out.loc[0, "TOI"])


def test_toi_penalizes_imputation():
    df = pd.DataFrame({
        "node_id": ["low_imp", "high_imp"],
        "shadow_score": [0.5, 0.5],
        "I_R3": [0.0, 0.8],
        "C_phys": [1.0, 1.0],
        "S_net": [1.0, 1.0],
        "TOI": [0.5, 0.1],
    })
    out = add_toi_decomposition(df)
    assert out.loc[0, "TOI_recomputed"] > out.loc[1, "TOI_recomputed"]


def test_ati_uses_positive_deficit():
    df = pd.DataFrame({
        "node_id": ["a", "b"],
        "anchor_pl_name": ["p1", "p2"],
        "TOI": [1.0, 1.0],
        "delta_rel_neighbors_best": [-0.2, 0.2],
        "r3_imputation_score": [0.0, 0.0],
        "anchor_representativeness": [1.0, 1.0],
        "ATI": [0.0, 0.2],
    })
    out = add_ati_decomposition(df)
    assert out.loc[0, "ATI_recomputed"] == 0
    assert np.isclose(out.loc[1, "ATI_recomputed"], 0.2)


def test_summarize_deficit_by_radius():
    df = pd.DataFrame({
        "node_id": ["a", "a", "a"],
        "anchor_pl_name": ["p", "p", "p"],
        "radius_type": ["r1", "r2", "r3"],
        "delta_rel_neighbors": [0.1, 0.3, -0.5],
    })
    out = summarize_deficit_by_radius(df)
    assert np.isclose(out.loc[0, "delta_rel_neighbors_best"], 0.3)
    assert out.loc[0, "best_radius"] == "r2"


def test_forbidden_claims_detected():
    bad = "descubrimos planetas faltantes confirmados"
    assert contains_forbidden_claim(bad)
    good = "priorizamos regiones candidatas a incompletitud observacional"
    assert not contains_forbidden_claim(good)
