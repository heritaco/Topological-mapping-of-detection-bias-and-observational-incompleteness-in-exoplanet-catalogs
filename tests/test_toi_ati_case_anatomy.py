from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.toi_ati_case_anatomy.case_selection import select_final_presentation_cases
from src.toi_ati_case_anatomy.decomposition import (
    add_ati_decomposition,
    add_toi_decomposition,
    audit_deficit_formulas,
    summarize_deficit_by_radius,
)
from src.toi_ati_case_anatomy.reporting import write_markdown_summary
from src.toi_ati_case_anatomy.validation import (
    compute_delta_n,
    compute_delta_rel,
    contains_forbidden_claim,
    deficit_stability_label,
    safe_ratio,
)


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


def test_delta_n_formula():
    assert np.isclose(compute_delta_n(11, 10), 1.0)


def test_delta_rel_formula():
    assert np.isclose(compute_delta_rel(11, 10), (11 - 10) / (11 + 1e-9))


def test_delta_rel_not_confused_with_delta_n():
    raw = pd.DataFrame({
        "anchor_pl_name": ["p"],
        "node_id": ["n"],
        "radius_type": ["r_kNN"],
        "radius_value": [0.5],
        "N_obs": [2],
        "N_exp_neighbors": [5],
        "delta_N_neighbors": [3],
        "delta_rel_neighbors": [3],
    })
    audited, audit = audit_deficit_formulas(raw)
    assert audited.loc[0, "deficit_formula_check"] == "mismatch_recomputed_used"
    assert np.isclose(audited.loc[0, "Delta_rel_neighbors"], 3 / (5 + 1e-9))
    assert audit["raw_delta_rel_gt_one_count"] == 1


def test_summarize_deficit_by_radius():
    df = pd.DataFrame({
        "node_id": ["a", "a", "a"],
        "anchor_pl_name": ["p", "p", "p"],
        "radius_type": ["r_kNN", "r_node_median", "r_node_q75"],
        "Delta_rel_neighbors": [0.1, 0.3, -0.5],
    })
    out = summarize_deficit_by_radius(df)
    assert np.isclose(out.loc[0, "delta_rel_neighbors_best"], 0.3)
    assert out.loc[0, "best_radius"] == "r_node_median"


def test_deficit_stability_label():
    assert deficit_stability_label([0.2, 0.1, 0.05]) == "consistent_positive_deficit"
    assert deficit_stability_label([0.2, 0.0, -0.1]) == "radius_sensitive_deficit"
    assert deficit_stability_label([-0.1, 0.0, -0.2]) == "no_consistent_deficit"


def test_repeated_anchor_selection():
    regions = pd.DataFrame({
        "node_id": ["n1", "n2"],
        "TOI": [0.3, 0.2],
        "n_members": [5, 4],
        "I_R3": [0.1, 0.1],
        "S_net": [0.9, 0.8],
        "shadow_score": [0.4, 0.3],
        "C_phys": [0.9, 0.8],
    })
    anchors = pd.DataFrame({
        "anchor_pl_name": ["planet A", "planet A", "planet B"],
        "node_id": ["n1", "n2", "n3"],
        "ATI": [0.4, 0.3, 0.2],
        "TOI": [0.3, 0.2, 0.1],
        "delta_rel_neighbors_best": [0.2, 0.1, 0.05],
        "r3_imputation_score": [0.0, 0.0, 0.0],
        "anchor_representativeness": [0.9, 0.8, 0.7],
    })
    deficit_summary = pd.DataFrame({
        "anchor_pl_name": ["planet A", "planet A", "planet B"],
        "node_id": ["n1", "n2", "n3"],
        "deficit_stability_label": ["consistent_positive_deficit", "radius_sensitive_deficit", "no_consistent_deficit"],
    })
    out = select_final_presentation_cases(regions, anchors, deficit_summary)
    repeated = out[out["case_type"] == "repeated_anchor_multi_node"].iloc[0]
    assert repeated["anchor_pl_name"] == "planet A"
    assert repeated["n_nodes_as_anchor"] == 2


def test_final_cases_has_three_rows():
    regions = pd.DataFrame({
        "node_id": ["n1", "n2"],
        "TOI": [0.3, 0.2],
        "n_members": [5, 4],
        "I_R3": [0.1, 0.1],
        "S_net": [0.9, 0.8],
        "shadow_score": [0.4, 0.3],
        "C_phys": [0.9, 0.8],
    })
    anchors = pd.DataFrame({
        "anchor_pl_name": ["planet A", "planet A", "planet B"],
        "node_id": ["n1", "n2", "n3"],
        "ATI": [0.4, 0.3, 0.2],
        "TOI": [0.3, 0.2, 0.1],
        "delta_rel_neighbors_best": [0.2, 0.1, 0.05],
        "r3_imputation_score": [0.0, 0.0, 0.0],
        "anchor_representativeness": [0.9, 0.8, 0.7],
    })
    deficit_summary = pd.DataFrame({
        "anchor_pl_name": ["planet A", "planet A", "planet B"],
        "node_id": ["n1", "n2", "n3"],
        "deficit_stability_label": ["consistent_positive_deficit", "radius_sensitive_deficit", "no_consistent_deficit"],
    })
    out = select_final_presentation_cases(regions, anchors, deficit_summary)
    assert len(out) == 3


def test_no_forbidden_claims(tmp_path: Path):
    path = tmp_path / "summary.md"
    write_markdown_summary(
        path,
        sentences=["Priorizacion observacional prudente."],
        top_regions=pd.DataFrame({"node_id": ["n1"], "TOI": [0.3], "shadow_score": [0.4], "I_R3": [0.1], "C_phys": [0.9], "S_net": [0.8], "top_method": ["RV"]}),
        top_anchors=pd.DataFrame({"anchor_pl_name": ["planet A"], "node_id": ["n1"], "ATI": [0.4], "TOI": [0.3], "delta_rel_neighbors_best": [0.2], "deficit_class": ["weak_deficit"]}),
        top_anchor_radius_summary=pd.DataFrame({"anchor_pl_name": ["planet A"], "node_id": ["n1"], "ATI": [0.4], "best_radius_type": ["r_kNN"], "Delta_rel_neighbors_best": [0.2], "mean_Delta_rel_neighbors": [0.15], "median_Delta_rel_neighbors": [0.2], "deficit_stability_label": ["consistent_positive_deficit"], "interpretation_short": ["Deficit estable."]}),
        final_cases=pd.DataFrame({"case_type": ["top_toi_region"], "anchor_pl_name": [np.nan], "node_id": ["n1"], "TOI": [0.3], "ATI": [np.nan], "Delta_rel_neighbors_best": [np.nan], "deficit_stability_label": ["not_anchor_case"], "how_to_present": ["Presentar como region top."], "caution_text": ["Lectura prudente."]}),
        deficit_audit={"raw_delta_rel_gt_one_count": 0, "recomputed_delta_rel_gt_one_count": 0, "mismatch_count": 0},
    )
    text = path.read_text(encoding="utf-8")
    assert not contains_forbidden_claim(text)
