from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from topological_incompleteness_index.anchor_index import anchor_representativeness, compute_ati
from topological_incompleteness_index.neighbor_deficit import delta_rel, delta_rel_best
from topological_incompleteness_index.r3_geometry import R3Columns, build_r3_frame, safe_log10_series
from topological_incompleteness_index.regional_index import compute_toi_scores
from topological_incompleteness_index.reporting import build_interpretation_summary


class TopologicalIncompletenessIndexTests(unittest.TestCase):
    def test_delta_rel_formula(self) -> None:
        self.assertAlmostEqual(float(delta_rel(10.0, 4.0, 1.0e-9)), 0.6)

    def test_delta_rel_best_nonnegative(self) -> None:
        self.assertEqual(delta_rel_best([-0.5, -0.2, None]), 0.0)
        self.assertAlmostEqual(delta_rel_best([-0.5, 0.2, 0.1]), 0.2)

    def test_toi_penalizes_imputation(self) -> None:
        nodes = pd.DataFrame(
            {
                "node_id": ["a", "b"],
                "n_members": [10, 10],
                "degree": [3, 3],
                "shadow_score": [0.4, 0.4],
                "I_R3": [0.0, 0.5],
                "physical_distance_v_to_N1": [0.2, 0.2],
                "S_net": [0.8, 0.8],
            }
        )
        out = compute_toi_scores(nodes, sigma=1.0, epsilon=1.0e-9, min_node_members=3, high_priority_quantile=0.9)
        scores = dict(zip(out["node_id"], out["TOI"]))
        self.assertGreater(scores["a"], scores["b"])

    def test_toi_prefers_physical_continuity(self) -> None:
        nodes = pd.DataFrame(
            {
                "node_id": ["a", "b"],
                "n_members": [10, 10],
                "degree": [3, 3],
                "shadow_score": [0.4, 0.4],
                "I_R3": [0.0, 0.0],
                "physical_distance_v_to_N1": [0.1, 2.0],
                "S_net": [0.8, 0.8],
            }
        )
        out = compute_toi_scores(nodes, sigma=1.0, epsilon=1.0e-9, min_node_members=3, high_priority_quantile=0.9)
        a = out[out["node_id"] == "a"].iloc[0]
        b = out[out["node_id"] == "b"].iloc[0]
        self.assertGreater(float(a["C_phys"]), float(b["C_phys"]))
        self.assertGreater(float(a["TOI"]), float(b["TOI"]))

    def test_ati_zero_when_no_deficit(self) -> None:
        self.assertEqual(compute_ati(0.5, -0.2, 0.0, 0.8), 0.0)

    def test_anchor_representativeness(self) -> None:
        df = pd.DataFrame(
            {
                "r3_z_mass": [0.0, 1.0],
                "r3_z_period": [0.0, 1.0],
                "r3_z_semimajor": [0.0, 1.0],
            }
        )
        near = pd.Series({"r3_z_mass": 0.0, "r3_z_period": 0.0, "r3_z_semimajor": 0.0})
        far = pd.Series({"r3_z_mass": 2.0, "r3_z_period": 2.0, "r3_z_semimajor": 2.0})
        near_score, _ = anchor_representativeness(near, df, ["r3_z_mass", "r3_z_period", "r3_z_semimajor"], 1.0e-9)
        far_score, _ = anchor_representativeness(far, df, ["r3_z_mass", "r3_z_period", "r3_z_semimajor"], 1.0e-9)
        self.assertGreater(near_score, far_score)

    def test_no_absolute_claims(self) -> None:
        regions = pd.DataFrame({"node_id": ["n1"], "TOI": [0.4], "shadow_score": [0.3], "region_class": ["high_toi_region"], "top_method": ["Radial Velocity"]})
        anchors = pd.DataFrame({"anchor_pl_name": ["p1"], "node_id": ["n1"], "ATI": [0.2], "delta_rel_neighbors_best": [0.1], "deficit_class": ["weak_deficit"], "expected_incompleteness_direction": ["menor proxy RV"]})
        summary = pd.DataFrame({"x": [1]})
        deficits = pd.DataFrame()
        text = build_interpretation_summary(regions, anchors, summary, deficits).lower()
        for phrase in ["planetas faltantes confirmados", "descubrimos planetas", "faltan exactamente", "predicción definitiva"]:
            self.assertNotIn(phrase, text)

    def test_invalid_logs_are_marked(self) -> None:
        values, invalid = safe_log10_series(pd.Series([10.0, 0.0, -1.0]))
        self.assertTrue(pd.isna(values.iloc[1]))
        self.assertTrue(bool(invalid.iloc[1]))
        df = pd.DataFrame({"pl_bmasse": [1.0, -1.0], "pl_orbper": [2.0, 2.0], "pl_orbsmax": [3.0, 0.0]})
        out, _ = build_r3_frame(df, R3Columns("pl_bmasse", "pl_orbper", "pl_orbsmax"), warnings=[], skipped_items=[])
        self.assertTrue(bool(out.loc[0, "r3_valid"]))
        self.assertFalse(bool(out.loc[1, "r3_valid"]))


if __name__ == "__main__":
    unittest.main()
