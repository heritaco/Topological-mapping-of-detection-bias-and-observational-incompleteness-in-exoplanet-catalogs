from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observational_shadow.node_profiles import build_node_shadow_profiles
from observational_shadow.shadow_metrics import compute_shadow_scores, method_l1_boundary, size_weight


class ObservationalShadowTests(unittest.TestCase):
    def test_equal_node_and_neighbor_method_composition_has_zero_l1(self) -> None:
        self.assertAlmostEqual(method_l1_boundary(pd.Series([0.5, 0.5]).to_numpy(), pd.Series([0.5, 0.5]).to_numpy()), 0.0)

    def test_pure_node_has_larger_shadow_component_than_mixed_node(self) -> None:
        frame = pd.DataFrame(
            {
                "n_members": [20, 20],
                "top_method_fraction": [1.0, 0.5],
                "method_entropy_norm": [0.0, 1.0],
                "mean_imputation_fraction": [0.0, 0.0],
                "method_l1_boundary": [0.8, 0.8],
                "no_neighbors": [False, False],
            }
        )
        scored = compute_shadow_scores(frame, has_imputation=True)
        self.assertGreater(scored.loc[0, "shadow_score_raw"], scored.loc[1, "shadow_score_raw"])

    def test_size_weight_grows_with_n_members(self) -> None:
        self.assertLess(size_weight(2, 100), size_weight(20, 100))

    def test_shadow_score_is_in_reasonable_unit_range(self) -> None:
        frame = pd.DataFrame(
            {
                "n_members": [10],
                "top_method_fraction": [0.8],
                "method_entropy_norm": [0.2],
                "mean_imputation_fraction": [0.1],
                "method_l1_boundary": [1.0],
                "no_neighbors": [False],
            }
        )
        score = compute_shadow_scores(frame, has_imputation=True).loc[0, "shadow_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_nodes_without_neighbors_are_marked(self) -> None:
        membership = pd.DataFrame(
            {
                "config_id": ["toy", "toy", "toy"],
                "node_id": ["n1", "n1", "n2"],
                "row_index": [0, 1, 2],
                "discoverymethod": ["Transit", "Transit", "Radial Velocity"],
                "pl_rade": [1.0, 1.2, 2.0],
                "pl_bmasse": [2.0, 2.2, 5.0],
                "pl_dens": [4.0, 4.2, 3.0],
                "pl_orbper": [3.0, 3.2, 20.0],
                "pl_orbsmax": [0.04, 0.05, 0.2],
                "pl_insol": [100.0, 110.0, 20.0],
                "pl_eqt": [900.0, 910.0, 500.0],
            }
        )
        node_table = pd.DataFrame(
            {
                "node_id": ["n1", "n2"],
                "mean_imputation_fraction": [0.0, 0.0],
                "physically_derived_fraction": [0.0, 0.0],
                "component_id": [0, 1],
            }
        )
        profiles, _ = build_node_shadow_profiles(
            config_id="toy",
            membership=membership,
            node_table=node_table,
            edge_table=pd.DataFrame(columns=["source", "target"]),
            requested_physical_variables=["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"],
            peripheral_degree_threshold=1,
            peripheral_component_max_nodes=3,
            epsilon=1e-9,
            warnings=[],
        )
        self.assertTrue(profiles["no_neighbors"].all())


if __name__ == "__main__":
    unittest.main()

