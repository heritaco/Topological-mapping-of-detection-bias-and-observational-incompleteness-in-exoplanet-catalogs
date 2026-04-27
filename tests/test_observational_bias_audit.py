from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from observational_bias_audit.metrics import benjamini_hochberg, purity_from_counts, shannon_entropy_from_counts
from observational_bias_audit.permutation import prepare_permutation_inputs, run_permutation_audit


class ObservationalBiasAuditTests(unittest.TestCase):
    def test_entropy_of_pure_node_is_zero(self) -> None:
        self.assertEqual(shannon_entropy_from_counts([3, 0, 0]), 0.0)

    def test_purity_fraction(self) -> None:
        self.assertAlmostEqual(purity_from_counts([2, 1]), 2.0 / 3.0)

    def test_permutation_preserves_global_method_distribution(self) -> None:
        membership = pd.DataFrame(
            {
                "node_id": ["n1", "n1", "n2", "n2"],
                "row_index": [0, 1, 2, 3],
                "discoverymethod": ["Transit", "RV", "Transit", "Imaging"],
            }
        )
        node_metrics = pd.DataFrame(
            {
                "node_id": ["n1", "n2"],
                "n_members": [2, 2],
                "top_method": ["Transit", "Transit"],
            }
        )
        prepared = prepare_permutation_inputs(membership, node_metrics, pd.DataFrame(columns=["source", "target"]))
        observed = np.bincount(prepared.original_method_codes_by_member, minlength=prepared.n_methods)
        permuted = np.bincount(np.random.default_rng(42).permutation(prepared.original_method_codes_by_member), minlength=prepared.n_methods)
        np.testing.assert_array_equal(observed, permuted)

    def test_empirical_p_values_and_q_values_in_range(self) -> None:
        membership = pd.DataFrame(
            {
                "node_id": ["n1", "n1", "n2", "n2", "n2"],
                "row_index": [0, 1, 2, 3, 4],
                "discoverymethod": ["Transit", "Transit", "RV", "Transit", "Imaging"],
            }
        )
        node_metrics = pd.DataFrame(
            {
                "node_id": ["n1", "n2"],
                "n_members": [2, 3],
                "top_method": ["Transit", "RV"],
            }
        )
        permutation_df, enrichment_df, _ = run_permutation_audit(
            config_id="toy",
            membership_with_metadata=membership,
            node_metrics=node_metrics,
            edge_table=pd.DataFrame({"source": ["n1"], "target": ["n2"]}),
            n_permutations=25,
            seed=42,
        )
        self.assertTrue(((permutation_df["empirical_p_value"] >= 0.0) & (permutation_df["empirical_p_value"] <= 1.0)).all())
        self.assertTrue(((enrichment_df["fdr_q_value"] >= 0.0) & (enrichment_df["fdr_q_value"] <= 1.0)).all())

    def test_bh_q_values_in_range(self) -> None:
        q_values = benjamini_hochberg(pd.Series([0.01, 0.20, 0.05, 0.50]))
        self.assertTrue(((q_values >= 0.0) & (q_values <= 1.0)).all())


if __name__ == "__main__":
    unittest.main()
