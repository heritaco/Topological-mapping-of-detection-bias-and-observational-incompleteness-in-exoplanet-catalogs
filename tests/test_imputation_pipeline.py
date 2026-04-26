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

from imputation.pipeline import ImputationConfig, default_log_features, run_imputation_pipeline
from imputation.steps.log_transform import apply_log10_transform, invert_log10_transform
from imputation.steps.physical_derivation import EARTH_DENSITY_G_CM3, derive_planet_density


class PhysicalDerivationTests(unittest.TestCase):
    def test_derives_missing_density_without_overwriting_observed_values(self) -> None:
        df = pd.DataFrame(
            {
                "pl_rade": [1.0, 2.0, np.nan],
                "pl_bmasse": [1.0, 8.0, 3.0],
                "pl_dens": [np.nan, 2.0, np.nan],
            }
        )

        out, audit = derive_planet_density(df)

        self.assertAlmostEqual(out.loc[0, "pl_dens"], EARTH_DENSITY_G_CM3)
        self.assertEqual(out.loc[0, "pl_dens_source"], "derived_from_pl_bmasse_pl_rade")
        self.assertEqual(out.loc[1, "pl_dens"], 2.0)
        self.assertEqual(out.loc[1, "pl_dens_source"], "observed")
        self.assertTrue(pd.isna(out.loc[2, "pl_dens"]))
        self.assertEqual(audit.derived_count, 1)


class LogTransformTests(unittest.TestCase):
    def test_nonpositive_log_values_become_missing_and_invert_roundtrips_positive_values(self) -> None:
        matrix = pd.DataFrame(
            {
                "pl_orbper": [10.0, 0.0, -1.0, np.nan],
                "st_met": [-0.1, 0.0, 0.2, np.nan],
            }
        )

        transformed, audit = apply_log10_transform(matrix, ["pl_orbper"])
        restored = invert_log10_transform(transformed, ["pl_orbper"])

        self.assertAlmostEqual(transformed.loc[0, "pl_orbper"], 1.0)
        self.assertEqual(int(transformed["pl_orbper"].isna().sum()), 3)
        self.assertAlmostEqual(restored.loc[0, "pl_orbper"], 10.0)
        self.assertEqual(audit.to_frame().set_index("feature").loc["pl_orbper", "nonpositive_set_missing"], 2)
        self.assertEqual(transformed.loc[0, "st_met"], -0.1)


class PipelineTests(unittest.TestCase):
    def test_pipeline_outputs_auditable_knn_and_median_matrices(self) -> None:
        df = pd.DataFrame(
            {
                "pl_name": ["a", "b", "c", "d", "e", "f"],
                "hostname": ["ha", "hb", "hc", "hd", "he", "hf"],
                "discoverymethod": ["Transit"] * 6,
                "disc_year": [2010, 2011, 2012, 2013, 2014, 2015],
                "pl_rade": [1.0, 1.5, 2.0, np.nan, 4.0, 5.0],
                "pl_bmasse": [1.0, 5.0, np.nan, 16.0, 64.0, 80.0],
            }
        )
        features = ("pl_rade", "pl_bmasse", "pl_dens")
        config = ImputationConfig(
            features=features,
            log_features=default_log_features(features),
            methods=("knn", "median"),
            primary_method="knn",
            n_neighbors=2,
            validation_mask_fraction=0.2,
            random_state=7,
        )

        result = run_imputation_pipeline(df, config)

        self.assertEqual(set(result.imputed_by_method), {"knn", "median"})
        for method, matrix in result.imputed_by_method.items():
            self.assertFalse(matrix[list(features)].isna().any().any(), method)
            self.assertIn("any_feature_was_imputed", result.output_by_method[method].columns)
            self.assertIn("pl_dens_source", result.output_by_method[method].columns)
        self.assertIn("derived_density_count", result.missingness_audit.columns)
        self.assertFalse(result.complete_case_comparison.empty)
        self.assertFalse(result.validation_by_feature.empty)


if __name__ == "__main__":
    unittest.main()

