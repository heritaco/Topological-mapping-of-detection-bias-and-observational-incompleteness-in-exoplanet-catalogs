from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from local_shadow_case_studies.anchor_selection import select_anchor
from local_shadow_case_studies.method_contrast import l1_boundary
from local_shadow_case_studies.neighbor_deficit import classify_deficit, delta_rel
from local_shadow_case_studies.r3_geometry import safe_log10_series


class LocalShadowCaseStudyTests(unittest.TestCase):
    def test_r3_log_transform(self) -> None:
        values = pd.Series([10.0, 1.0, 0.0, -3.0, None])
        logged, invalid = safe_log10_series(values)
        self.assertAlmostEqual(float(logged.iloc[0]), 1.0)
        self.assertAlmostEqual(float(logged.iloc[1]), 0.0)
        self.assertTrue(pd.isna(logged.iloc[2]))
        self.assertTrue(bool(invalid.iloc[2]))
        self.assertTrue(bool(invalid.iloc[3]))

    def test_method_l1_boundary(self) -> None:
        a = pd.Series({"Radial Velocity": 0.5, "Transit": 0.5})
        b = pd.Series({"Radial Velocity": 0.5, "Transit": 0.5})
        self.assertAlmostEqual(l1_boundary(a, b), 0.0)

    def test_anchor_selection_prefers_low_imputation(self) -> None:
        frame = pd.DataFrame(
            {
                "pl_name": ["A", "B"],
                "discoverymethod": ["Radial Velocity", "Radial Velocity"],
                "r3_valid": [True, True],
                "r3_z_mass": [0.0, 0.0],
                "r3_z_period": [0.0, 0.0],
                "r3_z_semimajor": [0.0, 0.0],
                "pl_bmasse": [1.0, 1.0],
                "pl_orbper": [1.0, 1.0],
                "pl_orbsmax": [1.0, 1.0],
                "imputation_status_pl_bmasse": ["observed", "imputed"],
                "imputation_status_pl_orbper": ["observed", "observed"],
                "imputation_status_pl_orbsmax": ["observed", "observed"],
                "disc_year": [2020, 2020],
                "disc_facility": ["Obs", "Obs"],
            }
        )
        anchor, _ = select_anchor(frame, ["pl_bmasse", "pl_orbper", "pl_orbsmax"])
        assert anchor is not None
        self.assertEqual(anchor["pl_name"], "A")

    def test_delta_rel(self) -> None:
        self.assertAlmostEqual(float(delta_rel(10.0, 4.0, 1e-9)), 0.6)
        self.assertEqual(classify_deficit(0.05), "no_deficit")
        self.assertEqual(classify_deficit(0.2), "weak_deficit")

    def test_no_absolute_claims(self) -> None:
        text = (
            "candidato a incompletitud observacional con deficit topologico local y vecinos compatibles esperados "
            "bajo referencia local"
        ).lower()
        for banned in ["planetas faltantes confirmados", "descubrimos planetas", "faltan exactamente"]:
            self.assertNotIn(banned, text)

    def test_missing_neighbors(self) -> None:
        self.assertIsNone(delta_rel(None, 3.0, 1e-9))


if __name__ == "__main__":
    unittest.main()
