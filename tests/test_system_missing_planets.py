from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from system_missing_planets.config import SystemMissingPlanetsConfig
from system_missing_planets.features import build_system_metadata
from system_missing_planets.gap_model import build_gap_statistics, expand_gap_candidates, find_candidate_gaps
from system_missing_planets.io import _derive_semimajor_axis
from system_missing_planets.run_system_missing_planets import run_pipeline
from system_missing_planets.topology_prior import TopologyResources, attach_topology_to_candidates
from system_missing_planets.validation import run_leave_one_out_validation


def synthetic_catalog() -> pd.DataFrame:
    rows = [
        {
            "hostname": "SysGap",
            "pl_name": "SysGap b",
            "pl_orbper": 10.0,
            "pl_orbsmax": None,
            "pl_bmasse": 5.0,
            "pl_rade": 1.8,
            "pl_dens": 4.0,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "discoverymethod": "Transit",
            "disc_year": 2020,
            "disc_facility": "Synthetic",
            "pl_orbper_was_observed": True,
            "pl_orbsmax_was_observed": False,
            "pl_orbsmax_was_physically_derived": True,
            "pl_bmasse_was_observed": True,
            "pl_rade_was_observed": True,
        },
        {
            "hostname": "SysGap",
            "pl_name": "SysGap c",
            "pl_orbper": 40.0,
            "pl_orbsmax": None,
            "pl_bmasse": 8.0,
            "pl_rade": 2.1,
            "pl_dens": 3.5,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "discoverymethod": "Transit",
            "disc_year": 2021,
            "disc_facility": "Synthetic",
            "pl_orbper_was_observed": True,
            "pl_orbsmax_was_observed": False,
            "pl_orbsmax_was_physically_derived": True,
            "pl_bmasse_was_observed": True,
            "pl_rade_was_observed": True,
        },
        {
            "hostname": "SysVal",
            "pl_name": "SysVal b",
            "pl_orbper": 10.0,
            "pl_orbsmax": None,
            "pl_bmasse": 4.0,
            "pl_rade": 1.5,
            "pl_dens": 4.3,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "discoverymethod": "Transit",
            "disc_year": 2019,
            "disc_facility": "Synthetic",
            "pl_orbper_was_observed": True,
            "pl_orbsmax_was_observed": False,
            "pl_orbsmax_was_physically_derived": True,
            "pl_bmasse_was_observed": True,
            "pl_rade_was_observed": True,
        },
        {
            "hostname": "SysVal",
            "pl_name": "SysVal c",
            "pl_orbper": 20.0,
            "pl_orbsmax": None,
            "pl_bmasse": 5.0,
            "pl_rade": 1.7,
            "pl_dens": 4.1,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "discoverymethod": "Transit",
            "disc_year": 2020,
            "disc_facility": "Synthetic",
            "pl_orbper_was_observed": True,
            "pl_orbsmax_was_observed": False,
            "pl_orbsmax_was_physically_derived": True,
            "pl_bmasse_was_observed": True,
            "pl_rade_was_observed": True,
        },
        {
            "hostname": "SysVal",
            "pl_name": "SysVal d",
            "pl_orbper": 40.0,
            "pl_orbsmax": None,
            "pl_bmasse": 6.0,
            "pl_rade": 1.9,
            "pl_dens": 3.8,
            "st_mass": 1.0,
            "st_rad": 1.0,
            "discoverymethod": "Transit",
            "disc_year": 2021,
            "disc_facility": "Synthetic",
            "pl_orbper_was_observed": True,
            "pl_orbsmax_was_observed": False,
            "pl_orbsmax_was_physically_derived": True,
            "pl_bmasse_was_observed": True,
            "pl_rade_was_observed": True,
        },
    ]
    frame = pd.DataFrame(rows)
    return _derive_semimajor_axis(frame)


class SystemMissingPlanetsTests(unittest.TestCase):
    def test_kepler_semimajor_derivation(self) -> None:
        frame = pd.DataFrame({"pl_orbper": [365.25], "pl_orbsmax": [None], "st_mass": [1.0]})
        out = _derive_semimajor_axis(frame)
        self.assertAlmostEqual(float(out.loc[0, "pl_orbsmax"]), 1.0, places=6)
        self.assertEqual(out.loc[0, "pl_orbsmax_system_module_source"], "derived_kepler")

    def test_detects_large_gap_and_creates_candidate(self) -> None:
        catalog = synthetic_catalog()
        system_metadata = build_system_metadata(catalog, min_planets_per_system=2)
        stats = build_gap_statistics(catalog, system_metadata)
        gaps = find_candidate_gaps(catalog[catalog["hostname"] == "SysGap"], system_metadata[system_metadata["hostname"] == "SysGap"], stats, 2.8, 4)
        self.assertEqual(len(gaps), 1)
        candidates = expand_gap_candidates(gaps)
        self.assertEqual(len(candidates), 1)
        self.assertAlmostEqual(float(candidates.loc[0, "candidate_period_days"]), 20.0, places=6)

    def test_empty_topology_resources_do_not_break(self) -> None:
        catalog = synthetic_catalog()
        system_metadata = build_system_metadata(catalog, min_planets_per_system=2)
        stats = build_gap_statistics(catalog, system_metadata)
        gaps = find_candidate_gaps(catalog[catalog["hostname"] == "SysGap"], system_metadata[system_metadata["hostname"] == "SysGap"], stats, 2.8, 4)
        candidates = expand_gap_candidates(gaps)
        out = attach_topology_to_candidates(candidates, TopologyResources())
        self.assertTrue((out["topology_score"] == 0.0).all())

    def test_leave_one_out_recovers_removed_middle_planet(self) -> None:
        catalog = synthetic_catalog()
        system_metadata = build_system_metadata(catalog, min_planets_per_system=2)
        stats = build_gap_statistics(catalog, system_metadata)
        validation, summary = run_leave_one_out_validation(catalog[catalog["hostname"] == "SysVal"], system_metadata[system_metadata["hostname"] == "SysVal"], stats, min_gap_ratio=2.8, max_candidates_per_gap=4)
        self.assertGreaterEqual(int(summary["n_holdout_tests"]), 1)
        self.assertLessEqual(float(validation["abs_logP_error"].min()), 0.1)

    def test_pipeline_runs_without_topology_tables_and_scores_are_bounded(self) -> None:
        catalog = synthetic_catalog()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            catalog_path = tmp / "synthetic_catalog.csv"
            catalog.to_csv(catalog_path, index=False)
            empty_toi = tmp / "empty_toi.csv"
            empty_toi.write_text("node_id,TOI\n", encoding="utf-8")
            empty_ati = tmp / "empty_ati.csv"
            empty_ati.write_text("pl_name,ATI\n", encoding="utf-8")
            empty_shadow = tmp / "empty_shadow.csv"
            empty_shadow.write_text("node_id,shadow_score\n", encoding="utf-8")
            empty_membership = tmp / "empty_membership.csv"
            empty_membership.write_text("node_id,pl_name\n", encoding="utf-8")

            result = run_pipeline(
                SystemMissingPlanetsConfig(
                    catalog=str(catalog_path),
                    output_dir=str(tmp / "outputs"),
                    mode="all",
                    toi_table=str(empty_toi),
                    ati_table=str(empty_ati),
                    shadow_table=str(empty_shadow),
                    node_membership_table=str(empty_membership),
                )
            )
            candidates = result["candidates"]
            self.assertFalse(candidates.empty)
            self.assertTrue(((candidates["candidate_priority_score"] >= 0.0) & (candidates["candidate_priority_score"] <= 1.0)).all())
            self.assertTrue((candidates["topology_score"] == 0.0).all())
            self.assertTrue((Path(result["output_dir"]) / "candidate_missing_planets_by_system.csv").exists())


if __name__ == "__main__":
    unittest.main()
