from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapper_tda.feature_sets import MAPPER_FEATURE_SPACES
from mapper_tda.io import (
    DEFAULT_MAPPER_OUTPUTS_DIR,
    PROJECT_ROOT,
    resolve_mapper_features_path,
    resolve_outputs_dir,
)
from mapper_tda.metrics import build_node_table, compute_graph_metrics, mapper_graph_to_networkx
from mapper_tda.pipeline import MapperConfig, config_id
from mapper_tda.static_outputs import write_figures, write_latex_report


def synthetic_physical_df() -> pd.DataFrame:
    rows = 6
    return pd.DataFrame(
        {
            "pl_name": [f"planet_{idx}" for idx in range(rows)],
            "hostname": [f"star_{idx // 2}" for idx in range(rows)],
            "discoverymethod": ["Transit", "RV", "Transit", "Transit", "RV", "Imaging"],
            "disc_year": [2010, 2012, 2014, 2016, 2018, 2020],
            "pl_rade": [1.0, 1.2, 2.2, 3.1, 8.0, 11.5],
            "pl_bmasse": [1.1, 1.4, 5.0, 8.5, 120.0, 300.0],
            "pl_dens": [5.4, 5.1, 3.0, 2.5, 1.1, 0.8],
            "pl_orbper": [5.0, 9.0, 20.0, 80.0, 300.0, 900.0],
            "pl_orbsmax": [0.05, 0.08, 0.15, 0.45, 1.5, 3.0],
            "pl_insol": [1000.0, 600.0, 220.0, 80.0, 3.0, 0.3],
            "pl_eqt": [1400.0, 1100.0, 650.0, 420.0, 180.0, 90.0],
            "pl_rade_source": ["observed", "observed", "imputed_iterative", "observed", "observed", "observed"],
            "pl_bmasse_source": ["observed"] * rows,
            "pl_dens_source": ["derived_density", "observed", "derived_density", "observed", "imputed_iterative", "observed"],
            "pl_orbper_source": ["observed"] * rows,
            "pl_orbsmax_source": ["derived_kepler", "observed", "observed", "imputed_iterative", "observed", "observed"],
            "pl_insol_source": ["observed"] * rows,
            "pl_eqt_source": ["observed"] * rows,
        }
    ).assign(
        pl_rade_was_observed=lambda df: df["pl_rade_source"].eq("observed"),
        pl_rade_was_physically_derived=False,
        pl_rade_was_imputed=lambda df: df["pl_rade_source"].str.startswith("imputed_"),
        pl_bmasse_was_observed=True,
        pl_bmasse_was_physically_derived=False,
        pl_bmasse_was_imputed=False,
        pl_dens_was_observed=lambda df: df["pl_dens_source"].eq("observed"),
        pl_dens_was_physically_derived=lambda df: df["pl_dens_source"].isin(["derived_density", "derived_kepler"]),
        pl_dens_was_imputed=lambda df: df["pl_dens_source"].str.startswith("imputed_"),
        pl_orbper_was_observed=True,
        pl_orbper_was_physically_derived=False,
        pl_orbper_was_imputed=False,
        pl_orbsmax_was_observed=lambda df: df["pl_orbsmax_source"].eq("observed"),
        pl_orbsmax_was_physically_derived=lambda df: df["pl_orbsmax_source"].eq("derived_kepler"),
        pl_orbsmax_was_imputed=lambda df: df["pl_orbsmax_source"].str.startswith("imputed_"),
        pl_insol_was_observed=True,
        pl_insol_was_physically_derived=False,
        pl_insol_was_imputed=False,
        pl_eqt_was_observed=True,
        pl_eqt_was_physically_derived=False,
        pl_eqt_was_imputed=False,
    )


class MapperDefaultsTests(unittest.TestCase):
    def test_mapper_default_input_iterative(self) -> None:
        resolved = resolve_mapper_features_path(input_method="iterative")
        self.assertTrue(resolved.name.endswith("mapper_features_imputed_iterative.csv"))

    def test_no_knn_default(self) -> None:
        resolved = resolve_mapper_features_path(input_method="iterative")
        self.assertNotIn("knn", resolved.name.lower())

    def test_outputs_outside_reports(self) -> None:
        resolved = resolve_outputs_dir(None)
        self.assertEqual(resolved, DEFAULT_MAPPER_OUTPUTS_DIR)
        self.assertNotIn("reports", str(resolved).lower())

    def test_feature_spaces_defined(self) -> None:
        for key in ["phys_min", "phys_density", "orbital", "thermal", "orb_thermal", "joint_no_density", "joint"]:
            self.assertIn(key, MAPPER_FEATURE_SPACES)


class MapperMetricsTests(unittest.TestCase):
    def test_node_table_source_audit(self) -> None:
        graph = {"nodes": {"n1": [0, 1, 2], "n2": [2, 3, 4]}, "links": {"n1": ["n2"], "n2": ["n1"]}, "sample_id_lookup": list(range(5))}
        nx_graph = mapper_graph_to_networkx(graph)
        lens = np.array([[0.0, 0.1], [0.1, 0.2], [0.3, 0.4], [0.4, 0.5], [0.6, 0.7]])
        table = build_node_table(graph, nx_graph, lens, synthetic_physical_df().iloc[:5].copy(), ["pl_rade", "pl_bmasse", "pl_dens"], "cfg")
        for column in ["mean_imputation_fraction", "physically_derived_fraction", "imputed_fraction", "observed_fraction"]:
            self.assertIn(column, table.columns)

    def test_graph_metrics_beta1(self) -> None:
        graph = {
            "nodes": {"n1": [0, 1], "n2": [1, 2], "n3": [2, 3]},
            "links": {"n1": ["n2", "n3"], "n2": ["n1", "n3"], "n3": ["n1", "n2"]},
        }
        nx_graph = mapper_graph_to_networkx(graph)
        metrics = compute_graph_metrics(nx_graph, graph)
        self.assertEqual(metrics["beta_1"], metrics["n_edges"] - metrics["n_nodes"] + metrics["beta_0"])
        self.assertGreaterEqual(metrics["beta_1"], 0)


class StaticOutputTests(unittest.TestCase):
    def test_static_pdf_figures_created(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "config_id": config_id(MapperConfig(space="phys_min", lens="pca2")),
                    "space": "phys_min",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 5,
                    "n_edges": 6,
                    "beta_0": 1,
                    "beta_1": 2,
                    "graph_density": 0.5,
                    "average_degree": 2.4,
                    "average_clustering": 0.1,
                    "mean_node_size": 3.2,
                    "mean_node_imputation_fraction": 0.15,
                    "mean_node_physically_derived_fraction": 0.25,
                    "max_node_imputation_fraction": 0.2,
                    "frac_nodes_high_imputation": 0.0,
                },
                {
                    "config_id": config_id(MapperConfig(space="joint", lens="pca2")),
                    "space": "joint",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 7,
                    "n_edges": 8,
                    "beta_0": 1,
                    "beta_1": 2,
                    "graph_density": 0.38,
                    "average_degree": 2.2,
                    "average_clustering": 0.18,
                    "mean_node_size": 4.1,
                    "mean_node_imputation_fraction": 0.22,
                    "mean_node_physically_derived_fraction": 0.31,
                    "max_node_imputation_fraction": 0.4,
                    "frac_nodes_high_imputation": 0.25,
                },
            ]
        )
        distances_df = pd.DataFrame(
            [{"graph_a": metrics_df.iloc[0]["config_id"], "graph_b": metrics_df.iloc[1]["config_id"], "metric_zscore_l2_distance": 1.5}]
        )
        batch_result = {
            "metrics_df": metrics_df,
            "distances_df": distances_df,
            "results": [],
            "alignment_summary": {"n_rows_mapper_features": 2, "n_rows_physical": 2, "alignment_key_used": "preserved_index", "n_matched_rows": 2, "n_unmatched_rows": 0, "warnings": ""},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch_result, out_dir)
            for filename in ["01_mapper_graph_size_complexity.pdf", "02_mapper_metrics_zscore_heatmap.pdf"]:
                path = out_dir / "figures_pdf" / filename
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_latex_report_created(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "config_id": "joint_pca2_cubes10_overlap0p35",
                    "space": "joint",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 5,
                    "n_edges": 6,
                    "beta_0": 1,
                    "beta_1": 2,
                    "graph_density": 0.5,
                    "average_degree": 2.4,
                    "average_clustering": 0.1,
                    "mean_node_size": 3.2,
                    "mean_node_imputation_fraction": 0.15,
                    "mean_node_physically_derived_fraction": 0.25,
                    "max_node_imputation_fraction": 0.2,
                    "frac_nodes_high_imputation": 0.0,
                }
            ]
        )
        batch_result = {
            "metrics_df": metrics_df,
            "distances_df": pd.DataFrame(),
            "results": [],
            "alignment_summary": {"n_rows_mapper_features": 1, "n_rows_physical": 1, "alignment_key_used": "preserved_index", "n_matched_rows": 1, "n_unmatched_rows": 0, "warnings": ""},
        }
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp) / "outputs" / "mapper"
            figures_dir = outputs_dir / "figures_pdf"
            figures_dir.mkdir(parents=True)
            (figures_dir / "01_mapper_graph_size_complexity.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "02_mapper_metrics_zscore_heatmap.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "04_mapper_nodes_vs_cycles.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "05_mapper_imputation_audit_by_config.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "06_mapper_density_feature_sensitivity.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "07_mapper_lens_sensitivity.pdf").write_bytes(b"%PDF-1.4 test")
            (figures_dir / "08_mapper_imputation_method_sensitivity.pdf").write_bytes(b"%PDF-1.4 test")
            latex_dir = Path(tmp) / "latex"
            write_latex_report(batch_result, outputs_dir, latex_dir)
            report = latex_dir / "mapper_report.tex"
            self.assertTrue(report.exists())
            content = report.read_text(encoding="utf-8")
            self.assertIn("\\documentclass", content)
            self.assertIn("\\includegraphics", (latex_dir / "sections" / "04_mapper_results.tex").read_text(encoding="utf-8"))
            self.assertIn("Mapper", content)
            limitations = (latex_dir / "sections" / "07_limitations.tex").read_text(encoding="utf-8")
            self.assertIn("observed != physically_derived != imputed", limitations)

    def test_no_html_generated_by_default(self) -> None:
        metrics_df = pd.DataFrame(
            [
                {
                    "config_id": "joint_pca2_cubes10_overlap0p35",
                    "space": "joint",
                    "lens": "pca2",
                    "n_cubes": 10,
                    "overlap": 0.35,
                    "n_nodes": 5,
                    "n_edges": 6,
                    "beta_0": 1,
                    "beta_1": 2,
                    "graph_density": 0.5,
                    "average_degree": 2.4,
                    "average_clustering": 0.1,
                    "mean_node_size": 3.2,
                    "mean_node_imputation_fraction": 0.15,
                    "mean_node_physically_derived_fraction": 0.25,
                    "max_node_imputation_fraction": 0.2,
                    "frac_nodes_high_imputation": 0.0,
                }
            ]
        )
        batch_result = {
            "metrics_df": metrics_df,
            "distances_df": pd.DataFrame(),
            "results": [],
            "alignment_summary": {"n_rows_mapper_features": 1, "n_rows_physical": 1, "alignment_key_used": "preserved_index", "n_matched_rows": 1, "n_unmatched_rows": 0, "warnings": ""},
        }
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch_result, out_dir)
            html_files = list(out_dir.rglob("*.html"))
            self.assertEqual(html_files, [])


if __name__ == "__main__":
    unittest.main()
