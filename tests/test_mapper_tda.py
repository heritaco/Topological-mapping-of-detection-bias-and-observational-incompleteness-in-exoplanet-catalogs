from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mapper_tda.feature_sets import MAPPER_FEATURE_SPACES
from mapper_tda.io import DEFAULT_MAPPER_OUTPUTS_DIR, resolve_mapper_features_path, resolve_outputs_dir
from mapper_tda.metrics import build_node_table, compute_graph_metrics, mapper_graph_to_networkx
from mapper_tda.node_selection import build_component_summary, build_highlighted_nodes
from mapper_tda.pipeline import MapperConfig, config_id
from mapper_tda.planet_classes import add_planet_physical_labels
from mapper_tda.static_outputs import (
    write_figures,
    write_interpretation_figures,
    write_interpretation_tables,
    write_latex_report,
    write_validation_outputs,
)


def synthetic_physical_df() -> pd.DataFrame:
    rows = 10
    return pd.DataFrame(
        {
            "pl_name": [f"planet_{idx}" for idx in range(rows)],
            "hostname": [f"star_{idx // 2}" for idx in range(rows)],
            "discoverymethod": ["Transit", "RV", "Transit", "Transit", "RV", "Imaging", "Transit", "RV", "Transit", "Imaging"],
            "disc_year": [2010 + idx for idx in range(rows)],
            "pl_rade": [1.0, 1.3, 2.2, 3.1, 8.0, 11.5, 1.2, 9.5, 4.5, 2.0],
            "pl_bmasse": [1.1, 1.4, 5.0, 8.5, 120.0, 300.0, 2.0, 200.0, 25.0, 6.0],
            "pl_dens": [5.4, 5.1, 3.0, 2.5, 1.1, 0.8, 4.0, 1.2, 2.2, 3.5],
            "pl_orbper": [5.0, 9.0, 20.0, 80.0, 300.0, 900.0, 2.0, 6.0, 150.0, 40.0],
            "pl_orbsmax": [0.05, 0.08, 0.15, 0.45, 1.5, 3.0, 0.03, 0.06, 1.8, 0.3],
            "pl_insol": [1000.0, 600.0, 220.0, 80.0, 3.0, 0.3, 2500.0, 1600.0, 2.0, 120.0],
            "pl_eqt": [1400.0, 1100.0, 650.0, 420.0, 180.0, 90.0, 1800.0, 1600.0, 150.0, 500.0],
            "pl_rade_source": ["observed", "observed", "imputed_iterative", "observed", "observed", "observed", "observed", "observed", "observed", "observed"],
            "pl_bmasse_source": ["observed"] * rows,
            "pl_dens_source": ["derived_density", "observed", "derived_density", "observed", "imputed_iterative", "observed", "observed", "derived_density", "observed", "observed"],
            "pl_orbper_source": ["observed"] * rows,
            "pl_orbsmax_source": ["derived_kepler", "observed", "observed", "imputed_iterative", "observed", "observed", "observed", "observed", "derived_kepler", "observed"],
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


def synthetic_result(config_name: str = "orbital_pca2_cubes10_overlap0p35") -> dict:
    graph = {
        "nodes": {
            "n1": [0, 1, 2, 3],
            "n2": [3, 4, 5],
            "n3": [6, 7, 8],
            "n4": [8, 9],
        },
        "links": {"n1": ["n2"], "n2": ["n1"], "n3": ["n4"], "n4": ["n3"]},
        "sample_id_lookup": list(range(10)),
    }
    physical = synthetic_physical_df()
    nx_graph = mapper_graph_to_networkx(graph)
    lens = np.array([[0.0, 0.1], [0.1, 0.2], [0.3, 0.4], [0.4, 0.5], [0.6, 0.7], [0.7, 0.8], [0.2, 0.1], [0.1, 0.0], [0.5, 0.2], [0.8, 0.3]])
    node_table = build_node_table(graph, nx_graph, lens, physical.copy(), ["pl_rade", "pl_bmasse", "pl_dens", "pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"], config_name)
    return {
        "config": MapperConfig(space="orbital", lens="pca2"),
        "config_id": config_name,
        "graph": graph,
        "nx_graph": nx_graph,
        "lens": lens,
        "mapper_df": pd.DataFrame({"pl_orbper": np.linspace(0, 1, 10), "pl_orbsmax": np.linspace(1, 2, 10)}),
        "physical_df": add_planet_physical_labels(physical.copy()),
        "used_features": ["pl_orbper", "pl_orbsmax", "pl_insol", "pl_eqt"],
        "node_table": node_table,
        "graph_metrics": compute_graph_metrics(nx_graph, graph),
    }


def synthetic_batch_result() -> dict:
    result = synthetic_result()
    metrics_df = pd.DataFrame(
        [
            {
                "config_id": "phys_min_pca2_cubes10_overlap0p35",
                "space": "phys_min",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 6,
                "n_edges": 7,
                "beta_0": 1,
                "beta_1": 2,
                "graph_density": 0.4,
                "average_degree": 2.3,
                "average_clustering": 0.2,
                "mean_node_size": 4.0,
                "mean_node_imputation_fraction": 0.05,
                "mean_node_physically_derived_fraction": 0.0,
                "max_node_imputation_fraction": 0.12,
                "frac_nodes_high_imputation": 0.0,
                "largest_component_fraction": 0.8,
            },
            {
                "config_id": "phys_density_pca2_cubes10_overlap0p35",
                "space": "phys_density",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 5,
                "n_edges": 5,
                "beta_0": 1,
                "beta_1": 1,
                "graph_density": 0.3,
                "average_degree": 2.0,
                "average_clustering": 0.2,
                "mean_node_size": 4.0,
                "mean_node_imputation_fraction": 0.02,
                "mean_node_physically_derived_fraction": 0.35,
                "max_node_imputation_fraction": 0.08,
                "frac_nodes_high_imputation": 0.0,
                "largest_component_fraction": 0.8,
            },
            {
                "config_id": "orbital_pca2_cubes10_overlap0p35",
                "space": "orbital",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 12,
                "n_edges": 20,
                "beta_0": 2,
                "beta_1": 10,
                "graph_density": 0.25,
                "average_degree": 3.3,
                "average_clustering": 0.5,
                "mean_node_size": 4.2,
                "mean_node_imputation_fraction": 0.01,
                "mean_node_physically_derived_fraction": 0.02,
                "max_node_imputation_fraction": 0.05,
                "frac_nodes_high_imputation": 0.0,
                "largest_component_fraction": 0.7,
            },
            {
                "config_id": "joint_no_density_pca2_cubes10_overlap0p35",
                "space": "joint_no_density",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 9,
                "n_edges": 14,
                "beta_0": 1,
                "beta_1": 6,
                "graph_density": 0.2,
                "average_degree": 3.0,
                "average_clustering": 0.4,
                "mean_node_size": 5.0,
                "mean_node_imputation_fraction": 0.12,
                "mean_node_physically_derived_fraction": 0.01,
                "max_node_imputation_fraction": 0.18,
                "frac_nodes_high_imputation": 0.1,
                "largest_component_fraction": 0.75,
            },
            {
                "config_id": "joint_pca2_cubes10_overlap0p35",
                "space": "joint",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 8,
                "n_edges": 11,
                "beta_0": 1,
                "beta_1": 4,
                "graph_density": 0.18,
                "average_degree": 2.7,
                "average_clustering": 0.35,
                "mean_node_size": 5.0,
                "mean_node_imputation_fraction": 0.14,
                "mean_node_physically_derived_fraction": 0.14,
                "max_node_imputation_fraction": 0.2,
                "frac_nodes_high_imputation": 0.1,
                "largest_component_fraction": 0.7,
            },
            {
                "config_id": "thermal_pca2_cubes10_overlap0p35",
                "space": "thermal",
                "lens": "pca2",
                "n_cubes": 10,
                "overlap": 0.35,
                "n_nodes": 10,
                "n_edges": 13,
                "beta_0": 1,
                "beta_1": 4,
                "graph_density": 0.15,
                "average_degree": 2.6,
                "average_clustering": 0.3,
                "mean_node_size": 4.8,
                "mean_node_imputation_fraction": 0.55,
                "mean_node_physically_derived_fraction": 0.0,
                "max_node_imputation_fraction": 0.9,
                "frac_nodes_high_imputation": 0.8,
                "largest_component_fraction": 0.5,
            },
        ]
    )
    return {
        "metrics_df": metrics_df,
        "distances_df": pd.DataFrame([{"graph_a": "phys_min_pca2_cubes10_overlap0p35", "graph_b": "phys_density_pca2_cubes10_overlap0p35", "metric_zscore_l2_distance": 1.2}]),
        "results": [result],
        "alignment_summary": {"n_rows_mapper_features": 10, "n_rows_physical": 10, "alignment_key_used": "preserved_index", "n_matched_rows": 10, "n_unmatched_rows": 0, "warnings": ""},
    }


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


class InterpretationTests(unittest.TestCase):
    def test_planet_physical_labels(self) -> None:
        labels = add_planet_physical_labels(synthetic_physical_df())
        for column in ["radius_class", "orbit_class", "thermal_class", "candidate_population"]:
            self.assertIn(column, labels.columns)

    def test_node_table_source_audit(self) -> None:
        table = synthetic_result()["node_table"]
        for column in ["mean_imputation_fraction", "physically_derived_fraction", "imputed_fraction", "observed_fraction"]:
            self.assertIn(column, table.columns)

    def test_graph_metrics_beta1(self) -> None:
        graph = {"nodes": {"n1": [0, 1], "n2": [1, 2], "n3": [2, 3]}, "links": {"n1": ["n2", "n3"], "n2": ["n1", "n3"], "n3": ["n1", "n2"]}}
        nx_graph = mapper_graph_to_networkx(graph)
        metrics = compute_graph_metrics(nx_graph, graph)
        self.assertEqual(metrics["beta_1"], metrics["n_edges"] - metrics["n_nodes"] + metrics["beta_0"])
        self.assertGreaterEqual(metrics["beta_1"], 0)

    def test_highlighted_nodes_created(self) -> None:
        highlighted = build_highlighted_nodes(synthetic_result())
        self.assertFalse(highlighted.empty)
        self.assertIn("interpretation_text", highlighted.columns)

    def test_component_summary_created(self) -> None:
        summary = build_component_summary(synthetic_result())
        self.assertFalse(summary.empty)
        self.assertTrue((summary["beta_1_component"] >= 0).all())


class StaticOutputTests(unittest.TestCase):
    def test_node_physical_interpretation_columns(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            tables = write_interpretation_tables(batch, out_dir)
            path = out_dir / "tables" / "node_physical_interpretation.csv"
            self.assertTrue(path.exists())
            frame = pd.read_csv(path)
            for column in ["candidate_population_top", "radius_class_top", "orbit_class_top", "thermal_class_top", "mean_imputation_fraction"]:
                self.assertIn(column, frame.columns)

    def test_static_pdf_figures_created(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch, out_dir)
            for filename in ["01_mapper_graph_size_complexity.pdf", "02_mapper_metrics_zscore_heatmap.pdf"]:
                path = out_dir / "figures_pdf" / filename
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_interpretation_figures_created(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch, out_dir)
            tables = write_interpretation_tables(batch, out_dir)
            write_interpretation_figures(batch, out_dir, tables, {})
            for filename in ["01_main_graphs_by_population.pdf", "02_main_graphs_by_imputation.pdf", "04_orbital_mapper_interpretation.pdf"]:
                path = out_dir / "figures_pdf" / "interpretation" / filename
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_latex_report_created(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch, outputs_dir)
            tables = write_interpretation_tables(batch, outputs_dir)
            write_interpretation_figures(batch, outputs_dir, tables, {})
            latex_dir = Path(tmp) / "latex"
            write_latex_report(batch, outputs_dir, latex_dir, tables, {})
            report = latex_dir / "mapper_report.tex"
            self.assertTrue(report.exists())
            content = report.read_text(encoding="utf-8")
            self.assertIn("\\documentclass", content)
            self.assertIn("Key Findings", content)
            self.assertIn("component_summary_short", content)

    def test_latex_no_giant_tables(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch, outputs_dir)
            tables = write_interpretation_tables(batch, outputs_dir)
            latex_dir = Path(tmp) / "latex"
            write_latex_report(batch, outputs_dir, latex_dir, tables, {})
            content = (latex_dir / "mapper_report.tex").read_text(encoding="utf-8")
            self.assertNotIn("mapper_graph_metrics.csv", content)

    def test_bootstrap_optional(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp) / "outputs" / "mapper"
            validation = write_validation_outputs(batch, outputs_dir, run_bootstrap=True, n_bootstrap=2, bootstrap_frac=0.8)
            self.assertTrue((outputs_dir / "bootstrap" / "bootstrap_metrics.csv").exists())

    def test_null_models_optional(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            outputs_dir = Path(tmp) / "outputs" / "mapper"
            validation = write_validation_outputs(batch, outputs_dir, run_null=True, n_null=2)
            self.assertTrue((outputs_dir / "null_models" / "null_model_metrics.csv").exists())

    def test_interpretive_summary_created(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            tables = write_interpretation_tables(batch, out_dir)
            self.assertTrue((out_dir / "tables" / "mapper_interpretive_summary.tex").exists())

    def test_no_html_generated_by_default(self) -> None:
        batch = synthetic_batch_result()
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "outputs" / "mapper"
            write_figures(batch, out_dir)
            html_files = list(out_dir.rglob("*.html"))
            self.assertEqual(html_files, [])


if __name__ == "__main__":
    unittest.main()
