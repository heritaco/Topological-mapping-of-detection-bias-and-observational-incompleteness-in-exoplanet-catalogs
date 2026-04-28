from __future__ import annotations
import argparse
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd

from .config import load_config
from .paths import resolve_output_dirs
from .io import (
    load_required_tables, normalize_node_column, normalize_planet_column,
    normalize_edge_columns, filter_config, write_csv, write_json
)
from .r3_geometry import R3Columns, add_r3_coordinates
from .network_metrics import build_graph, graph_metrics, neighbors
from .regional_index import compute_regional_toi
from .anchor_index import select_anchor, compute_neighbor_deficits, best_positive_deficit, anchor_quality
from .plotting import plot_top_regions, plot_top_anchors, plot_toi_vs_deficit
from .reporting import write_interpretation_summary, write_latex_template

def git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:
        return None

def node_planets(planet_members: pd.DataFrame, node_id: str) -> pd.DataFrame:
    return planet_members[planet_members["node_id"].astype(str) == str(node_id)].copy()

def planets_for_nodes(planet_members: pd.DataFrame, node_ids: set[str]) -> pd.DataFrame:
    return planet_members[planet_members["node_id"].astype(str).isin({str(n) for n in node_ids})].copy()

def run(config_path: str) -> None:
    cfg = load_config(config_path)
    out_dirs = resolve_output_dirs(cfg.root, cfg.outputs)
    tables, input_paths = load_required_tables(cfg.root, cfg.inputs)

    node_metrics = normalize_node_column(tables["node_shadow_metrics"])
    members = normalize_node_column(normalize_planet_column(tables["mapper_membership"]))
    edges = normalize_edge_columns(tables["mapper_edges"])
    catalog = normalize_planet_column(tables["imputed_catalog"])

    config_id = cfg.analysis.get("config_id", "orbital_pca2_cubes10_overlap0p35")
    node_metrics = filter_config(node_metrics, config_id)
    members = filter_config(members, config_id)
    edges = filter_config(edges, config_id)

    r3_vars = cfg.analysis.get("r3_variables", {})
    r3_cols = R3Columns(
        mass=r3_vars.get("mass", "pl_bmasse"),
        period=r3_vars.get("period", "pl_orbper"),
        semimajor=r3_vars.get("semimajor", "pl_orbsmax"),
    )
    catalog_r3, r3_meta = add_r3_coordinates(catalog, r3_cols)
    planet_members = members.merge(catalog_r3, on="pl_name", how="left", suffixes=("", "_catalog"))

    g = build_graph(edges, all_nodes=node_metrics["node_id"].astype(str).tolist())
    net = graph_metrics(g)
    nodes = node_metrics.merge(net, on="node_id", how="left", suffixes=("", "_network"))

    regional = compute_regional_toi(nodes, cfg.analysis)
    write_csv(regional, out_dirs["tables"] / "regional_toi_scores.csv")

    preferred = cfg.analysis.get("preferred_method", "Radial Velocity")
    candidates = regional.copy()
    if "top_method" in candidates.columns:
        preferred_rows = candidates[candidates["top_method"].astype(str) == preferred]
        if not preferred_rows.empty:
            candidates = preferred_rows
    if "n_members" in candidates.columns:
        candidates = candidates[pd.to_numeric(candidates["n_members"], errors="coerce").fillna(0) >= int(cfg.analysis.get("min_members_for_regional_ranking", 3))]
    candidates = candidates.sort_values("toi_score", ascending=False).head(int(cfg.analysis.get("max_anchor_nodes", 50)))

    z_cols = r3_cols.z
    anchor_records = []
    deficit_records = []

    for _, region in candidates.iterrows():
        node_id = str(region["node_id"])
        np_node = node_planets(planet_members, node_id)
        anchor = select_anchor(np_node, z_cols, preferred)
        if anchor is None:
            continue

        n1 = neighbors(g, node_id, 1)
        n2 = neighbors(g, node_id, 2)
        np_n1 = planets_for_nodes(planet_members, n1)
        np_n2 = planets_for_nodes(planet_members, n2)

        deficits = compute_neighbor_deficits(anchor, np_node, np_n1, np_n2, z_cols, cfg.analysis)
        best_def = best_positive_deficit(deficits)
        aq = anchor_quality(anchor, np_node, z_cols)
        anchor_imp = float(anchor.get("anchor_r3_imputation_score", 0.0))
        ati = float(region["toi_score"]) * best_def * (1 - anchor_imp) * aq

        rec = {
            "node_id": node_id,
            "anchor_pl_name": anchor.get("pl_name"),
            "discoverymethod": anchor.get("discoverymethod"),
            "disc_year": anchor.get("disc_year"),
            "disc_facility": anchor.get("disc_facility"),
            "pl_bmasse": anchor.get(r3_cols.mass),
            "pl_orbper": anchor.get(r3_cols.period),
            "pl_orbsmax": anchor.get(r3_cols.semimajor),
            "r3_log_mass": anchor.get("r3_log_mass"),
            "r3_log_period": anchor.get("r3_log_period"),
            "r3_log_semimajor": anchor.get("r3_log_semimajor"),
            "anchor_r3_imputation_score": anchor_imp,
            "anchor_quality": aq,
            "toi_score": region.get("toi_score"),
            "shadow_score": region.get("shadow_score", np.nan),
            "top_method": region.get("top_method", None),
            "delta_rel_neighbors_best": best_def,
            "ati_score": ati,
            "expected_missing_direction": "menor masa planetaria o menor proxy RV a escala orbital comparable",
            "caution_text": "ATI is a prioritization score, not a confirmed count of missing exoplanets.",
        }
        anchor_records.append(rec)

        if not deficits.empty:
            d = deficits.copy()
            d.insert(0, "anchor_pl_name", anchor.get("pl_name"))
            d.insert(0, "node_id", node_id)
            deficit_records.append(d)

    anchors = pd.DataFrame(anchor_records)
    if not anchors.empty:
        anchors = anchors.sort_values("ati_score", ascending=False)
    deficits_all = pd.concat(deficit_records, ignore_index=True) if deficit_records else pd.DataFrame()

    write_csv(anchors, out_dirs["tables"] / "anchor_ati_scores.csv")
    write_csv(deficits_all, out_dirs["tables"] / "anchor_neighbor_deficits.csv")

    if cfg.figures.get("enabled", True):
        plot_top_regions(regional, out_dirs["figures"], int(cfg.figures.get("top_n_regions", 25)))
        plot_top_anchors(anchors, out_dirs["figures"], int(cfg.figures.get("top_n_anchors", 25)))
        plot_toi_vs_deficit(anchors, out_dirs["figures"])

    write_interpretation_summary(regional, anchors, out_dirs["root"] / "interpretation_summary.md")
    write_latex_template(out_dirs["latex_dir"])

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(cfg.root),
        "config_path": str(config_path),
        "config_id": config_id,
        "inputs": input_paths,
        "outputs": {k: str(v) for k, v in out_dirs.items()},
        "r3_variables": cfg.analysis.get("r3_variables", {}),
        "r3_meta": r3_meta,
        "n_regions_scored": int(len(regional)),
        "n_anchor_planets_scored": int(len(anchors)),
        "scientific_warning": "Scores rank incompleteness candidates; they do not prove missing exoplanets.",
    }
    write_json(manifest, out_dirs["metadata"] / "run_manifest.json")

def main() -> None:
    parser = argparse.ArgumentParser(description="Compute regional TOI and anchor ATI scores.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(args.config)

if __name__ == "__main__":
    main()
