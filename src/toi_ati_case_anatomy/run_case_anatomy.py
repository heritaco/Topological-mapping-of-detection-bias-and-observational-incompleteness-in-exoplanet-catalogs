from __future__ import annotations

import argparse
from pathlib import Path

from .case_profiles import build_interpretation_sentences, make_anchor_profile, make_region_profile
from .case_selection import choose_detailed_cases, select_top_anchors, select_top_regions
from .config import load_config
from .decomposition import add_ati_decomposition, add_toi_decomposition, summarize_deficit_by_radius
from .io import load_input_tables, write_table
from .paths import find_repo_root, output_paths
from .plotting import (
    plot_ati_decomposition,
    plot_deficit_by_radius,
    plot_toi_decomposition,
    plot_top_anchors,
    plot_top_regions,
)
from .reporting import write_latex_report, write_manifest, write_markdown_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build TOI/ATI case-anatomy outputs.")
    parser.add_argument("--config", default="configs/toi_ati_case_anatomy.yaml")
    parser.add_argument("--config-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    repo_root = find_repo_root()
    out = output_paths(cfg, repo_root)
    tables, warnings = load_input_tables(cfg, repo_root)

    config_id = args.config_id or cfg.analysis.get("config_id")
    top_n_regions = int(cfg.analysis.get("top_n_regions", 10))
    top_n_anchors = int(cfg.analysis.get("top_n_anchors", 10))
    detailed_case_count = int(cfg.analysis.get("detailed_case_count", 5))

    regions = tables.get("regional_toi_scores")
    anchors = tables.get("anchor_ati_scores")
    deficits = tables.get("anchor_neighbor_deficits")
    if regions is None or regions.empty:
        raise FileNotFoundError("regional_toi_scores.csv is required for case anatomy.")
    if anchors is None or anchors.empty:
        raise FileNotFoundError("anchor_ati_scores.csv is required for case anatomy.")

    toi_decomp = add_toi_decomposition(regions)
    ati_decomp = add_ati_decomposition(anchors)
    deficit_by_radius = summarize_deficit_by_radius(deficits if deficits is not None else regions.head(0))

    top_regions = select_top_regions(toi_decomp, top_n=top_n_regions, config_id=config_id)
    top_anchors = select_top_anchors(ati_decomp, top_n=top_n_anchors, config_id=config_id)
    cases = choose_detailed_cases(
        top_regions,
        top_anchors,
        default_nodes=cfg.analysis.get("default_case_nodes", []),
        default_anchors=cfg.analysis.get("default_anchor_names", []),
        count=detailed_case_count,
    )

    region_profile = make_region_profile(top_regions, toi_decomp)
    anchor_profile = make_anchor_profile(top_anchors, ati_decomp)

    write_table(toi_decomp, out["tables"] / "toi_decomposition_all_regions.csv")
    write_table(ati_decomp, out["tables"] / "ati_decomposition_all_anchors.csv")
    write_table(top_regions, out["tables"] / "top_regions_case_anatomy.csv")
    write_table(top_anchors, out["tables"] / "top_anchors_case_anatomy.csv")
    write_table(region_profile, out["tables"] / "top_region_profiles.csv")
    write_table(anchor_profile, out["tables"] / "top_anchor_profiles.csv")
    write_table(deficit_by_radius, out["tables"] / "deficit_by_radius_summary.csv")
    write_table(cases, out["tables"] / "detailed_cases_to_review.csv")

    if cfg.section("figures").get("make_figures", True):
        plot_top_regions(top_regions, out["figures"] / "top_regions_toi_case_anatomy.pdf")
        plot_top_anchors(top_anchors, out["figures"] / "top_anchors_ati_case_anatomy.pdf")
        plot_toi_decomposition(top_regions, out["figures"] / "toi_factor_decomposition.pdf")
        plot_ati_decomposition(top_anchors, out["figures"] / "ati_factor_decomposition.pdf")
        plot_deficit_by_radius(deficit_by_radius, out["figures"] / "deficit_by_radius_summary.pdf")

    sentences = build_interpretation_sentences(region_profile, anchor_profile)
    write_markdown_summary(
        out["base"] / "interpretation_summary.md",
        sentences=sentences,
        top_regions=top_regions,
        top_anchors=top_anchors,
    )
    write_manifest(
        out["metadata"] / "run_manifest.json",
        config_path=str(Path(args.config)),
        inputs_loaded={k: int(len(v)) for k, v in tables.items()},
        warnings=warnings,
    )
    if cfg.section("report").get("make_latex", True):
        write_latex_report(out["latex"] / "toi_ati_case_anatomy.tex")

    print("TOI/ATI case anatomy complete.")
    print(f"Tables: {out['tables']}")
    print(f"Figures: {out['figures']}")
    print(f"Summary: {out['base'] / 'interpretation_summary.md'}")


if __name__ == "__main__":
    main()
