from __future__ import annotations

import argparse
import json
from pathlib import Path

from .case_profiles import build_interpretation_sentences, make_anchor_profile, make_region_profile
from .case_selection import (
    choose_detailed_cases,
    select_final_presentation_cases,
    select_top_anchors,
    select_top_regions,
)
from .config import load_config
from .decomposition import (
    add_ati_decomposition,
    add_toi_decomposition,
    audit_deficit_formulas,
    build_top_anchor_radius_tables,
    summarize_deficit_by_radius,
)
from .io import load_input_tables, write_table
from .paths import find_repo_root, output_paths
from .plotting import (
    plot_ati_decomposition,
    plot_deficit_absolute_by_radius,
    plot_deficit_by_radius,
    plot_deficit_relative_by_radius,
    plot_final_presentation_cases_summary,
    plot_toi_decomposition,
    plot_top_anchors,
    plot_top_regions,
)
from .reporting import (
    write_latex_report,
    write_manifest,
    write_markdown_summary,
    write_top_anchor_deficit_tables_tex,
)


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
    top_n_anchors_for_tables = int(cfg.analysis.get("top_n_anchors_for_tables", 5))
    detailed_case_count = int(cfg.analysis.get("detailed_case_count", 5))
    epsilon = float(cfg.analysis.get("epsilon", 1e-9))

    regions = tables.get("regional_toi_scores")
    anchors = tables.get("anchor_ati_scores")
    deficits = tables.get("anchor_neighbor_deficits")
    membership = tables.get("membership")
    if regions is None or regions.empty:
        raise FileNotFoundError("regional_toi_scores.csv is required for case anatomy.")
    if anchors is None or anchors.empty:
        raise FileNotFoundError("anchor_ati_scores.csv is required for case anatomy.")

    toi_decomp = add_toi_decomposition(regions)
    ati_decomp = add_ati_decomposition(anchors)
    audited_deficits, deficit_audit = audit_deficit_formulas(deficits if deficits is not None else regions.head(0), epsilon=epsilon)
    deficit_by_radius = summarize_deficit_by_radius(audited_deficits)

    top_regions = select_top_regions(toi_decomp, top_n=top_n_regions, config_id=config_id)
    top_anchors = select_top_anchors(ati_decomp, top_n=top_n_anchors, config_id=config_id)
    region_context_cols = [
        column
        for column in ["node_id", "shadow_score", "I_R3", "S_net", "C_phys", "top_method", "n_members"]
        if column in toi_decomp.columns
    ]
    if region_context_cols:
        top_anchors = top_anchors.merge(
            toi_decomp[region_context_cols].drop_duplicates(subset=["node_id"]),
            on="node_id",
            how="left",
            suffixes=("", "_region"),
        )

    if not deficit_by_radius.empty:
        top_anchors = top_anchors.merge(
            deficit_by_radius[["node_id", "anchor_pl_name", "deficit_stability_label"]],
            on=["node_id", "anchor_pl_name"],
            how="left",
        )

    cases = choose_detailed_cases(
        top_regions,
        top_anchors,
        default_nodes=cfg.analysis.get("default_case_nodes", []),
        default_anchors=cfg.analysis.get("default_anchor_names", []),
        count=detailed_case_count,
    )
    final_cases = select_final_presentation_cases(
        top_regions,
        top_anchors,
        deficit_by_radius,
        membership=membership if membership is not None else None,
    )
    top_anchor_radius_tables, top_anchor_radius_summary = build_top_anchor_radius_tables(
        top_anchors,
        audited_deficits,
        top_n=top_n_anchors_for_tables,
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
    write_table(top_anchor_radius_tables, out["tables"] / "top_anchor_radius_deficit_tables.csv")
    write_table(top_anchor_radius_summary, out["tables"] / "top_anchor_radius_deficit_summary.csv")
    write_table(final_cases, out["tables"] / "final_presentation_cases.csv")

    if cfg.section("figures").get("make_figures", True):
        plot_top_regions(top_regions, out["figures"] / "top_regions_toi_case_anatomy.pdf")
        plot_top_anchors(top_anchors, out["figures"] / "top_anchors_ati_case_anatomy.pdf")
        plot_toi_decomposition(top_regions, out["figures"] / "toi_factor_decomposition.pdf")
        plot_ati_decomposition(top_anchors, out["figures"] / "ati_factor_decomposition.pdf")
        legacy_figure_info = plot_deficit_by_radius(deficit_by_radius, out["figures"] / "deficit_by_radius_summary.pdf")
        relative_figure_info = plot_deficit_relative_by_radius(top_anchor_radius_tables, out["figures"] / "deficit_relative_by_radius.pdf")
        absolute_figure_info = plot_deficit_absolute_by_radius(top_anchor_radius_tables, out["figures"] / "deficit_absolute_by_radius.pdf")
        plot_final_presentation_cases_summary(final_cases, out["figures"] / "final_presentation_cases_summary.pdf")
    else:
        legacy_figure_info = {"previous_y_column": None, "previous_y_max": None}
        relative_figure_info = {"y_col": None, "y_max": None}
        absolute_figure_info = {"y_col": None, "y_max": None}

    figure5_audit = {
        "figure": "deficit_by_radius_summary.pdf",
        "previous_y_column": legacy_figure_info.get("previous_y_column"),
        "previous_y_max": legacy_figure_info.get("previous_y_max"),
        "recomputed_delta_rel_max": deficit_audit.get("recomputed_delta_rel_max"),
        "recomputed_delta_N_max": deficit_audit.get("recomputed_delta_n_max"),
        "decision": "plot_delta_rel",
        "reason": "La version corregida separa el deficit relativo recomputado desde N_obs y N_exp_neighbors del deficit absoluto, para evitar confundir escalas de conteo con fracciones normalizadas.",
        "warnings": [
            warning
            for warning in [
                "recomputed_delta_rel_gt_one_unexpected" if deficit_audit.get("recomputed_delta_rel_gt_one_count", 0) > 0 else "",
                "legacy_figure_visual_scale_did_not_match_recomputed_delta_rel" if legacy_figure_info.get("previous_y_max", 0) and deficit_audit.get("recomputed_delta_rel_max", 0) and legacy_figure_info.get("previous_y_max", 0) > max(1.0, 5 * deficit_audit.get("recomputed_delta_rel_max", 0)) else "",
            ]
            if warning
        ],
        "relative_figure": {"file": "deficit_relative_by_radius.pdf", "y_column": relative_figure_info.get("y_col"), "y_max": relative_figure_info.get("y_max")},
        "absolute_figure": {"file": "deficit_absolute_by_radius.pdf", "y_column": absolute_figure_info.get("y_col"), "y_max": absolute_figure_info.get("y_max")},
    }
    (out["metadata"] / "figure5_deficit_audit.json").write_text(
        json.dumps(figure5_audit, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    sentences = build_interpretation_sentences(region_profile, anchor_profile)
    write_markdown_summary(
        out["base"] / "interpretation_summary.md",
        sentences=sentences,
        top_regions=top_regions,
        top_anchors=top_anchors,
        top_anchor_radius_summary=top_anchor_radius_summary,
        final_cases=final_cases,
        deficit_audit=deficit_audit,
        figure5_audit=figure5_audit,
    )
    write_manifest(
        out["metadata"] / "run_manifest.json",
        config_path=str(Path(args.config)),
        inputs_loaded={k: int(len(v)) for k, v in tables.items()},
        warnings=warnings,
        audit=deficit_audit,
    )
    if cfg.section("report").get("make_latex", True):
        write_top_anchor_deficit_tables_tex(
            out["latex"] / "top_anchor_deficit_tables.tex",
            top_anchor_radius_tables,
            top_anchor_radius_summary,
        )
        write_latex_report(
            out["latex"] / "toi_ati_case_anatomy.tex",
            final_cases=final_cases,
            top_anchor_deficit_input_path="top_anchor_deficit_tables.tex",
        )

    print("TOI/ATI case anatomy complete.")
    print(f"Tables: {out['tables']}")
    print(f"Figures: {out['figures']}")
    print(f"Summary: {out['base'] / 'interpretation_summary.md'}")


if __name__ == "__main__":
    main()
