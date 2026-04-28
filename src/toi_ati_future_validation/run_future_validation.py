from __future__ import annotations

import argparse

from .candidate_ranking import (
    build_final_future_work_cases,
    build_observational_priority_candidates,
    build_technical_audit_cases,
)
from .case_registry import build_case_registry
from .config import load_config
from .deficit_stability import audit_and_recompute_deficits, compute_deficit_stability
from .io import git_commit_hash, load_input_tables, save_log, write_json, write_table
from .observational_context import add_observational_context
from .paths import find_repo_root, output_paths
from .plotting import (
    plot_ati_vs_ati_conservative,
    plot_deficit_profiles_selected_anchors,
    plot_final_future_work_cases,
    plot_observational_priority_ranking,
    plot_rank_shift_after_stability_penalty,
    plot_stable_vs_sensitive_deficit,
    plot_technical_audit_cases_summary,
)
from .reporting import write_latex_report, write_manifest, write_markdown_summary
from .robust_indices import compute_robust_anchor_indices, compute_robust_region_indices
from .sensitivity_analysis import compute_anchor_sensitivity, compute_region_sensitivity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run future validation for TOI/ATI rankings.")
    parser.add_argument("--config", default="configs/toi_ati_future_validation.yaml")
    parser.add_argument("--config-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"analysis": {"config_id": args.config_id}} if args.config_id else None
    cfg = load_config(args.config, overrides=overrides)
    repo_root = find_repo_root()
    out = output_paths(cfg, repo_root)
    tables, warnings, input_paths = load_input_tables(cfg, repo_root)

    regions = tables["regional_toi_scores"].copy()
    anchors = tables["anchor_ati_scores"].copy()
    deficits = tables["anchor_neighbor_deficits"].copy()
    if regions.empty:
        raise FileNotFoundError("regional_toi_scores is required. Run the TOI/ATI index pipeline first.")
    if anchors.empty:
        raise FileNotFoundError("anchor_ati_scores is required. Run the TOI/ATI index pipeline first.")
    if deficits.empty:
        raise FileNotFoundError("anchor_neighbor_deficits is required. Run the TOI/ATI index pipeline first.")

    epsilon = float(cfg.analysis.get("epsilon", 1e-9))
    radii = list(cfg.deficit_stability.get("radii", ["r_kNN", "r_node_median", "r_node_q75"]))
    weight_grid = cfg.robust_indices.get("weight_grid", {})
    candidate_n = int(cfg.analysis.get("candidate_cases_n", 10))

    anchors = add_observational_context(anchors)
    deficits_audited, deficit_audit = audit_and_recompute_deficits(deficits, epsilon=epsilon)
    stability = compute_deficit_stability(deficits_audited, anchors, regions, radii=radii)
    stability = add_observational_context(stability)

    region_sensitivity = compute_region_sensitivity(regions, weight_grid, epsilon=epsilon)
    anchor_sensitivity = compute_anchor_sensitivity(anchors, weight_grid, epsilon=epsilon)

    robust_regions = compute_robust_region_indices(regions, region_sensitivity)
    robust_anchors = compute_robust_anchor_indices(
        anchors,
        stability,
        anchor_sensitivity,
        penalize_negative_large_radius=bool(cfg.robust_indices.get("penalize_negative_large_radius", True)),
        penalize_imputation=bool(cfg.robust_indices.get("penalize_imputation", True)),
        penalize_small_nodes=bool(cfg.robust_indices.get("penalize_small_nodes", True)),
        epsilon=epsilon,
    )
    robust_anchors = add_observational_context(robust_anchors)

    final_cases = build_final_future_work_cases(robust_regions, robust_anchors, candidate_n=5)
    observational_priority = build_observational_priority_candidates(robust_anchors, top_n=candidate_n)
    technical_audit = build_technical_audit_cases(robust_anchors, deficits_audited)
    case_registry = build_case_registry(robust_regions, robust_anchors, final_cases)

    write_table(stability, out["tables"] / "deficit_stability_by_anchor.csv")
    write_table(robust_anchors, out["tables"] / "robust_anchor_indices.csv")
    write_table(robust_regions, out["tables"] / "robust_region_indices.csv")
    write_table(final_cases, out["tables"] / "final_future_work_cases.csv")
    write_table(observational_priority, out["tables"] / "observational_priority_candidates.csv")
    write_table(technical_audit, out["tables"] / "technical_audit_cases.csv")
    write_table(case_registry, out["tables"] / "case_registry.csv")
    write_table(deficits_audited, out["tables"] / "audited_anchor_neighbor_deficits.csv")

    if cfg.report.get("make_figures", True):
        plot_stable_vs_sensitive_deficit(stability, out["figures"] / "stable_vs_sensitive_deficit.pdf")
        plot_ati_vs_ati_conservative(robust_anchors, out["figures"] / "ati_vs_ati_conservative.pdf")
        plot_rank_shift_after_stability_penalty(robust_anchors, out["figures"] / "rank_shift_after_stability_penalty.pdf", top_n=min(10, len(robust_anchors)))
        plot_final_future_work_cases(final_cases, out["figures"] / "final_future_work_cases.pdf")
        plot_deficit_profiles_selected_anchors(stability, out["figures"] / "deficit_profiles_selected_anchors.pdf")
        plot_observational_priority_ranking(observational_priority, out["figures"] / "observational_priority_ranking.pdf")
        plot_technical_audit_cases_summary(technical_audit, out["figures"] / "technical_audit_cases_summary.pdf")

    if cfg.report.get("make_summary", True):
        write_markdown_summary(
            out["base"] / "interpretation_summary.md",
            regions=robust_regions,
            anchors=robust_anchors,
            final_cases=final_cases,
            candidates=observational_priority,
            technical_cases=technical_audit,
        )

    if cfg.report.get("make_latex", True):
        write_latex_report(out["latex"] / "toi_ati_future_validation.tex")

    summary_counts = {
        "n_regions": int(len(robust_regions)),
        "n_anchors": int(len(robust_anchors)),
        "n_stable_cases": int(stability["deficit_stability_class"].astype(str).isin(["stable_positive_deficit", "small_but_stable_deficit"]).sum()) if not stability.empty else 0,
        "n_radius_sensitive_cases": int(stability["deficit_stability_class"].astype(str).isin(["radius_sensitive_deficit", "unstable_due_to_large_radius"]).sum()) if not stability.empty else 0,
    }
    write_manifest(
        out["metadata"] / "run_manifest.json",
        config_path=args.config,
        config=cfg.raw,
        input_paths=input_paths,
        output_paths={key: str(value) for key, value in out.items()},
        warnings=warnings,
        summary_counts=summary_counts,
        commit_hash=git_commit_hash(repo_root),
    )
    write_json(deficit_audit, out["metadata"] / "deficit_audit.json")
    save_log(warnings, out["logs"] / "toi_ati_future_validation.log")

    print("TOI/ATI future validation complete.")
    print(f"Outputs: {out['base']}")
    print(f"LaTeX: {out['latex'] / 'toi_ati_future_validation.tex'}")


if __name__ == "__main__":
    main()
