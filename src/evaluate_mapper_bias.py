from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from mapper_tda.bias_audit import display_path, resolve_outputs_dir, write_bias_audit_tables
from mapper_tda.bias_nulls import run_discoverymethod_permutation_tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate observational discovery-method bias on existing selected Mapper graphs.",
    )
    parser.add_argument("--outputs-dir", default="outputs/mapper", help="Mapper outputs directory. Default: outputs/mapper.")
    parser.add_argument("--n-perm", type=int, default=1000, help="Number of label permutations. Default: 1000.")
    parser.add_argument("--seed", type=int, default=42, help="Permutation random seed. Default: 42.")
    return parser.parse_args()


def _fmt(value: object, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not np.isfinite(number):
        return "NA"
    return f"{number:.{digits}f}"


def _top_rows(frame: pd.DataFrame, columns: list[str], limit: int = 5) -> str:
    if frame.empty:
        return "None available."
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.head(limit).to_dict(orient="records"):
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(_fmt(value))
            else:
                values.append(str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def _strongest_graph(null_df: pd.DataFrame) -> pd.Series | None:
    if null_df.empty:
        return None
    frame = null_df.copy()
    frame["_score"] = pd.to_numeric(frame.get("nmi_z"), errors="coerce")
    if frame["_score"].isna().all():
        frame["_score"] = pd.to_numeric(frame.get("purity_z"), errors="coerce")
    if frame["_score"].isna().all():
        frame["_score"] = pd.to_numeric(frame.get("observed_weighted_mean_method_purity"), errors="coerce")
    return frame.sort_values("_score", ascending=False).iloc[0]


def _orbital_assessment(null_df: pd.DataFrame) -> str:
    subset = null_df[null_df["config_id"] == "orbital_pca2_cubes10_overlap0p35"] if "config_id" in null_df else pd.DataFrame()
    if subset.empty:
        return "The selected orbital PCA graph was not present in the permutation results."
    row = subset.iloc[0]
    purity = float(row.get("observed_weighted_mean_method_purity", np.nan))
    nmi = float(row.get("observed_nmi", np.nan))
    nmi_p = float(row.get("nmi_empirical_p", np.nan))
    purity_z = float(row.get("purity_z", np.nan))
    if np.isfinite(nmi_p) and nmi_p <= 0.05 and np.isfinite(nmi) and nmi >= 0.25 and np.isfinite(purity) and purity >= 0.80:
        level = "appears strongly dominated by discovery method labels"
    elif (np.isfinite(nmi_p) and nmi_p <= 0.10) or (np.isfinite(purity_z) and purity_z >= 2.0):
        level = "appears partially enriched by discovery method labels, but not fully dominated"
    else:
        level = "does not show strong graph-level discovery-method domination in this audit"
    return (
        f"`orbital_pca2_cubes10_overlap0p35` {level}. "
        f"Observed weighted purity={_fmt(purity)}, observed NMI={_fmt(nmi)}, NMI empirical p={_fmt(nmi_p)}."
    )


def _high_bias_imputation_statement(merged: pd.DataFrame) -> str:
    if merged.empty:
        return "No merged node bias/imputation table was available."
    biased = merged[
        (pd.to_numeric(merged.get("enrichment_z"), errors="coerce") >= 2.0)
        | (pd.to_numeric(merged.get("empirical_p_value"), errors="coerce") <= 0.05)
    ].copy()
    if biased.empty:
        return "No nodes crossed the high-bias heuristic threshold used here (`z >= 2` or empirical p <= 0.05)."
    high_imp = biased[pd.to_numeric(biased.get("mean_imputation_fraction"), errors="coerce") >= 0.30]
    return (
        f"{len(high_imp)} of {len(biased)} high-bias nodes also have mean imputation fraction >= 0.30. "
        f"This checks whether observational enrichment and imputation risk are co-located."
    )


def _join_report_markdown(join_report: pd.DataFrame) -> str:
    columns = ["config_id", "metadata_source", "join_method", "join_coverage", "n_unique_members", "n_members_missing_metadata"]
    return _top_rows(join_report[columns], columns, limit=len(join_report))


def build_summary_markdown(
    node_bias: pd.DataFrame,
    component_bias: pd.DataFrame,
    null_df: pd.DataFrame,
    enrichment_df: pd.DataFrame,
    join_report: pd.DataFrame,
    n_perm: int,
    seed: int,
) -> str:
    strongest = _strongest_graph(null_df)
    strongest_text = (
        "No graph-level permutation results were available."
        if strongest is None
        else (
            f"The strongest graph-level discoverymethod enrichment is `{strongest['config_id']}` "
            f"(purity z={_fmt(strongest.get('purity_z'))}, NMI z={_fmt(strongest.get('nmi_z'))}, "
            f"observed NMI={_fmt(strongest.get('observed_nmi'))})."
        )
    )

    merged_nodes = enrichment_df.merge(
        node_bias,
        on=["config_id", "feature_space", "lens", "node_id"],
        how="left",
        suffixes=("_enrichment", ""),
    )
    top_nodes = merged_nodes.sort_values(
        ["enrichment_z", "observed_dominant_method_fraction"],
        ascending=[False, False],
        na_position="last",
    )
    top_components = component_bias.sort_values(
        ["discoverymethod_js_divergence_vs_global", "dominant_discoverymethod_fraction"],
        ascending=[False, False],
        na_position="last",
    )

    physically_interpretable = merged_nodes[
        (pd.to_numeric(merged_nodes.get("radius_class_purity"), errors="coerce") >= 0.65)
        & (pd.to_numeric(merged_nodes.get("mean_imputation_fraction"), errors="coerce") < 0.15)
        & (pd.to_numeric(merged_nodes.get("discoverymethod_js_divergence_vs_global"), errors="coerce") < 0.15)
    ].sort_values(["radius_class_purity", "n_members"], ascending=[False, False], na_position="last")
    suspicious = merged_nodes[
        (pd.to_numeric(merged_nodes.get("observed_dominant_method_fraction"), errors="coerce") >= 0.80)
        & (
            (pd.to_numeric(merged_nodes.get("enrichment_z"), errors="coerce") >= 2.0)
            | (pd.to_numeric(merged_nodes.get("empirical_p_value"), errors="coerce") <= 0.05)
        )
    ].sort_values(["enrichment_z", "observed_dominant_method_fraction"], ascending=[False, False], na_position="last")

    report_paragraph = (
        "A label-permutation audit was applied to the selected pca2 Mapper graphs while keeping graph topology and node "
        "memberships fixed. The audit compares observed discovery-method concentration against random relabelings of "
        "the same planets. It therefore identifies regions where Mapper structure aligns unusually strongly with "
        "observational metadata, but it does not prove that discovery method caused the topology."
    )

    lines = [
        "# Mapper Bias Audit Summary",
        "",
        f"Permutation settings: `n_perm={n_perm}`, `seed={seed}`.",
        "",
        "## Metadata Join Coverage",
        "",
        _join_report_markdown(join_report),
        "",
        "## Strongest Graph-Level Enrichment",
        "",
        strongest_text,
        "",
        "## Orbital Mapper Assessment",
        "",
        _orbital_assessment(null_df),
        "",
        "## Most Observationally Biased Nodes",
        "",
        _top_rows(
            top_nodes,
            [
                "config_id",
                "node_id",
                "dominant_discoverymethod",
                "observed_dominant_method_fraction",
                "enrichment_z",
                "empirical_p_value",
                "mean_imputation_fraction",
            ],
        ),
        "",
        "## Most Observationally Biased Components",
        "",
        _top_rows(
            top_components,
            [
                "config_id",
                "component_id",
                "dominant_discoverymethod",
                "dominant_discoverymethod_fraction",
                "discoverymethod_js_divergence_vs_global",
                "mean_imputation_fraction",
            ],
        ),
        "",
        "## Bias And Imputation",
        "",
        _high_bias_imputation_statement(merged_nodes),
        "",
        "## More Physically Interpretable Regions",
        "",
        _top_rows(
            physically_interpretable,
            [
                "config_id",
                "node_id",
                "radius_class_dominant",
                "radius_class_purity",
                "dominant_discoverymethod",
                "discoverymethod_js_divergence_vs_global",
                "mean_imputation_fraction",
            ],
        ),
        "",
        "## Observationally Suspicious Regions",
        "",
        _top_rows(
            suspicious,
            [
                "config_id",
                "node_id",
                "dominant_discoverymethod",
                "observed_dominant_method_fraction",
                "enrichment_z",
                "empirical_p_value",
                "radius_class_dominant",
            ],
        ),
        "",
        "## Caution",
        "",
        "This is a label-permutation bias audit, not causal proof. It keeps the Mapper topology fixed and asks whether observed discovery labels are unusually concentrated relative to random label assignments.",
        "",
        "## Report-Ready Paragraph",
        "",
        report_paragraph,
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    outputs_dir = resolve_outputs_dir(args.outputs_dir)
    (outputs_dir / "tables").mkdir(parents=True, exist_ok=True)
    (outputs_dir / "bias_audit").mkdir(parents=True, exist_ok=True)

    node_bias, component_bias, graph_inputs, join_report = write_bias_audit_tables(outputs_dir)
    null_df, enrichment_df = run_discoverymethod_permutation_tests(graph_inputs, n_perm=args.n_perm, seed=args.seed)

    null_path = outputs_dir / "tables" / "discoverymethod_permutation_null.csv"
    enrichment_path = outputs_dir / "tables" / "discoverymethod_enrichment_summary.csv"
    summary_path = outputs_dir / "bias_audit" / "mapper_bias_audit_summary.md"
    null_df.to_csv(null_path, index=False)
    enrichment_df.to_csv(enrichment_path, index=False)
    summary = build_summary_markdown(node_bias, component_bias, null_df, enrichment_df, join_report, args.n_perm, args.seed)
    summary_path.write_text(summary, encoding="utf-8")

    print("Mapper observational bias audit complete.")
    print(f"node_discovery_bias: {display_path(outputs_dir / 'tables' / 'node_discovery_bias.csv')}")
    print(f"component_discovery_bias: {display_path(outputs_dir / 'tables' / 'component_discovery_bias.csv')}")
    print(f"discoverymethod_permutation_null: {display_path(null_path)}")
    print(f"discoverymethod_enrichment_summary: {display_path(enrichment_path)}")
    print(f"summary: {display_path(summary_path)}")


if __name__ == "__main__":
    main()
