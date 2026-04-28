from __future__ import annotations

import pandas as pd


def build_case_registry(
    regions: pd.DataFrame,
    anchors: pd.DataFrame,
    final_cases: pd.DataFrame,
) -> pd.DataFrame:
    region_rows: list[dict[str, object]] = []
    if not regions.empty:
        top_cut = pd.to_numeric(regions.get("TOI_original"), errors="coerce").quantile(0.90)
        for _, row in regions.iterrows():
            region_rows.append(
                {
                    "case_kind": "region",
                    "case_id": str(row.get("node_id")),
                    "anchor_pl_name": None,
                    "node_id": row.get("node_id"),
                    "TOI": row.get("TOI_original", row.get("TOI")),
                    "ATI_original": None,
                    "ATI_conservative": None,
                    "deficit_stability_class": None,
                    "is_top_region": float(row.get("TOI_original", 0) or 0) >= float(top_cut or 0),
                    "is_repeated_anchor": False,
                    "is_stable_deficit_case": False,
                    "is_radius_sensitive_case": False,
                    "high_toi_weak_deficit": False,
                    "low_toi_moderate_deficit": False,
                    "source": "robust_region_indices",
                }
            )
    anchor_rows: list[dict[str, object]] = []
    if not anchors.empty:
        repeated = anchors.groupby("anchor_pl_name", dropna=False)["node_id"].nunique().reset_index(name="n_nodes")
        repeated_names = set(repeated[repeated["n_nodes"] >= 2]["anchor_pl_name"].astype(str))
        top_toi_cut = pd.to_numeric(anchors.get("TOI"), errors="coerce").quantile(0.90)
        for _, row in anchors.iterrows():
            stability = str(row.get("deficit_stability_class", ""))
            anchor_rows.append(
                {
                    "case_kind": "anchor",
                    "case_id": f"{row.get('anchor_pl_name')}::{row.get('node_id')}",
                    "anchor_pl_name": row.get("anchor_pl_name"),
                    "node_id": row.get("node_id"),
                    "TOI": row.get("TOI"),
                    "ATI_original": row.get("ATI_original", row.get("ATI")),
                    "ATI_conservative": row.get("ATI_conservative"),
                    "deficit_stability_class": stability,
                    "is_top_region": False,
                    "is_repeated_anchor": str(row.get("anchor_pl_name")) in repeated_names,
                    "is_stable_deficit_case": stability in {"stable_positive_deficit", "small_but_stable_deficit"},
                    "is_radius_sensitive_case": stability in {"radius_sensitive_deficit", "unstable_due_to_large_radius"},
                    "high_toi_weak_deficit": float(row.get("TOI", 0) or 0) >= float(top_toi_cut or 0) and stability == "no_deficit_or_overpopulated",
                    "low_toi_moderate_deficit": float(row.get("TOI", 0) or 0) < float(top_toi_cut or 0) and stability in {"stable_positive_deficit", "radius_sensitive_deficit", "small_but_stable_deficit"},
                    "source": "robust_anchor_indices",
                }
            )
    out = pd.DataFrame(region_rows + anchor_rows)
    if not out.empty and not final_cases.empty:
        final_pairs = {
            (str(row.get("anchor_pl_name")), str(row.get("node_id")))
            for _, row in final_cases.iterrows()
            if pd.notna(row.get("node_id"))
        }
        final_nodes = {str(row.get("node_id")) for _, row in final_cases.iterrows() if pd.notna(row.get("node_id"))}
        out["appears_in_final_cases"] = out.apply(
            lambda row: (
                (row.get("case_kind") == "anchor" and (str(row.get("anchor_pl_name")), str(row.get("node_id"))) in final_pairs)
                or (row.get("case_kind") == "region" and str(row.get("node_id")) in final_nodes)
            ),
            axis=1,
        )
    return out
