from __future__ import annotations

import numpy as np
import pandas as pd


def select_top_regions(regions: pd.DataFrame, *, top_n: int = 10, config_id: str | None = None) -> pd.DataFrame:
    df = regions.copy()
    if config_id and "config_id" in df.columns:
        df = df[df["config_id"].astype(str) == str(config_id)]
    sort_col = "TOI" if "TOI" in df.columns else "shadow_score"
    if sort_col not in df.columns:
        return df.head(0)
    return df.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)


def select_top_anchors(anchors: pd.DataFrame, *, top_n: int = 10, config_id: str | None = None) -> pd.DataFrame:
    df = anchors.copy()
    if config_id and "config_id" in df.columns:
        df = df[df["config_id"].astype(str) == str(config_id)]
    sort_col = "ATI" if "ATI" in df.columns else "delta_rel_neighbors_best"
    if sort_col not in df.columns:
        return df.head(0)
    return df.sort_values(sort_col, ascending=False).head(top_n).reset_index(drop=True)


def choose_detailed_cases(
    top_regions: pd.DataFrame,
    top_anchors: pd.DataFrame,
    *,
    default_nodes: list[str] | None = None,
    default_anchors: list[str] | None = None,
    count: int = 5,
) -> pd.DataFrame:
    """Combine top-ranked and user-prioritized cases into a case list."""
    default_nodes = default_nodes or []
    default_anchors = default_anchors or []
    rows = []

    if not top_anchors.empty:
        for _, row in top_anchors.iterrows():
            rows.append({
                "node_id": row.get("node_id"),
                "anchor_pl_name": row.get("anchor_pl_name"),
                "selection_reason": "top_ATI",
            })

    for node in default_nodes:
        rows.append({"node_id": node, "anchor_pl_name": None, "selection_reason": "default_node"})
    for anchor in default_anchors:
        rows.append({"node_id": None, "anchor_pl_name": anchor, "selection_reason": "default_anchor"})

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=["node_id", "anchor_pl_name"], keep="first")
    return out.head(count).reset_index(drop=True)


def select_final_presentation_cases(
    regions: pd.DataFrame,
    anchors: pd.DataFrame,
    deficit_summary: pd.DataFrame,
    *,
    membership: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Select exactly three cases for presentation: region, anchor, repeated anchor."""
    rows: list[dict[str, object]] = []

    region_case = _select_top_toi_region(regions)
    if region_case:
        rows.append(region_case)

    anchor_case = _select_top_ati_anchor(anchors)
    if anchor_case:
        rows.append(anchor_case)

    repeated_case = _select_repeated_anchor(anchors, deficit_summary, membership=membership)
    if repeated_case:
        rows.append(repeated_case)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    desired_order = ["top_toi_region", "top_ati_anchor", "repeated_anchor_multi_node"]
    out["case_order"] = out["case_type"].map({name: idx for idx, name in enumerate(desired_order)})
    out = out.sort_values("case_order").drop(columns="case_order").reset_index(drop=True)
    return out.head(3)


def _select_top_toi_region(regions: pd.DataFrame) -> dict[str, object]:
    if regions.empty:
        return {}
    df = regions.copy()
    for col in ["TOI", "n_members", "I_R3", "S_net"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(["TOI", "n_members", "I_R3", "S_net"], ascending=[False, False, True, False])
    row = df.iloc[0]
    return {
        "case_type": "top_toi_region",
        "selected_id": row.get("node_id"),
        "anchor_pl_name": row.get("anchor_pl_name"),
        "node_id": row.get("node_id"),
        "TOI": row.get("TOI"),
        "ATI": np.nan,
        "shadow_score": row.get("shadow_score"),
        "I_R3": row.get("I_R3"),
        "S_net": row.get("S_net"),
        "C_phys": row.get("C_phys"),
        "Delta_rel_neighbors_best": np.nan,
        "anchor_representativeness": np.nan,
        "deficit_stability_label": "not_anchor_case",
        "n_nodes_as_anchor": np.nan,
        "nodes_list": "",
        "reason_selected": "Mayor TOI global; desempate por n_members, menor I_R3 y mayor S_net.",
        "how_to_present": "Presentar como la region Mapper con mayor prioridad regional. Explicar que gana por la combinacion de sombra observacional, baja imputacion, continuidad fisica y soporte de red.",
        "caution_text": "El caso regional no afirma objetos ausentes; prioriza una zona Mapper para inspeccion observacional.",
    }


def _select_top_ati_anchor(anchors: pd.DataFrame) -> dict[str, object]:
    if anchors.empty:
        return {}
    df = anchors.copy()
    for col in ["ATI", "delta_rel_neighbors_best", "r3_imputation_score", "anchor_representativeness"]:
        if col not in df.columns:
            df[col] = np.nan
    df = df.sort_values(
        ["ATI", "delta_rel_neighbors_best", "r3_imputation_score", "anchor_representativeness"],
        ascending=[False, False, True, False],
    )
    row = df.iloc[0]
    return {
        "case_type": "top_ati_anchor",
        "selected_id": f"{row.get('anchor_pl_name')}::{row.get('node_id')}",
        "anchor_pl_name": row.get("anchor_pl_name"),
        "node_id": row.get("node_id"),
        "TOI": row.get("TOI"),
        "ATI": row.get("ATI"),
        "shadow_score": row.get("shadow_score"),
        "I_R3": row.get("I_R3"),
        "S_net": row.get("S_net"),
        "C_phys": row.get("C_phys"),
        "Delta_rel_neighbors_best": row.get("delta_rel_neighbors_best"),
        "anchor_representativeness": row.get("anchor_representativeness"),
        "anchor_I_R3": row.get("r3_imputation_score"),
        "deficit_stability_label": row.get("deficit_stability_label", ""),
        "n_nodes_as_anchor": 1,
        "nodes_list": str(row.get("node_id", "")),
        "reason_selected": "Mayor ATI global; desempate por deficit local, menor imputacion del ancla y mayor representatividad.",
        "how_to_present": "Presentar como el planeta ancla mas prioritario para inspeccion local. Explicar que combina una region TOI alta con deficit local y baja imputacion.",
        "caution_text": "El caso ancla no afirma un objeto ausente; prioriza una vecindad local en R^3.",
    }


def _select_repeated_anchor(
    anchors: pd.DataFrame,
    deficit_summary: pd.DataFrame,
    *,
    membership: pd.DataFrame | None = None,
) -> dict[str, object]:
    if anchors.empty:
        return {}
    df = anchors.copy()
    grouped = (
        df.groupby("anchor_pl_name", dropna=False)
        .agg(
            n_nodes_as_anchor=("node_id", "nunique"),
            max_ATI=("ATI", "max"),
            mean_ATI=("ATI", "mean"),
            sum_ATI=("ATI", "sum"),
            max_TOI=("TOI", "max"),
            nodes_list=("node_id", lambda s: ", ".join(sorted({str(value) for value in s if pd.notna(value)}))),
        )
        .reset_index()
    )
    repeated = grouped[grouped["n_nodes_as_anchor"] >= 2].sort_values(
        ["n_nodes_as_anchor", "max_ATI", "sum_ATI"], ascending=[False, False, False]
    )
    fallback_reason = ""
    if repeated.empty:
        if membership is not None and not membership.empty and "pl_name" in membership.columns and "node_id" in membership.columns:
            membership_grouped = (
                membership.groupby("pl_name")
                .agg(n_nodes_as_member=("node_id", "nunique"), nodes_list=("node_id", lambda s: ", ".join(sorted({str(v) for v in s}))))
                .reset_index()
            )
            membership_repeated = membership_grouped[membership_grouped["n_nodes_as_member"] >= 2]
            if not membership_repeated.empty:
                best_name = membership_repeated.sort_values("n_nodes_as_member", ascending=False).iloc[0]["pl_name"]
                repeated = grouped[grouped["anchor_pl_name"] == best_name]
        if repeated.empty:
            fallback_reason = "no_repeated_anchor_found"
            if len(df) < 2:
                return {}
            row = df.sort_values("ATI", ascending=False).iloc[1]
            return {
                "case_type": "repeated_anchor_multi_node",
                "selected_id": f"{row.get('anchor_pl_name')}::{row.get('node_id')}",
                "anchor_pl_name": row.get("anchor_pl_name"),
                "node_id": row.get("node_id"),
                "TOI": row.get("TOI"),
                "ATI": row.get("ATI"),
                "shadow_score": row.get("shadow_score"),
                "I_R3": row.get("I_R3"),
                "S_net": row.get("S_net"),
                "C_phys": row.get("C_phys"),
                "Delta_rel_neighbors_best": row.get("delta_rel_neighbors_best"),
                "anchor_representativeness": row.get("anchor_representativeness"),
                "anchor_I_R3": row.get("r3_imputation_score"),
                "deficit_stability_label": row.get("deficit_stability_label", ""),
                "n_nodes_as_anchor": 1,
                "nodes_list": str(row.get("node_id", "")),
                "reason_selected": f"Fallback por {fallback_reason}.",
                "how_to_present": "Presentar como un caso auxiliar cuando no aparece un ancla repetida. Explicar que el ranking sigue siendo util, pero no ilustra solapamiento Mapper de forma directa.",
                "caution_text": "No se encontro un ancla repetida; este reemplazo no debe venderse como caso de transicion topologica.",
            }

    repeated_row = repeated.iloc[0]
    anchor_rows = df[df["anchor_pl_name"] == repeated_row["anchor_pl_name"]].copy()
    if "delta_rel_neighbors_best" in anchor_rows.columns:
        anchor_rows = anchor_rows.sort_values(["ATI", "delta_rel_neighbors_best"], ascending=[False, False])
    else:
        anchor_rows = anchor_rows.sort_values("ATI", ascending=False)
    row = anchor_rows.iloc[0]
    stability = ""
    if not deficit_summary.empty:
        summary_match = deficit_summary[
            (deficit_summary["anchor_pl_name"] == row.get("anchor_pl_name"))
            & (deficit_summary["node_id"] == row.get("node_id"))
        ]
        if not summary_match.empty and "deficit_stability_label" in summary_match.columns:
            stability = summary_match.iloc[0]["deficit_stability_label"]
    return {
        "case_type": "repeated_anchor_multi_node",
        "selected_id": repeated_row.get("anchor_pl_name"),
        "anchor_pl_name": row.get("anchor_pl_name"),
        "node_id": row.get("node_id"),
        "TOI": row.get("TOI"),
        "ATI": row.get("ATI"),
        "shadow_score": row.get("shadow_score"),
        "I_R3": row.get("I_R3"),
        "S_net": row.get("S_net"),
        "C_phys": row.get("C_phys"),
        "Delta_rel_neighbors_best": row.get("delta_rel_neighbors_best"),
        "anchor_representativeness": row.get("anchor_representativeness"),
        "anchor_I_R3": row.get("r3_imputation_score"),
        "deficit_stability_label": stability,
        "n_nodes_as_anchor": repeated_row.get("n_nodes_as_anchor"),
        "nodes_list": repeated_row.get("nodes_list"),
        "reason_selected": "Ancla repetida en varios nodos Mapper; priorizada por frecuencia de aparicion y ATI maximo.",
        "how_to_present": "Presentar como un caso de transicion Mapper. El planeta aparece en varios nodos por el solapamiento de la cubierta, lo que puede indicar que vive en una frontera topologica entre vecindarios.",
        "caution_text": "El caso repetido no es duplicacion erronea; puede reflejar solapamiento de cubiertas y transicion topologica.",
    }
