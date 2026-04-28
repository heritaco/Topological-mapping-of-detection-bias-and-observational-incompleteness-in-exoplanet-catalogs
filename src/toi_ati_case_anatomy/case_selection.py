from __future__ import annotations

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
