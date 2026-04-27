from __future__ import annotations

import json
from typing import Any

import pandas as pd


def split_node_id(node_id: str) -> tuple[str | None, int | None]:
    text = str(node_id)
    if "_cluster" not in text:
        return None, None
    cube_id, cluster_raw = text.split("_cluster", 1)
    try:
        return cube_id, int(cluster_raw)
    except ValueError:
        return cube_id, None


def parse_member_indices(value: Any) -> list[int]:
    if isinstance(value, list):
        return [int(item) for item in value]
    if pd.isna(value):
        return []
    try:
        parsed = json.loads(value)
    except Exception:
        return []
    if not isinstance(parsed, list):
        return []
    return [int(item) for item in parsed]


def _membership_row(
    config_id: str,
    node_id: str,
    row_index: int,
    physical_df: pd.DataFrame,
) -> dict[str, Any]:
    cube_id, cluster_id = split_node_id(node_id)
    if 0 <= row_index < len(physical_df):
        member = physical_df.iloc[row_index]
        pl_name = member.get("pl_name")
        hostname = member.get("hostname")
    else:
        member = pd.Series(dtype=object)
        pl_name = None
        hostname = None
    member_id = str(pl_name) if pd.notna(pl_name) else str(row_index)
    return {
        "config_id": config_id,
        "node_id": str(node_id),
        "member_id": member_id,
        "pl_name": pl_name,
        "hostname": hostname,
        "row_index": int(row_index),
        "cube_id": cube_id,
        "cluster_id": cluster_id,
    }


def membership_from_graph_payload(
    config_id: str,
    graph_payload: dict[str, Any],
    physical_df: pd.DataFrame,
) -> pd.DataFrame:
    graph = graph_payload.get("graph", graph_payload)
    nodes = graph.get("nodes", {})
    sample_lookup = graph.get("sample_id_lookup") or list(range(len(physical_df)))
    rows: list[dict[str, Any]] = []
    for node_id, members in nodes.items():
        for member_index in members:
            mapped_index = int(sample_lookup[int(member_index)]) if int(member_index) < len(sample_lookup) else int(member_index)
            rows.append(_membership_row(config_id=config_id, node_id=str(node_id), row_index=mapped_index, physical_df=physical_df))
    return pd.DataFrame(rows)


def membership_from_node_table(
    config_id: str,
    node_table: pd.DataFrame,
    physical_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in node_table.itertuples(index=False):
        member_indices = parse_member_indices(getattr(row, "member_indices", None))
        for row_index in member_indices:
            rows.append(_membership_row(config_id=config_id, node_id=str(getattr(row, "node_id")), row_index=row_index, physical_df=physical_df))
    return pd.DataFrame(rows)
