from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from .paths import first_existing

def normalize_node_column(df: pd.DataFrame) -> pd.DataFrame:
    if "node_id" in df.columns:
        return df
    for c in ["node", "mapper_node", "cluster_id", "cluster", "id"]:
        if c in df.columns:
            return df.rename(columns={c: "node_id"})
    raise KeyError(f"Could not identify node_id column in columns={list(df.columns)}")

def normalize_planet_column(df: pd.DataFrame) -> pd.DataFrame:
    if "pl_name" in df.columns:
        return df
    for c in ["planet", "planet_name", "member", "member_id", "name"]:
        if c in df.columns:
            return df.rename(columns={c: "pl_name"})
    raise KeyError(f"Could not identify pl_name column in columns={list(df.columns)}")

def normalize_edge_columns(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        ("source", "target"),
        ("node_u", "node_v"),
        ("u", "v"),
        ("node1", "node2"),
        ("from", "to"),
    ]
    cols = set(df.columns)
    for a, b in candidates:
        if a in cols and b in cols:
            return df.rename(columns={a: "source", b: "target"})
    raise KeyError(f"Could not identify edge columns in columns={list(df.columns)}")

def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)

def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def write_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

def load_required_tables(root: Path, cfg_inputs: dict) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    resolved = {}
    resolved["node_shadow_metrics"] = first_existing(
        root, cfg_inputs.get("node_shadow_metrics"),
        cfg_inputs.get("node_shadow_metrics_patterns", []),
        "node shadow metrics"
    )
    resolved["mapper_membership"] = first_existing(
        root, cfg_inputs.get("mapper_membership"),
        cfg_inputs.get("mapper_membership_patterns", []),
        "Mapper node-planet membership"
    )
    resolved["mapper_edges"] = first_existing(
        root, cfg_inputs.get("mapper_edges"),
        cfg_inputs.get("mapper_edges_patterns", []),
        "Mapper edges"
    )
    resolved["imputed_catalog"] = first_existing(
        root, cfg_inputs.get("imputed_catalog"),
        cfg_inputs.get("imputed_catalog_patterns", []),
        "imputed catalog"
    )
    tables = {k: pd.read_csv(v) for k, v in resolved.items()}
    return tables, {k: str(v) for k, v in resolved.items()}

def filter_config(df: pd.DataFrame, config_id: str) -> pd.DataFrame:
    if "config_id" not in df.columns:
        return df
    sub = df[df["config_id"].astype(str) == config_id].copy()
    return sub if not sub.empty else df.copy()
