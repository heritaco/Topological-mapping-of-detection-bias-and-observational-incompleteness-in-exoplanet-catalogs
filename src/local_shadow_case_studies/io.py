from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..observational_bias_audit.io import (
    discover_available_config_ids,
    git_commit_hash,
    load_or_rebuild_membership,
    load_physical_catalog,
)
from .paths import repo_relative


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, comment="#", low_memory=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def discover_input_path(candidates: list[Path]) -> Path | None:
    for path in candidates:
        if path.exists():
            return path
    return None


def load_node_table(mapper_outputs_dir: Path, config_id: str) -> tuple[Path, pd.DataFrame]:
    path = mapper_outputs_dir / "nodes" / f"nodes_{config_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta la tabla de nodos Mapper para {config_id}: {path}")
    return path, read_table(path)


def load_edge_table(mapper_outputs_dir: Path, config_id: str, warnings: list[str]) -> tuple[Path | None, pd.DataFrame]:
    path = mapper_outputs_dir / "edges" / f"edges_{config_id}.csv"
    if not path.exists():
        warnings.append(f"WARNING: faltan aristas para {config_id}; se marcaran nodos sin vecinos.")
        return None, pd.DataFrame(columns=["config_id", "source", "target"])
    frame = read_table(path)
    if not {"source", "target"}.issubset(frame.columns):
        warnings.append(f"WARNING: la tabla de aristas de {config_id} no contiene source/target.")
        return path, pd.DataFrame(columns=["config_id", "source", "target"])
    return path, frame


def _membership_join_columns(catalog: pd.DataFrame) -> list[str]:
    keep_cols = ["row_index"]
    for column in [
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "disc_facility",
        "st_mass",
        "pl_bmasse",
        "pl_orbper",
        "pl_orbsmax",
    ]:
        if column in catalog.columns:
            keep_cols.append(column)
    keep_cols.extend([column for column in catalog.columns if column.endswith("_was_imputed")])
    keep_cols.extend([column for column in catalog.columns if column.endswith("_was_physically_derived")])
    return list(dict.fromkeys(keep_cols))


def load_membership_with_catalog(
    mapper_outputs_dir: Path,
    audit_outputs_dir: Path,
    local_metadata_dir: Path,
    config_id: str,
    physical_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[Path | None, pd.DataFrame]:
    audit_membership = audit_outputs_dir / "metadata" / f"membership_with_observational_metadata_{config_id}.csv"
    if audit_membership.exists():
        membership = read_table(audit_membership)
        source_path: Path | None = audit_membership
    else:
        membership, source = load_or_rebuild_membership(
            mapper_outputs_dir=mapper_outputs_dir,
            audit_metadata_dir=local_metadata_dir,
            config_id=config_id,
            physical_df=physical_df,
        )
        source_path = Path(source.replace("rebuilt:", "")) if source.startswith("rebuilt:") else None

    if membership.empty:
        raise RuntimeError(f"No fue posible construir membresia nodo-planeta para {config_id}.")
    if "row_index" not in membership.columns:
        raise RuntimeError(f"La membresia de {config_id} no contiene row_index.")
    membership = membership.drop(columns=[column for column in membership.columns if column.endswith("_catalog")], errors="ignore")

    catalog = physical_df.reset_index(drop=False).rename(columns={"index": "row_index"}).copy()
    joined = membership.merge(catalog[_membership_join_columns(catalog)], on="row_index", how="left", suffixes=("", "_catalog"))
    for column in ["pl_name", "hostname", "discoverymethod", "disc_year", "disc_facility", "st_mass", "pl_bmasse", "pl_orbper", "pl_orbsmax"]:
        catalog_col = f"{column}_catalog"
        if catalog_col in joined.columns:
            if column in joined.columns:
                joined[column] = joined[column].where(joined[column].notna(), joined[catalog_col])
                joined = joined.drop(columns=[catalog_col])
            else:
                joined = joined.rename(columns={catalog_col: column})
    joined["discoverymethod"] = joined.get("discoverymethod", pd.Series(index=joined.index, dtype="string")).astype("string").fillna("Unknown")
    joined["disc_facility"] = joined.get("disc_facility", pd.Series(index=joined.index, dtype="string")).astype("string").fillna("Unknown")
    joined["disc_year"] = pd.to_numeric(joined.get("disc_year"), errors="coerce")
    out_path = local_metadata_dir / f"membership_with_local_shadow_inputs_{config_id}.csv"
    joined.to_csv(out_path, index=False)
    return source_path, joined


def load_required_case_inputs(
    mapper_outputs_dir: Path,
    audit_outputs_dir: Path,
    shadow_outputs_dir: Path,
    config_id: str,
    physical_csv_path: str | None,
    local_metadata_dir: Path,
    warnings: list[str],
) -> dict[str, Any]:
    physical_path, physical_df = load_physical_catalog(physical_csv_path, input_method="iterative")
    node_path, node_table = load_node_table(mapper_outputs_dir, config_id)
    edge_path, edge_table = load_edge_table(mapper_outputs_dir, config_id, warnings)
    membership_path, membership = load_membership_with_catalog(
        mapper_outputs_dir=mapper_outputs_dir,
        audit_outputs_dir=audit_outputs_dir,
        local_metadata_dir=local_metadata_dir,
        config_id=config_id,
        physical_df=physical_df,
        warnings=warnings,
    )

    shadow_metrics_path = discover_input_path(
        [
            shadow_outputs_dir / "tables" / "node_observational_shadow_metrics.csv",
        ]
    )
    top_candidates_path = discover_input_path(
        [
            shadow_outputs_dir / "tables" / "top_shadow_candidates.csv",
        ]
    )
    method_bias_path = discover_input_path(
        [
            audit_outputs_dir / "tables" / "node_method_bias_metrics.csv",
        ]
    )
    method_fraction_path = discover_input_path(
        [
            audit_outputs_dir / "tables" / "node_method_fraction_matrix.csv",
        ]
    )
    missing = [
        ("shadow_metrics", shadow_metrics_path),
        ("top_candidates", top_candidates_path),
        ("method_bias", method_bias_path),
        ("method_fraction", method_fraction_path),
    ]
    unresolved = [name for name, path in missing if path is None]
    if unresolved:
        raise FileNotFoundError(f"Faltan tablas requeridas para el caso local: {', '.join(unresolved)}")

    shadow_metrics = read_table(shadow_metrics_path)
    top_candidates = read_table(top_candidates_path)
    method_bias = read_table(method_bias_path)
    method_fraction = read_table(method_fraction_path)

    return {
        "physical_path": physical_path,
        "physical_df": physical_df,
        "node_path": node_path,
        "node_table": node_table,
        "edge_path": edge_path,
        "edge_table": edge_table,
        "membership_path": membership_path,
        "membership": membership,
        "shadow_metrics_path": shadow_metrics_path,
        "shadow_metrics": shadow_metrics,
        "top_candidates_path": top_candidates_path,
        "top_candidates": top_candidates,
        "method_bias_path": method_bias_path,
        "method_bias": method_bias,
        "method_fraction_path": method_fraction_path,
        "method_fraction": method_fraction,
    }


def manifest_input_paths(payload: dict[str, Any]) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, value in payload.items():
        if key.endswith("_path") and isinstance(value, Path):
            paths[key] = repo_relative(value)
        elif key.endswith("_path") and value is None:
            paths[key] = "missing"
    return paths
