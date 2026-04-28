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


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, comment="#", low_memory=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_graph_payload(mapper_outputs_dir: Path, config_id: str) -> dict[str, Any]:
    path = mapper_outputs_dir / "graphs" / f"graph_{config_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta el grafo Mapper para {config_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_node_table(mapper_outputs_dir: Path, config_id: str) -> pd.DataFrame:
    path = mapper_outputs_dir / "nodes" / f"nodes_{config_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta la tabla de nodos Mapper para {config_id}: {path}")
    return read_table(path)


def load_edge_table(mapper_outputs_dir: Path, config_id: str, warnings: list[str]) -> pd.DataFrame:
    path = mapper_outputs_dir / "edges" / f"edges_{config_id}.csv"
    if not path.exists():
        warnings.append(f"WARNING: faltan aristas para {config_id}; se marcaran nodos sin vecinos locales.")
        return pd.DataFrame(columns=["config_id", "source", "target"])
    frame = read_table(path)
    if not {"source", "target"}.issubset(frame.columns):
        warnings.append(f"WARNING: la tabla de aristas de {config_id} no tiene columnas source/target.")
        return pd.DataFrame(columns=["config_id", "source", "target"])
    return frame


def load_membership_with_catalog(
    mapper_outputs_dir: Path,
    audit_outputs_dir: Path,
    shadow_metadata_dir: Path,
    config_id: str,
    physical_df: pd.DataFrame,
    warnings: list[str],
) -> tuple[pd.DataFrame, str]:
    audit_membership = audit_outputs_dir / "metadata" / f"membership_with_observational_metadata_{config_id}.csv"
    source = str(audit_membership)
    if audit_membership.exists():
        membership = read_table(audit_membership)
    else:
        membership, source = load_or_rebuild_membership(
            mapper_outputs_dir=mapper_outputs_dir,
            audit_metadata_dir=shadow_metadata_dir,
            config_id=config_id,
            physical_df=physical_df,
        )
        source = f"rebuilt:{source}"

    if membership.empty:
        raise RuntimeError(f"No fue posible construir membresia nodo-planeta para {config_id}.")
    if "row_index" not in membership.columns:
        raise RuntimeError(f"La membresia de {config_id} no contiene row_index, necesario para unir catalogo fisico.")
    membership = membership.drop(columns=[column for column in membership.columns if column.endswith("_catalog")], errors="ignore")

    catalog = physical_df.reset_index(drop=False).rename(columns={"index": "row_index"}).copy()
    keep_cols = ["row_index"]
    for column in [
        "pl_name",
        "hostname",
        "discoverymethod",
        "disc_year",
        "disc_facility",
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_orbper",
        "pl_orbsmax",
        "pl_insol",
        "pl_eqt",
    ]:
        if column in catalog.columns:
            keep_cols.append(column)
    keep_cols.extend([column for column in catalog.columns if column.endswith("_was_imputed")])
    keep_cols.extend([column for column in catalog.columns if column.endswith("_was_physically_derived")])
    keep_cols = list(dict.fromkeys(keep_cols))

    joined = membership.merge(catalog[keep_cols], on="row_index", how="left", suffixes=("", "_catalog"))
    for column in ["pl_name", "hostname", "discoverymethod", "disc_year", "disc_facility"]:
        catalog_col = f"{column}_catalog"
        if catalog_col in joined.columns:
            if column in joined.columns:
                joined[column] = joined[column].where(joined[column].notna(), joined[catalog_col])
                joined = joined.drop(columns=[catalog_col])
            else:
                joined = joined.rename(columns={catalog_col: column})
    if "discoverymethod" not in joined.columns:
        raise RuntimeError(f"Falta discoverymethod en la membresia/catalogo de {config_id}.")
    joined["discoverymethod"] = joined["discoverymethod"].astype("string").fillna("Unknown")
    if joined["discoverymethod"].isna().all():
        raise RuntimeError(f"discoverymethod quedo vacio para {config_id}.")
    if "disc_facility" not in joined.columns:
        joined["disc_facility"] = "Unknown"
    if "disc_year" not in joined.columns:
        joined["disc_year"] = pd.NA
    joined["disc_year"] = pd.to_numeric(joined["disc_year"], errors="coerce")
    joined["disc_facility"] = joined["disc_facility"].astype("string").fillna("Unknown")
    if not any(column.endswith("_was_imputed") for column in joined.columns):
        warnings.append(f"WARNING: no se encontraron trazas de imputacion para {config_id}; se usara el factor disponible en nodos si existe.")
    out_path = shadow_metadata_dir / f"membership_with_shadow_inputs_{config_id}.csv"
    joined.to_csv(out_path, index=False)
    return joined, source


def save_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
