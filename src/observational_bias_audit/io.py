from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .membership import membership_from_graph_payload, membership_from_node_table
from .paths import OUTPUTS_DIR, PROJECT_ROOT, resolve_repo_path


PHYSICAL_CANDIDATES: dict[str, list[str]] = {
    "iterative": [
        "outputs/imputation/data/PSCompPars_imputed_iterative.csv",
        "outputs/imputation/PSCompPars_imputed_iterative.csv",
        "reports/imputation/PSCompPars_imputed_iterative.csv",
        "reports/imputation/PSCompPars_imputed_knn.csv",
    ],
    "knn": [
        "outputs/imputation/data/PSCompPars_imputed_knn.csv",
        "outputs/imputation/PSCompPars_imputed_knn.csv",
        "reports/imputation/PSCompPars_imputed_knn.csv",
    ],
    "median": [
        "outputs/imputation/data/PSCompPars_imputed_median.csv",
        "outputs/imputation/PSCompPars_imputed_median.csv",
        "reports/imputation/PSCompPars_imputed_median.csv",
    ],
    "complete_case": [
        "outputs/imputation/data/PSCompPars_imputed_iterative.csv",
        "outputs/imputation/PSCompPars_imputed_iterative.csv",
        "reports/imputation/PSCompPars_imputed_iterative.csv",
    ],
    "raw": [],
}

OBSERVATIONAL_COLUMN_ALIASES = {
    "pl_name": ["pl_name", "planet_name", "pl_name_clean"],
    "discoverymethod": ["discoverymethod", "discovery_method", "disc_method"],
    "disc_year": ["disc_year", "discovery_year", "year_discovered"],
    "disc_facility": ["disc_facility", "discovery_facility", "facility_discovered"],
    "disc_telescope": ["disc_telescope", "discovery_telescope"],
    "disc_instrument": ["disc_instrument", "discovery_instrument"],
}


def _latest_raw_catalog() -> Path:
    candidates = sorted((PROJECT_ROOT / "data").glob("PSCompPars_*.csv"))
    if not candidates:
        raise FileNotFoundError("No se encontro ningun data/PSCompPars_*.csv como fallback del catalogo.")
    return candidates[-1]


def _first_existing(paths: list[str]) -> Path | None:
    for path_like in paths:
        path = resolve_repo_path(path_like, PROJECT_ROOT / path_like)
        if path.exists():
            return path
    return None


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, comment="#", low_memory=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def resolve_physical_catalog_path(explicit_path: str | None, input_method: str) -> Path:
    if explicit_path:
        path = resolve_repo_path(explicit_path, PROJECT_ROOT / explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el physical CSV solicitado: {path}")
        return path
    candidate = _first_existing(PHYSICAL_CANDIDATES.get(input_method, []))
    return candidate if candidate is not None else _latest_raw_catalog()


def discover_available_config_ids(mapper_outputs_dir: Path) -> list[str]:
    config_ids: set[str] = set()
    for path in (mapper_outputs_dir / "nodes").glob("nodes_*.csv"):
        config_ids.add(path.stem.replace("nodes_", "", 1))
    for path in (mapper_outputs_dir / "graphs").glob("graph_*.json"):
        config_ids.add(path.stem.replace("graph_", "", 1))
    return sorted(config_ids)


def load_graph_payload(mapper_outputs_dir: Path, config_id: str) -> dict[str, Any]:
    path = mapper_outputs_dir / "graphs" / f"graph_{config_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Falta el grafo JSON para {config_id}: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_node_table(mapper_outputs_dir: Path, config_id: str) -> pd.DataFrame:
    path = mapper_outputs_dir / "nodes" / f"nodes_{config_id}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Falta la tabla de nodos para {config_id}: {path}")
    return read_table(path)


def load_edge_table(mapper_outputs_dir: Path, config_id: str) -> pd.DataFrame:
    path = mapper_outputs_dir / "edges" / f"edges_{config_id}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["config_id", "source", "target"])
    return read_table(path)


def resolve_column_name(frame: pd.DataFrame, canonical_name: str, required: bool = True) -> str | None:
    candidates = OBSERVATIONAL_COLUMN_ALIASES.get(canonical_name, [canonical_name])
    lower_map = {column.lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    if required:
        raise KeyError(
            f"No se encontro la columna requerida '{canonical_name}'. Se intentaron equivalentes: {candidates}."
        )
    return None


def standardize_observational_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    pl_name_col = resolve_column_name(frame, "pl_name", required=True)
    method_col = resolve_column_name(frame, "discoverymethod", required=True)
    year_col = resolve_column_name(frame, "disc_year", required=False)
    facility_col = resolve_column_name(frame, "disc_facility", required=False)
    telescope_col = resolve_column_name(frame, "disc_telescope", required=False)
    instrument_col = resolve_column_name(frame, "disc_instrument", required=False)

    out = frame.copy()
    out = out.rename(columns={pl_name_col: "pl_name", method_col: "discoverymethod"})
    if year_col:
        out = out.rename(columns={year_col: "disc_year"})
    else:
        out["disc_year"] = pd.NA
    if facility_col:
        out = out.rename(columns={facility_col: "disc_facility"})
    else:
        out["disc_facility"] = "Unknown"
    if telescope_col:
        out = out.rename(columns={telescope_col: "disc_telescope"})
    if instrument_col:
        out = out.rename(columns={instrument_col: "disc_instrument"})

    out["pl_name"] = out["pl_name"].astype("string")
    out["discoverymethod"] = out["discoverymethod"].astype("string").fillna("Unknown")
    out["disc_facility"] = out["disc_facility"].astype("string").fillna("Unknown")
    out["disc_year"] = pd.to_numeric(out["disc_year"], errors="coerce")
    return out


def load_physical_catalog(explicit_path: str | None, input_method: str) -> tuple[Path, pd.DataFrame]:
    path = resolve_physical_catalog_path(explicit_path=explicit_path, input_method=input_method)
    frame = read_table(path)
    return path, standardize_observational_metadata(frame)


def load_or_rebuild_membership(
    mapper_outputs_dir: Path,
    audit_metadata_dir: Path,
    config_id: str,
    physical_df: pd.DataFrame,
) -> tuple[pd.DataFrame, str]:
    direct_membership = mapper_outputs_dir / "memberships" / f"memberships_{config_id}.csv"
    if direct_membership.exists():
        return read_table(direct_membership), str(direct_membership)

    consolidated = mapper_outputs_dir / "tables" / "mapper_memberships_all.csv"
    if consolidated.exists():
        frame = read_table(consolidated)
        subset = frame[frame["config_id"].astype(str) == str(config_id)].copy()
        if not subset.empty:
            return subset.reset_index(drop=True), str(consolidated)

    graph_payload = load_graph_payload(mapper_outputs_dir, config_id)
    try:
        membership = membership_from_graph_payload(config_id=config_id, graph_payload=graph_payload, physical_df=physical_df)
        source = "graph_json"
    except Exception:
        node_table = load_node_table(mapper_outputs_dir, config_id)
        membership = membership_from_node_table(config_id=config_id, node_table=node_table, physical_df=physical_df)
        source = "nodes_csv"

    rebuilt_path = audit_metadata_dir / f"rebuilt_membership_{config_id}.csv"
    membership.to_csv(rebuilt_path, index=False)
    return membership, f"rebuilt:{source}"


def git_commit_hash() -> str | None:
    head = PROJECT_ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref_path = PROJECT_ROOT / ".git" / content.split(" ", 1)[1]
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
    return content or None


def append_log(lines: list[str], message: str) -> None:
    lines.append(message)


def save_log(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
