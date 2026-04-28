from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ..observational_bias_audit.io import load_or_rebuild_membership, load_physical_catalog
from .paths import PROJECT_ROOT, repo_relative, resolve_repo_path


LEGACY_DATASET_FALLBACKS = [
    "reports/imputation/PSCompPars_imputed_iterative.csv",
    "reports/imputation/PSCompPars_imputed_knn.csv",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path, comment="#", low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_log(lines: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def git_commit_hash() -> str | None:
    head = PROJECT_ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref_path = PROJECT_ROOT / ".git" / content.split(" ", 1)[1]
        return ref_path.read_text(encoding="utf-8").strip() if ref_path.exists() else None
    return content or None


def _first_existing(root: Path, candidates: list[str], label: str, warnings: list[str], allow_legacy_warning: bool = False) -> tuple[Path, list[str]]:
    attempted: list[str] = []
    for candidate in candidates:
        path = resolve_repo_path(candidate, root / candidate)
        attempted.append(repo_relative(path))
        if path.exists():
            if allow_legacy_warning and any(candidate == legacy for legacy in LEGACY_DATASET_FALLBACKS):
                warnings.append(f"WARNING: {label} usa fallback legacy {repo_relative(path)}.")
            return path, attempted
    raise FileNotFoundError(f"No se encontro {label}. Rutas intentadas: {attempted}")


def normalize_node_column(frame: pd.DataFrame) -> pd.DataFrame:
    if "node_id" in frame.columns:
        return frame
    for candidate in ["node", "cluster_id", "mapper_node", "id"]:
        if candidate in frame.columns:
            return frame.rename(columns={candidate: "node_id"})
    raise KeyError(f"No se pudo identificar node_id en columnas: {list(frame.columns)}")


def normalize_planet_column(frame: pd.DataFrame, preferred: str = "pl_name") -> pd.DataFrame:
    if preferred in frame.columns:
        return frame
    for candidate in ["pl_name", "planet_name", "member_id", "member", "planet", "name"]:
        if candidate in frame.columns:
            return frame.rename(columns={candidate: preferred})
    raise KeyError(f"No se pudo identificar {preferred} en columnas: {list(frame.columns)}")


def normalize_edge_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if {"source", "target"}.issubset(frame.columns):
        return frame
    for a, b in [("node_u", "node_v"), ("u", "v"), ("node1", "node2"), ("from", "to")]:
        if {a, b}.issubset(frame.columns):
            return frame.rename(columns={a: "source", b: "target"})
    raise KeyError(f"No se pudo identificar source/target en columnas: {list(frame.columns)}")


def _dataset_candidates(explicit: str | None) -> list[str]:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            "outputs/imputation/data/mapper_features_imputed_iterative.csv",
            "outputs/imputation/data/PSCompPars_imputed_iterative.csv",
            "data/PSCompPars.csv",
            *LEGACY_DATASET_FALLBACKS,
        ]
    )
    return candidates


def _membership_candidates(config_id: str, explicit: str | None) -> list[str]:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            f"outputs/observational_bias_audit/metadata/membership_with_observational_metadata_{config_id}.csv",
            f"outputs/observational_shadow/metadata/membership_with_shadow_inputs_{config_id}.csv",
            f"outputs/local_shadow_case_studies/metadata/membership_with_local_shadow_inputs_{config_id}.csv",
            f"outputs/mapper/memberships/memberships_{config_id}.csv",
            "outputs/mapper/tables/mapper_memberships_all.csv",
            "outputs/local_shadow_case_studies/tables/case_r3_planet_members.csv",
        ]
    )
    return candidates


def _edges_candidates(config_id: str, explicit: str | None) -> list[str]:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    candidates.extend(
        [
            f"outputs/mapper/edges/edges_{config_id}.csv",
            f"outputs/mapper/graphs/graph_{config_id}.json",
            "outputs/observational_shadow/metadata",
        ]
    )
    return candidates


def _input_method_from_graph(config_id: str) -> str:
    graph_path = PROJECT_ROOT / "outputs" / "mapper" / "graphs" / f"graph_{config_id}.json"
    if not graph_path.exists():
        return "iterative"
    payload = json.loads(graph_path.read_text(encoding="utf-8"))
    return str(payload.get("config", {}).get("input_method", "iterative"))


def discover_dataset(root: Path, explicit: str | None, warnings: list[str]) -> tuple[Path, pd.DataFrame, list[str]]:
    path, attempted = _first_existing(root, _dataset_candidates(explicit), "dataset imputado/base", warnings, allow_legacy_warning=True)
    frame = read_table(path)
    return path, frame, attempted


def discover_membership(
    root: Path,
    config_id: str,
    explicit: str | None,
    dataset: pd.DataFrame,
    metadata_dir: Path,
    warnings: list[str],
) -> tuple[Path, pd.DataFrame, list[str]]:
    attempted: list[str] = []
    for candidate in _membership_candidates(config_id, explicit):
        path = resolve_repo_path(candidate, root / candidate)
        attempted.append(repo_relative(path))
        if path.is_file():
            frame = read_table(path)
            if path.name == "mapper_memberships_all.csv" and "config_id" in frame.columns:
                frame = frame[frame["config_id"].astype(str) == config_id].copy()
            if path.name == "case_r3_planet_members.csv":
                warnings.append("WARNING: membership usa fallback parcial desde local_shadow_case_studies; puede no cubrir todos los nodos.")
            if not frame.empty:
                return path, frame, attempted
        if candidate.endswith(".json") and path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            graph_nodes = payload.get("nodes", {})
            records: list[dict[str, Any]] = []
            catalog = dataset.reset_index(drop=False).rename(columns={"index": "row_index"})
            if "pl_name" in catalog.columns:
                catalog = normalize_planet_column(catalog, "pl_name")
            for node_id, members in graph_nodes.items():
                for member in members:
                    if isinstance(member, int):
                        records.append({"config_id": config_id, "node_id": str(node_id), "row_index": int(member)})
                    else:
                        records.append({"config_id": config_id, "node_id": str(node_id), "pl_name": str(member)})
            frame = pd.DataFrame(records)
            if not frame.empty:
                rebuilt = metadata_dir / f"membership_from_graph_{config_id}.csv"
                frame.to_csv(rebuilt, index=False)
                warnings.append(f"WARNING: membership reconstruida desde graph JSON para {config_id}.")
                return rebuilt, frame, attempted
    try:
        physical_path, physical_df = load_physical_catalog(None, _input_method_from_graph(config_id))
        rebuilt, source = load_or_rebuild_membership(
            mapper_outputs_dir=root / "outputs" / "mapper",
            audit_metadata_dir=metadata_dir,
            config_id=config_id,
            physical_df=physical_df,
        )
        warnings.append(f"WARNING: membership reconstruida via load_or_rebuild_membership ({source}).")
        rebuilt_path = metadata_dir / f"rebuilt_membership_{config_id}.csv"
        rebuilt.to_csv(rebuilt_path, index=False)
        return rebuilt_path, rebuilt, attempted + [f"rebuild:{repo_relative(physical_path)}"]
    except Exception as exc:
        raise FileNotFoundError(f"No se pudo identificar membership para {config_id}. Rutas intentadas: {attempted}. Error de rebuild: {exc}") from exc


def discover_edges(root: Path, config_id: str, explicit: str | None, warnings: list[str]) -> tuple[Path, pd.DataFrame, list[str]]:
    attempted: list[str] = []
    for candidate in _edges_candidates(config_id, explicit):
        path = resolve_repo_path(candidate, root / candidate)
        attempted.append(repo_relative(path))
        if path.is_file() and path.suffix.lower() == ".csv":
            frame = normalize_edge_columns(read_table(path))
            if "config_id" in frame.columns:
                frame = frame[frame["config_id"].astype(str) == config_id].copy()
            if not frame.empty:
                return path, frame, attempted
        if path.is_file() and path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            links = payload.get("links", [])
            frame = pd.DataFrame(links)
            if not frame.empty:
                frame = normalize_edge_columns(frame)
                rebuilt_path = root / "outputs" / "topological_incompleteness_index" / "metadata" / f"edges_from_graph_{config_id}.csv"
                rebuilt_path.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(rebuilt_path, index=False)
                warnings.append(f"WARNING: edges reconstruidas desde graph JSON para {config_id}.")
                return rebuilt_path, frame, attempted
    raise FileNotFoundError(f"No se pudo identificar edges para {config_id}. Rutas intentadas: {attempted}")


def _load_csv_input(root: Path, explicit: str | None, label: str) -> tuple[Path, pd.DataFrame]:
    path = resolve_repo_path(explicit, root / str(explicit)) if explicit else None
    if path is None or not path.exists():
        raise FileNotFoundError(f"Falta input critico {label}: {explicit}")
    return path, read_table(path)


def _merge_membership_with_catalog(membership: pd.DataFrame, dataset: pd.DataFrame, planet_name_column: str, warnings: list[str]) -> pd.DataFrame:
    frame = membership.copy()
    frame = frame.drop(columns=[column for column in frame.columns if column.endswith("_catalog")], errors="ignore")
    dataset_copy = dataset.copy()
    dataset_copy = normalize_planet_column(dataset_copy, planet_name_column)
    if "row_index" not in dataset_copy.columns:
        dataset_copy = dataset_copy.reset_index(drop=False).rename(columns={"index": "row_index"})
    join_keys: list[str] = []
    if "row_index" in frame.columns and "row_index" in dataset_copy.columns:
        join_keys = ["row_index"]
    elif planet_name_column in frame.columns and planet_name_column in dataset_copy.columns:
        join_keys = [planet_name_column]
    else:
        raise KeyError("No se pudo alinear membership con dataset: falta row_index y pl_name.")
    joined = frame.merge(dataset_copy, on=join_keys, how="left", suffixes=("", "_catalog"))
    for column in [planet_name_column, "hostname", "discoverymethod", "disc_year", "disc_facility", "st_mass"]:
        catalog_column = f"{column}_catalog"
        if catalog_column in joined.columns:
            if column in joined.columns:
                joined[column] = joined[column].where(joined[column].notna(), joined[catalog_column])
                joined = joined.drop(columns=[catalog_column])
            else:
                joined = joined.rename(columns={catalog_column: column})
    if planet_name_column not in joined.columns:
        raise KeyError(f"Membership unido sin columna {planet_name_column}.")
    if "row_index" not in joined.columns:
        warnings.append("WARNING: membership sin row_index tras join; se dependera de nombres de planeta.")
    return joined


def filter_config(frame: pd.DataFrame, config_id: str) -> pd.DataFrame:
    if "config_id" not in frame.columns:
        return frame.copy()
    subset = frame[frame["config_id"].astype(str) == str(config_id)].copy()
    return subset if not subset.empty else frame.copy()


def load_pipeline_inputs(root: Path, config: Any, metadata_dir: Path, warnings: list[str]) -> dict[str, Any]:
    dataset_path, dataset, dataset_attempts = discover_dataset(root, config.inputs.dataset, warnings)
    membership_path, membership, membership_attempts = discover_membership(
        root=root,
        config_id=config.analysis.config_id,
        explicit=config.inputs.membership,
        dataset=dataset,
        metadata_dir=metadata_dir,
        warnings=warnings,
    )
    edges_path, edges, edge_attempts = discover_edges(root, config.analysis.config_id, config.inputs.edges, warnings)
    shadow_path, node_shadow_metrics = _load_csv_input(root, config.inputs.node_shadow_metrics, "node_shadow_metrics")
    top_shadow_path, top_shadow_candidates = _load_csv_input(root, config.inputs.top_shadow_candidates, "top_shadow_candidates")
    method_metrics_path, node_method_metrics = _load_csv_input(root, config.inputs.node_method_metrics, "node_method_metrics")
    fraction_matrix_path, node_method_fraction_matrix = _load_csv_input(root, config.inputs.node_method_fraction_matrix, "node_method_fraction_matrix")

    node_shadow_metrics = normalize_node_column(filter_config(node_shadow_metrics, config.analysis.config_id))
    top_shadow_candidates = normalize_node_column(filter_config(top_shadow_candidates, config.analysis.config_id))
    node_method_metrics = normalize_node_column(filter_config(node_method_metrics, config.analysis.config_id))
    node_method_fraction_matrix = normalize_node_column(filter_config(node_method_fraction_matrix, config.analysis.config_id))
    membership = normalize_node_column(normalize_planet_column(filter_config(membership, config.analysis.config_id), config.analysis.planet_name_column))
    edges = normalize_edge_columns(filter_config(edges, config.analysis.config_id))
    joined_membership = _merge_membership_with_catalog(membership, dataset, config.analysis.planet_name_column, warnings)

    return {
        "dataset": dataset,
        "membership": joined_membership,
        "edges": edges,
        "node_shadow_metrics": node_shadow_metrics,
        "top_shadow_candidates": top_shadow_candidates,
        "node_method_metrics": node_method_metrics,
        "node_method_fraction_matrix": node_method_fraction_matrix,
        "input_paths": {
            "dataset": repo_relative(dataset_path),
            "membership": repo_relative(membership_path),
            "edges": repo_relative(edges_path),
            "node_shadow_metrics": repo_relative(shadow_path),
            "top_shadow_candidates": repo_relative(top_shadow_path),
            "node_method_metrics": repo_relative(method_metrics_path),
            "node_method_fraction_matrix": repo_relative(fraction_matrix_path),
        },
        "attempted_paths": {
            "dataset": dataset_attempts,
            "membership": membership_attempts,
            "edges": edge_attempts,
        },
    }
