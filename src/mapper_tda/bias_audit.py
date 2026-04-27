from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPER_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "mapper"
METADATA_PRIORITY = [
    PROJECT_ROOT / "outputs" / "mapper" / "data" / "planet_physical_labels.csv",
    PROJECT_ROOT / "reports" / "imputation" / "PSCompPars_imputed_iterative.csv",
    PROJECT_ROOT / "reports" / "imputation" / "mapper_features_imputed_iterative.csv",
]
EPS = 1e-12


@dataclass
class GraphBiasInput:
    config_id: str
    feature_space: str
    lens: str
    n_cubes: int | None
    overlap: float | None
    graph_path: Path
    nodes: dict[str, list[int]]
    links: dict[str, list[str]]
    used_features: list[str]
    metadata: pd.DataFrame
    metadata_source: Path
    join_method: str
    join_coverage: float
    n_unique_members: int
    n_members_missing_metadata: int


def resolve_outputs_dir(outputs_dir: str | Path | None = None) -> Path:
    path = Path(outputs_dir) if outputs_dir is not None else DEFAULT_MAPPER_OUTPUTS_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def load_selected_config_ids(outputs_dir: Path) -> list[str]:
    selection_path = outputs_dir / "tables" / "main_graph_selection.csv"
    if not selection_path.exists():
        raise FileNotFoundError(f"Missing selected graph table: {selection_path}")
    selection = pd.read_csv(selection_path, low_memory=False)
    if "config_id" not in selection.columns:
        raise ValueError(f"`config_id` column is missing from {selection_path}")
    config_ids = selection["config_id"].dropna().astype(str).tolist()
    if not config_ids:
        raise ValueError(f"No selected config IDs found in {selection_path}")
    return config_ids


def load_metadata_table(paths: list[Path] | None = None) -> tuple[pd.DataFrame, Path]:
    for path in paths or METADATA_PRIORITY:
        if path.exists():
            metadata = pd.read_csv(path, low_memory=False)
            metadata["_source_row_index"] = np.arange(len(metadata), dtype=int)
            return metadata, path
    searched = "\n".join(f"- {path}" for path in paths or METADATA_PRIORITY)
    raise FileNotFoundError(f"No metadata table found. Searched:\n{searched}")


def _metadata_for_config(metadata: pd.DataFrame, config_id: str) -> tuple[pd.DataFrame, str]:
    if "config_id" in metadata.columns and config_id in set(metadata["config_id"].dropna().astype(str)):
        subset = metadata[metadata["config_id"].astype(str) == config_id].copy()
        subset["_mapper_row_index"] = np.arange(len(subset), dtype=int)
        return subset.reset_index(drop=True), "config_id_then_sample_id_lookup_row_index"

    dedupe_keys = [column for column in ["pl_name", "hostname"] if column in metadata.columns]
    if dedupe_keys:
        subset = metadata.drop_duplicates(dedupe_keys, keep="first").copy()
        subset["_mapper_row_index"] = np.arange(len(subset), dtype=int)
        return subset.reset_index(drop=True), f"sample_id_lookup_row_index_after_dedup_{'_'.join(dedupe_keys)}"

    subset = metadata.copy()
    subset["_mapper_row_index"] = np.arange(len(subset), dtype=int)
    return subset.reset_index(drop=True), "sample_id_lookup_row_index"


def _read_graph_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Could not parse graph JSON {path}: {type(exc).__name__}: {exc}") from exc


def _feature_space_from_payload(payload: dict[str, Any], config_id: str) -> tuple[str, str, int | None, float | None]:
    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    metadata = payload.get("mapper_metadata", {}) if isinstance(payload.get("mapper_metadata"), dict) else {}
    cover = metadata.get("cover", {}) if isinstance(metadata.get("cover"), dict) else {}
    feature_space = str(config.get("space") or metadata.get("space") or config_id.rsplit("_", 3)[0])
    lens = str(config.get("lens") or metadata.get("lens") or "")
    n_cubes = config.get("n_cubes", cover.get("n_cubes"))
    overlap = config.get("overlap", cover.get("overlap"))
    return (
        feature_space,
        lens,
        int(n_cubes) if n_cubes is not None and not pd.isna(n_cubes) else None,
        float(overlap) if overlap is not None and not pd.isna(overlap) else None,
    )


def _members_from_graph(graph: dict[str, Any], graph_path: Path) -> tuple[dict[str, list[int]], dict[str, list[str]]]:
    nodes_raw = graph.get("nodes")
    if not isinstance(nodes_raw, dict):
        raise ValueError(f"Node membership cannot be recovered: `graph.nodes` is missing or invalid in {graph_path}")

    sample_lookup = graph.get("sample_id_lookup")
    if sample_lookup is not None and not isinstance(sample_lookup, list):
        raise ValueError(f"Node membership cannot be recovered: `graph.sample_id_lookup` is invalid in {graph_path}")

    nodes: dict[str, list[int]] = {}
    for node_id, raw_members in nodes_raw.items():
        if not isinstance(raw_members, list):
            raise ValueError(f"Node membership cannot be recovered: node `{node_id}` in {graph_path} is not a list")
        members: list[int] = []
        for member in raw_members:
            member_index = int(member)
            if sample_lookup is not None:
                if member_index >= len(sample_lookup):
                    raise ValueError(
                        f"Node membership cannot be recovered: node `{node_id}` references sample index "
                        f"{member_index}, but sample_id_lookup has length {len(sample_lookup)} in {graph_path}"
                    )
                members.append(int(sample_lookup[member_index]))
            else:
                members.append(member_index)
        nodes[str(node_id)] = members

    links_raw = graph.get("links", {})
    links: dict[str, list[str]] = {}
    if isinstance(links_raw, dict):
        links = {str(source): [str(target) for target in targets] for source, targets in links_raw.items() if isinstance(targets, list)}
    return nodes, links


def _used_features(payload: dict[str, Any]) -> list[str]:
    metadata = payload.get("mapper_metadata", {}) if isinstance(payload.get("mapper_metadata"), dict) else {}
    features = metadata.get("used_features", [])
    return [str(feature) for feature in features] if isinstance(features, list) else []


def _validate_metadata_coverage(
    metadata: pd.DataFrame,
    nodes: dict[str, list[int]],
    graph_path: Path,
    min_coverage: float = 0.95,
) -> tuple[float, int, int]:
    member_ids = sorted({member for members in nodes.values() for member in members})
    if not member_ids:
        raise ValueError(f"Node membership cannot be recovered: no members found in {graph_path}")
    max_member = max(member_ids)
    available = set(metadata["_mapper_row_index"].astype(int).tolist())
    missing = [member for member in member_ids if member not in available]
    coverage = 1.0 - (len(missing) / len(member_ids))
    if max_member >= len(metadata) or coverage < min_coverage:
        raise ValueError(
            "Metadata join coverage is too low for "
            f"{graph_path}: matched {len(member_ids) - len(missing)} of {len(member_ids)} unique members "
            f"({coverage:.3f}). Max member id={max_member}, metadata rows={len(metadata)}."
        )
    return coverage, len(member_ids), len(missing)


def prepare_selected_graphs(outputs_dir: str | Path | None = None) -> tuple[list[GraphBiasInput], pd.DataFrame]:
    root = resolve_outputs_dir(outputs_dir)
    selected_config_ids = load_selected_config_ids(root)
    metadata, metadata_source = load_metadata_table()
    graph_inputs: list[GraphBiasInput] = []
    join_rows: list[dict[str, Any]] = []

    for config_id in selected_config_ids:
        graph_path = root / "graphs" / f"graph_{config_id}.json"
        if not graph_path.exists():
            raise FileNotFoundError(f"Selected graph JSON is missing: {graph_path}")
        payload = _read_graph_json(graph_path)
        graph = payload.get("graph")
        if not isinstance(graph, dict):
            raise ValueError(f"Node membership cannot be recovered: `graph` object is missing in {graph_path}")

        nodes, links = _members_from_graph(graph, graph_path)
        config_metadata, join_method = _metadata_for_config(metadata, config_id)
        coverage, n_unique, n_missing = _validate_metadata_coverage(config_metadata, nodes, graph_path)
        feature_space, lens, n_cubes, overlap = _feature_space_from_payload(payload, config_id)
        graph_inputs.append(
            GraphBiasInput(
                config_id=config_id,
                feature_space=feature_space,
                lens=lens,
                n_cubes=n_cubes,
                overlap=overlap,
                graph_path=graph_path,
                nodes=nodes,
                links=links,
                used_features=_used_features(payload),
                metadata=config_metadata,
                metadata_source=metadata_source,
                join_method=join_method,
                join_coverage=coverage,
                n_unique_members=n_unique,
                n_members_missing_metadata=n_missing,
            )
        )
        join_rows.append(
            {
                "config_id": config_id,
                "metadata_source": display_path(metadata_source),
                "join_method": join_method,
                "join_coverage": coverage,
                "n_unique_members": n_unique,
                "n_members_missing_metadata": n_missing,
            }
        )
    return graph_inputs, pd.DataFrame(join_rows)


def label_distribution(values: pd.Series) -> pd.Series:
    valid = values.dropna().astype(str)
    valid = valid[valid.str.len() > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    return valid.value_counts(normalize=True).sort_index()


def entropy_from_distribution(distribution: pd.Series) -> float:
    if distribution.empty:
        return np.nan
    probs = distribution.to_numpy(dtype=float)
    probs = probs[probs > 0]
    return float(-(probs * np.log2(probs)).sum())


def dominant_label(values: pd.Series) -> tuple[str, float, float]:
    distribution = label_distribution(values)
    if distribution.empty:
        return "", np.nan, np.nan
    top_label = str(distribution.sort_values(ascending=False).index[0])
    return top_label, float(distribution[top_label]), entropy_from_distribution(distribution)


def divergence_vs_global(local: pd.Series, global_distribution: pd.Series) -> tuple[float, float]:
    local_distribution = label_distribution(local)
    if local_distribution.empty or global_distribution.empty:
        return np.nan, np.nan
    labels = sorted(set(local_distribution.index.astype(str)) | set(global_distribution.index.astype(str)))
    p = np.array([float(local_distribution.get(label, 0.0)) for label in labels], dtype=float)
    q = np.array([float(global_distribution.get(label, 0.0)) for label in labels], dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    q_safe = np.clip(q, EPS, None)
    q_safe = q_safe / q_safe.sum()
    p_safe = np.clip(p, EPS, None)
    p_safe = p_safe / p_safe.sum()
    kl = float(np.sum(p_safe * np.log2(p_safe / q_safe)))
    midpoint = 0.5 * (p_safe + q_safe)
    js = 0.5 * np.sum(p_safe * np.log2(p_safe / midpoint)) + 0.5 * np.sum(q_safe * np.log2(q_safe / midpoint))
    return float(js), kl


def row_imputation_fraction(metadata: pd.DataFrame, used_features: list[str]) -> pd.Series:
    feature_cols = [f"{feature}_was_imputed" for feature in used_features if f"{feature}_was_imputed" in metadata.columns]
    if not feature_cols:
        feature_cols = [column for column in metadata.columns if column.endswith("_was_imputed")]
    if not feature_cols:
        return pd.Series(np.nan, index=metadata.index, dtype=float)
    return metadata.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).mean(axis=1)


def _member_frame(graph_input: GraphBiasInput, members: list[int]) -> pd.DataFrame:
    frame = graph_input.metadata
    subset = frame[frame["_mapper_row_index"].isin(members)].copy()
    subset["_row_imputation_fraction"] = row_imputation_fraction(subset, graph_input.used_features)
    return subset


def _class_summary(frame: pd.DataFrame, column: str) -> tuple[str, float]:
    if column not in frame.columns:
        return "", np.nan
    label, fraction, _ = dominant_label(frame[column])
    return label, fraction


def _year_summary(frame: pd.DataFrame) -> tuple[float, float]:
    if "disc_year" not in frame.columns:
        return np.nan, np.nan
    years = pd.to_numeric(frame["disc_year"], errors="coerce").dropna()
    if years.empty:
        return np.nan, np.nan
    q75 = years.quantile(0.75)
    q25 = years.quantile(0.25)
    return float(years.median()), float(q75 - q25)


def _bias_summary_row(graph_input: GraphBiasInput, level: str, entity_id: str, members: list[int], component_node_count: int | None = None) -> dict[str, Any]:
    frame = _member_frame(graph_input, sorted(set(members)))
    if frame.empty:
        raise ValueError(f"No metadata rows matched {level} `{entity_id}` in {graph_input.config_id}")

    global_method_distribution = label_distribution(graph_input.metadata["discoverymethod"]) if "discoverymethod" in graph_input.metadata.columns else pd.Series(dtype=float)
    method_label, method_fraction, method_entropy = (
        dominant_label(frame["discoverymethod"]) if "discoverymethod" in frame.columns else ("", np.nan, np.nan)
    )
    js, kl = (
        divergence_vs_global(frame["discoverymethod"], global_method_distribution)
        if "discoverymethod" in frame.columns
        else (np.nan, np.nan)
    )
    facility_label, facility_fraction, facility_entropy = (
        dominant_label(frame["disc_facility"]) if "disc_facility" in frame.columns else ("", np.nan, np.nan)
    )
    disc_year_median, disc_year_iqr = _year_summary(frame)
    radius_label, radius_fraction = _class_summary(frame, "radius_class")
    orbit_label, _ = _class_summary(frame, "orbit_class")
    thermal_label, _ = _class_summary(frame, "thermal_class")
    imputation = pd.to_numeric(frame["_row_imputation_fraction"], errors="coerce").dropna()

    row = {
        "config_id": graph_input.config_id,
        "feature_space": graph_input.feature_space,
        "lens": graph_input.lens,
        f"{level}_id": entity_id,
        "n_members": int(len(set(members))),
        "dominant_discoverymethod": method_label,
        "dominant_discoverymethod_fraction": method_fraction,
        "discoverymethod_entropy": method_entropy,
        "discoverymethod_js_divergence_vs_global": js,
        "discoverymethod_kl_divergence_vs_global": kl,
        "dominant_disc_facility": facility_label,
        "dominant_disc_facility_fraction": facility_fraction,
        "disc_facility_entropy": facility_entropy,
        "disc_year_median": disc_year_median,
        "disc_year_iqr": disc_year_iqr,
        "radius_class_dominant": radius_label,
        "radius_class_purity": radius_fraction,
        "orbit_class_dominant": orbit_label,
        "thermal_class_dominant": thermal_label,
        "mean_imputation_fraction": float(imputation.mean()) if not imputation.empty else np.nan,
        "max_imputation_fraction": float(imputation.max()) if not imputation.empty else np.nan,
    }
    if component_node_count is not None:
        row["n_component_nodes"] = int(component_node_count)
    return row


def build_node_discovery_bias(graph_inputs: list[GraphBiasInput]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for graph_input in graph_inputs:
        for node_id, members in sorted(graph_input.nodes.items()):
            rows.append(_bias_summary_row(graph_input, "node", node_id, members))
    return pd.DataFrame(rows)


def _component_members(graph_input: GraphBiasInput) -> list[tuple[int, list[str], list[int]]]:
    graph = nx.Graph()
    for node_id in graph_input.nodes:
        graph.add_node(node_id)
    for source, targets in graph_input.links.items():
        for target in targets:
            if source != target:
                graph.add_edge(str(source), str(target))

    components: list[tuple[int, list[str], list[int]]] = []
    for component_id, component_nodes in enumerate(nx.connected_components(graph)):
        node_ids = sorted(str(node_id) for node_id in component_nodes)
        members = sorted({member for node_id in node_ids for member in graph_input.nodes.get(node_id, [])})
        components.append((component_id, node_ids, members))
    return components


def build_component_discovery_bias(graph_inputs: list[GraphBiasInput]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for graph_input in graph_inputs:
        for component_id, node_ids, members in _component_members(graph_input):
            row = _bias_summary_row(graph_input, "component", str(component_id), members, component_node_count=len(node_ids))
            row["component_node_ids"] = json.dumps(node_ids, ensure_ascii=False)
            rows.append(row)
    return pd.DataFrame(rows)


def write_bias_audit_tables(outputs_dir: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[GraphBiasInput], pd.DataFrame]:
    root = resolve_outputs_dir(outputs_dir)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    (root / "bias_audit").mkdir(parents=True, exist_ok=True)
    graph_inputs, join_report = prepare_selected_graphs(root)
    node_bias = build_node_discovery_bias(graph_inputs)
    component_bias = build_component_discovery_bias(graph_inputs)
    node_bias.to_csv(root / "tables" / "node_discovery_bias.csv", index=False)
    component_bias.to_csv(root / "tables" / "component_discovery_bias.csv", index=False)
    return node_bias, component_bias, graph_inputs, join_report
