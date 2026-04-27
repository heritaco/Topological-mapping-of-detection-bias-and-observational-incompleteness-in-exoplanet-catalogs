from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MAPPER_OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "mapper"
CONFIG_ID_RE = re.compile(
    r"^(?P<feature_space>.+)_(?P<lens>pca2|domain|density)_cubes(?P<n_cubes>\d+)_overlap(?P<overlap_slug>[0-9p]+)$"
)
NODE_SIZE_COLUMNS = ("n_members", "n_points", "node_size", "size")
MEMBER_COLUMNS = ("member_indices", "sample_indices")
IMPUTATION_COLUMNS = (
    "mean_imputation_fraction",
    "mean_node_imputation_fraction",
    "imputed_missing_fraction_mean",
)


@dataclass(frozen=True)
class ReconciliationOutputs:
    manifest: Path
    warnings: Path
    metrics_all_existing: Path
    lens_sensitivity_all_existing: Path
    space_comparison_all_existing: Path


def _resolve_outputs_dir(outputs_dir: str | Path | None = None) -> Path:
    path = Path(outputs_dir) if outputs_dir is not None else DEFAULT_MAPPER_OUTPUTS_DIR
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _read_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001 - manifest should record all parse failures.
        return None, f"{type(exc).__name__}: {exc}"


def _mtime(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds")


def _display_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def _parse_overlap(slug: str | None) -> float | None:
    if not slug:
        return None
    try:
        return float(slug.replace("p", "."))
    except ValueError:
        return None


def _parse_config_id(config_id: str) -> dict[str, Any]:
    match = CONFIG_ID_RE.match(config_id)
    if not match:
        return {
            "feature_space": "",
            "lens": "",
            "n_cubes": np.nan,
            "overlap": np.nan,
            "config_parse_error": f"Could not parse config_id: {config_id}",
        }
    return {
        "feature_space": match.group("feature_space"),
        "lens": match.group("lens"),
        "n_cubes": int(match.group("n_cubes")),
        "overlap": _parse_overlap(match.group("overlap_slug")),
        "config_parse_error": "",
    }


def _config_id_from_graph_path(path: Path) -> str:
    name = path.stem
    return name.removeprefix("graph_")


def _config_from_payload(config_id: str, payload: dict[str, Any] | None) -> dict[str, Any]:
    parsed = _parse_config_id(config_id)
    if not payload:
        return parsed

    config = payload.get("config", {}) if isinstance(payload.get("config"), dict) else {}
    metadata = payload.get("mapper_metadata", {}) if isinstance(payload.get("mapper_metadata"), dict) else {}
    cover = metadata.get("cover", {}) if isinstance(metadata.get("cover"), dict) else {}

    parsed["feature_space"] = str(config.get("space") or metadata.get("space") or parsed["feature_space"])
    parsed["lens"] = str(config.get("lens") or metadata.get("lens") or parsed["lens"])
    if config.get("n_cubes") is not None:
        parsed["n_cubes"] = int(config["n_cubes"])
    elif cover.get("n_cubes") is not None:
        parsed["n_cubes"] = int(cover["n_cubes"])
    if config.get("overlap") is not None:
        parsed["overlap"] = float(config["overlap"])
    elif cover.get("overlap") is not None:
        parsed["overlap"] = float(cover["overlap"])
    return parsed


def _graph_to_networkx(graph_payload: dict[str, Any]) -> nx.Graph:
    graph = nx.Graph()
    nodes = graph_payload.get("nodes", {})
    if not isinstance(nodes, dict):
        nodes = {}
    for node_id, members in nodes.items():
        member_list = members if isinstance(members, list) else []
        graph.add_node(str(node_id), size=len(member_list), members=member_list)

    links = graph_payload.get("links", {})
    if not isinstance(links, dict):
        return graph
    for source, targets in links.items():
        if not isinstance(targets, list):
            continue
        for target in targets:
            source_id = str(source)
            target_id = str(target)
            if source_id != target_id:
                graph.add_edge(source_id, target_id)
    return graph


def _compute_graph_metrics(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    graph_payload = payload.get("graph", payload)
    if not isinstance(graph_payload, dict):
        return {}

    nx_graph = _graph_to_networkx(graph_payload)
    n_nodes = int(nx_graph.number_of_nodes())
    n_edges = int(nx_graph.number_of_edges())
    beta_0 = int(nx.number_connected_components(nx_graph)) if n_nodes else 0
    beta_1 = int(n_edges - n_nodes + beta_0)
    degrees = [degree for _, degree in nx_graph.degree()]
    components = list(nx.connected_components(nx_graph)) if n_nodes else []
    largest_component_size = int(max((len(component) for component in components), default=0))
    node_sizes = [len(members) for members in graph_payload.get("nodes", {}).values() if isinstance(members, list)]
    unique_members = sorted({member for members in graph_payload.get("nodes", {}).values() if isinstance(members, list) for member in members})

    return {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "beta_0": beta_0,
        "beta_1": beta_1,
        "graph_density": float(nx.density(nx_graph)) if n_nodes else 0.0,
        "average_degree": float(np.mean(degrees)) if degrees else 0.0,
        "max_degree": int(max(degrees)) if degrees else 0,
        "n_connected_components": beta_0,
        "largest_component_size": largest_component_size,
        "largest_component_fraction": float(largest_component_size / n_nodes) if n_nodes else 0.0,
        "graph_node_size_mean": float(np.mean(node_sizes)) if node_sizes else np.nan,
        "graph_node_size_median": float(np.median(node_sizes)) if node_sizes else np.nan,
        "graph_node_size_min": int(min(node_sizes)) if node_sizes else np.nan,
        "graph_node_size_max": int(max(node_sizes)) if node_sizes else np.nan,
        "graph_total_unique_members": int(len(unique_members)),
    }


def _parse_members(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except Exception:
        return []


def _node_metrics_from_csv(path: Path | None) -> tuple[dict[str, Any], str]:
    if path is None or not path.exists():
        return {}, ""
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001 - record failure in output row.
        return {}, f"{type(exc).__name__}: {exc}"

    metrics: dict[str, Any] = {"node_csv_rows": int(len(frame))}
    size_col = next((column for column in NODE_SIZE_COLUMNS if column in frame.columns), None)
    if size_col:
        sizes = pd.to_numeric(frame[size_col], errors="coerce").dropna()
        metrics.update(
            {
                "node_size_mean": float(sizes.mean()) if not sizes.empty else np.nan,
                "node_size_median": float(sizes.median()) if not sizes.empty else np.nan,
                "node_size_min": int(sizes.min()) if not sizes.empty else np.nan,
                "node_size_max": int(sizes.max()) if not sizes.empty else np.nan,
            }
        )

    member_col = next((column for column in MEMBER_COLUMNS if column in frame.columns), None)
    if member_col:
        unique_members: set[str] = set()
        for value in frame[member_col]:
            for member in _parse_members(value):
                unique_members.add(str(member))
        metrics["total_unique_members"] = int(len(unique_members))

    imputation_col = next((column for column in IMPUTATION_COLUMNS if column in frame.columns), None)
    if imputation_col:
        imputation = pd.to_numeric(frame[imputation_col], errors="coerce").dropna()
        metrics["mean_node_imputation_fraction"] = float(imputation.mean()) if not imputation.empty else np.nan
        metrics["max_node_imputation_fraction"] = float(imputation.max()) if not imputation.empty else np.nan

    return metrics, ""


def _active_config_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return set()
    if "config_id" not in frame.columns:
        return set()
    return set(frame["config_id"].dropna().astype(str))


def _active_lenses(path: Path) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception:
        return set()
    if "lens" in frame.columns:
        return set(frame["lens"].dropna().astype(str))
    if "config_id" in frame.columns:
        return {_parse_config_id(config_id)["lens"] for config_id in frame["config_id"].dropna().astype(str)}
    return set()


def _active_table_summary(path: Path) -> str:
    if not path.exists():
        return f"`{_display_path(path)}` is missing."
    try:
        frame = pd.read_csv(path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        return f"`{_display_path(path)}` could not be read: {type(exc).__name__}: {exc}"
    lenses = sorted(_active_lenses(path))
    return f"`{_display_path(path)}` has {len(frame)} rows; lenses: {', '.join(lenses) if lenses else 'not detected'}."


def _collect_manifest_rows(outputs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    graphs_dir = outputs_dir / "graphs"
    nodes_dir = outputs_dir / "nodes"
    edges_dir = outputs_dir / "edges"
    config_dir = outputs_dir / "config"
    metrics_path = outputs_dir / "metrics" / "mapper_graph_metrics.csv"
    selection_path = outputs_dir / "tables" / "main_graph_selection.csv"
    active_metric_ids = _active_config_ids(metrics_path)
    selected_ids = _active_config_ids(selection_path)

    manifest_rows: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []

    for graph_path in sorted(graphs_dir.glob("graph_*.json")):
        config_id = _config_id_from_graph_path(graph_path)
        node_path = nodes_dir / f"nodes_{config_id}.csv"
        edge_path = edges_dir / f"edges_{config_id}.csv"
        config_path = config_dir / f"config_{config_id}.json"

        payload, parse_error = _read_json(graph_path)
        config = _config_from_payload(config_id, payload)
        graph_metrics = payload.get("graph_metrics", {}) if isinstance(payload, dict) and isinstance(payload.get("graph_metrics"), dict) else {}
        computed_metrics = _compute_graph_metrics(payload)
        node_metrics, node_parse_error = _node_metrics_from_csv(node_path if node_path.exists() else None)

        manifest_row = {
            "config_id": config_id,
            "feature_space": config["feature_space"],
            "lens": config["lens"],
            "n_cubes": config["n_cubes"],
            "overlap": config["overlap"],
            "graph_path": _display_path(graph_path),
            "node_csv_path": _display_path(node_path) if node_path.exists() else "",
            "edge_csv_path": _display_path(edge_path) if edge_path.exists() else "",
            "config_json_path": _display_path(config_path) if config_path.exists() else "",
            "has_graph": True,
            "has_nodes": bool(node_path.exists()),
            "has_edges": bool(edge_path.exists()),
            "has_config": bool(config_path.exists()),
            "graph_mtime": _mtime(graph_path),
            "nodes_mtime": _mtime(node_path if node_path.exists() else None),
            "edges_mtime": _mtime(edge_path if edge_path.exists() else None),
            "config_mtime": _mtime(config_path if config_path.exists() else None),
            "appears_in_mapper_graph_metrics": config_id in active_metric_ids,
            "appears_in_main_graph_selection": config_id in selected_ids,
            "graph_parse_error": parse_error,
            "node_csv_parse_error": node_parse_error,
            "config_parse_error": config.get("config_parse_error", ""),
        }
        manifest_rows.append(manifest_row)

        metrics_row: dict[str, Any] = {
            "config_id": config_id,
            "feature_space": config["feature_space"],
            "space": config["feature_space"],
            "lens": config["lens"],
            "n_cubes": config["n_cubes"],
            "overlap": config["overlap"],
            "graph_path": _display_path(graph_path),
            "node_csv_path": manifest_row["node_csv_path"],
            "edge_csv_path": manifest_row["edge_csv_path"],
            "config_json_path": manifest_row["config_json_path"],
            "graph_parse_error": parse_error,
            "node_csv_parse_error": node_parse_error,
            "appears_in_mapper_graph_metrics": config_id in active_metric_ids,
            "appears_in_main_graph_selection": config_id in selected_ids,
        }
        metric_keys = {
            "n_nodes",
            "n_edges",
            "beta_0",
            "beta_1",
            "graph_density",
            "average_degree",
            "average_clustering",
            "transitivity",
            "n_isolates",
            "largest_component_size",
            "largest_component_fraction",
            "mean_node_size",
            "median_node_size",
            "min_node_size",
            "max_node_size",
        }
        for key in sorted(metric_keys | set(computed_metrics)):
            metrics_row[key] = graph_metrics.get(key, computed_metrics.get(key, np.nan))
        for key, value in node_metrics.items():
            metrics_row[key] = value
        if "mean_node_size" not in metrics_row or pd.isna(metrics_row.get("mean_node_size")):
            metrics_row["mean_node_size"] = metrics_row.get("node_size_mean", metrics_row.get("graph_node_size_mean"))
        if "median_node_size" not in metrics_row or pd.isna(metrics_row.get("median_node_size")):
            metrics_row["median_node_size"] = metrics_row.get("node_size_median", metrics_row.get("graph_node_size_median"))
        if "min_node_size" not in metrics_row or pd.isna(metrics_row.get("min_node_size")):
            metrics_row["min_node_size"] = metrics_row.get("node_size_min", metrics_row.get("graph_node_size_min"))
        if "max_node_size" not in metrics_row or pd.isna(metrics_row.get("max_node_size")):
            metrics_row["max_node_size"] = metrics_row.get("node_size_max", metrics_row.get("graph_node_size_max"))
        metrics_rows.append(metrics_row)

    return pd.DataFrame(manifest_rows), pd.DataFrame(metrics_rows)


def _same_setting_key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].astype(str).agg("|".join, axis=1)


def build_lens_sensitivity_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    group_columns = ["feature_space", "n_cubes", "overlap"]
    frame["lens_group_key"] = _same_setting_key(frame, group_columns)
    frame["available_lenses_in_group"] = ""
    frame["n_lenses_in_group"] = 0
    for _, index in frame.groupby(group_columns, dropna=False).groups.items():
        lenses = sorted(frame.loc[index, "lens"].dropna().astype(str).unique().tolist())
        frame.loc[index, "available_lenses_in_group"] = ", ".join(lenses)
        frame.loc[index, "n_lenses_in_group"] = len(lenses)

        pca2_rows = frame.loc[index][frame.loc[index, "lens"] == "pca2"]
        if pca2_rows.empty:
            continue
        base = pca2_rows.iloc[0]
        for column in ["n_nodes", "n_edges", "beta_0", "beta_1", "mean_node_size", "mean_node_imputation_fraction"]:
            if column in frame.columns and pd.notna(base.get(column)):
                frame.loc[index, f"delta_{column}_vs_pca2"] = pd.to_numeric(frame.loc[index, column], errors="coerce") - float(base[column])
    sort_cols = ["feature_space", "n_cubes", "overlap", "lens"]
    return frame.sort_values(sort_cols, ignore_index=True)


def build_space_comparison_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    frame = metrics.copy()
    group_columns = ["lens", "n_cubes", "overlap"]
    frame["space_group_key"] = _same_setting_key(frame, group_columns)
    frame["available_spaces_in_group"] = ""
    frame["n_spaces_in_group"] = 0
    for _, index in frame.groupby(group_columns, dropna=False).groups.items():
        spaces = sorted(frame.loc[index, "feature_space"].dropna().astype(str).unique().tolist())
        frame.loc[index, "available_spaces_in_group"] = ", ".join(spaces)
        frame.loc[index, "n_spaces_in_group"] = len(spaces)
    sort_cols = ["lens", "n_cubes", "overlap", "feature_space"]
    return frame.sort_values(sort_cols, ignore_index=True)


def _format_list(values: list[str], limit: int = 40) -> str:
    values = [value for value in values if value]
    if not values:
        return "None."
    shown = values[:limit]
    suffix = "" if len(values) <= limit else f"\n\n...and {len(values) - limit} more."
    return "\n".join(f"- `{value}`" for value in shown) + suffix


def _settings_table(settings: pd.DataFrame) -> str:
    if settings.empty:
        return "No settings detected."
    rows = ["| n_cubes | overlap |", "| ---: | ---: |"]
    for row in settings.sort_values(["n_cubes", "overlap"]).to_dict(orient="records"):
        rows.append(f"| {row.get('n_cubes', '')} | {row.get('overlap', '')} |")
    return "\n".join(rows)


def build_warnings_markdown(outputs_dir: Path, manifest: pd.DataFrame, metrics: pd.DataFrame) -> str:
    tables_dir = outputs_dir / "tables"
    graph_count = len(list((outputs_dir / "graphs").glob("graph_*.json")))
    node_count = len(list((outputs_dir / "nodes").glob("nodes_*.csv")))
    edge_count = len(list((outputs_dir / "edges").glob("edges_*.csv")))
    config_count = len(list((outputs_dir / "config").glob("config_*.json")))

    metrics_path = outputs_dir / "metrics" / "mapper_graph_metrics.csv"
    space_path = tables_dir / "mapper_space_comparison.csv"
    lens_path = tables_dir / "mapper_lens_sensitivity.csv"
    selection_path = tables_dir / "main_graph_selection.csv"

    active_metric_ids = _active_config_ids(metrics_path)
    all_graph_ids = set(manifest["config_id"].astype(str)) if not manifest.empty else set()
    missing_from_active = sorted(all_graph_ids - active_metric_ids)
    extra_in_active = sorted(active_metric_ids - all_graph_ids)

    missing_nodes = sorted(manifest.loc[~manifest["has_nodes"], "config_id"].astype(str).tolist()) if not manifest.empty else []
    missing_edges = sorted(manifest.loc[~manifest["has_edges"], "config_id"].astype(str).tolist()) if not manifest.empty else []
    missing_configs = sorted(manifest.loc[~manifest["has_config"], "config_id"].astype(str).tolist()) if not manifest.empty else []
    parse_errors = manifest.loc[manifest["graph_parse_error"].astype(str).ne(""), ["config_id", "graph_parse_error"]] if not manifest.empty else pd.DataFrame()

    active_lens_sets = {
        "mapper_graph_metrics.csv": _active_lenses(metrics_path),
        "mapper_space_comparison.csv": _active_lenses(space_path),
        "mapper_lens_sensitivity.csv": _active_lenses(lens_path),
    }
    all_active_pca2_only = all(lenses == {"pca2"} for lenses in active_lens_sets.values() if lenses)
    existing_lenses = sorted(metrics["lens"].dropna().astype(str).unique().tolist()) if "lens" in metrics else []
    settings = metrics[["n_cubes", "overlap"]].drop_duplicates() if {"n_cubes", "overlap"}.issubset(metrics.columns) else pd.DataFrame()
    has_full_grid_evidence = len(settings) > 1

    stale_text = (
        "Non-pca2 artifacts are present but under-summarized by the active aggregate tables. "
        "Because the active aggregate tables are pca2-only while graph/node/edge/config files exist for other lenses, "
        "they may be stale from an earlier all-lens run or simply not represented in the latest aggregate tables. "
        "File presence alone cannot distinguish those cases."
    )
    active_layer = (
        "The active interpretation layer should remain `pca2` until the all-existing tables are reviewed and, if desired, "
        "the report is regenerated from a deliberately synchronized run. `main_graph_selection.csv` is pca2-only."
    )

    lines = [
        "# Output Consistency Warnings",
        "",
        "This report was generated by inspecting existing Mapper artifacts only. No Mapper graphs were regenerated.",
        "",
        "## File Counts",
        "",
        f"- Graph JSON files: {graph_count}",
        f"- Node CSV files: {node_count}",
        f"- Edge CSV files: {edge_count}",
        f"- Config JSON files: {config_count}",
        "",
        "## Matching Files",
        "",
        f"- Every graph has a matching node CSV: {'YES' if not missing_nodes else 'NO'}",
        f"- Every graph has a matching edge CSV: {'YES' if not missing_edges else 'NO'}",
        f"- Every graph has a matching config JSON: {'YES' if not missing_configs else 'NO'}",
        "",
        "Missing node files:",
        "",
        _format_list(missing_nodes),
        "",
        "Missing edge files:",
        "",
        _format_list(missing_edges),
        "",
        "Missing config files:",
        "",
        _format_list(missing_configs),
        "",
        "## Active Aggregate Tables",
        "",
        f"- {_active_table_summary(metrics_path)}",
        f"- {_active_table_summary(space_path)}",
        f"- {_active_table_summary(lens_path)}",
        f"- `{_display_path(selection_path)}` config IDs: {len(_active_config_ids(selection_path)) if selection_path.exists() else 0}",
        "",
        f"Active aggregate tables are pca2-only: {'YES' if all_active_pca2_only else 'NO or UNCLEAR'}",
        "",
        "Graph artifacts missing from active `mapper_graph_metrics.csv`:",
        "",
        _format_list(missing_from_active),
        "",
        "Config IDs in active `mapper_graph_metrics.csv` without graph JSON:",
        "",
        _format_list(extra_in_active),
        "",
        "## Staleness / Under-Summarization Assessment",
        "",
        stale_text,
        "",
        f"Existing lenses detected in graph artifacts: {', '.join(existing_lenses) if existing_lenses else 'none detected'}.",
        "",
        "## Grid Evidence",
        "",
        f"Unique `(n_cubes, overlap)` settings found: {len(settings)}.",
        "",
        _settings_table(settings),
        "",
        f"Evidence of a full n_cubes/overlap grid: {'YES' if has_full_grid_evidence else 'NO'}",
        "",
        "## Active Interpretation Layer",
        "",
        active_layer,
        "",
        "## Parse Errors",
        "",
    ]
    if parse_errors.empty:
        lines.append("None.")
    else:
        for row in parse_errors.to_dict(orient="records"):
            lines.append(f"- `{row['config_id']}`: {row['graph_parse_error']}")
    lines.append("")
    return "\n".join(lines)


def reconcile_mapper_outputs(outputs_dir: str | Path | None = None) -> ReconciliationOutputs:
    root = _resolve_outputs_dir(outputs_dir)
    tables_dir = root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    manifest, metrics = _collect_manifest_rows(root)
    lens_sensitivity = build_lens_sensitivity_table(metrics)
    space_comparison = build_space_comparison_table(metrics)
    warnings = build_warnings_markdown(root, manifest, metrics)

    manifest_path = tables_dir / "output_manifest.csv"
    warnings_path = tables_dir / "output_consistency_warnings.md"
    metrics_path = tables_dir / "mapper_graph_metrics_all_existing.csv"
    lens_path = tables_dir / "mapper_lens_sensitivity_all_existing.csv"
    space_path = tables_dir / "mapper_space_comparison_all_existing.csv"

    manifest.to_csv(manifest_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    lens_sensitivity.to_csv(lens_path, index=False)
    space_comparison.to_csv(space_path, index=False)
    warnings_path.write_text(warnings, encoding="utf-8")

    return ReconciliationOutputs(
        manifest=manifest_path,
        warnings=warnings_path,
        metrics_all_existing=metrics_path,
        lens_sensitivity_all_existing=lens_path,
        space_comparison_all_existing=space_path,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconcile existing Mapper output artifacts without rerunning Mapper.")
    parser.add_argument("--outputs-dir", default="outputs/mapper", help="Mapper outputs directory. Default: outputs/mapper.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    outputs = reconcile_mapper_outputs(args.outputs_dir)
    print("Mapper output reconciliation complete.")
    for name, path in outputs.__dict__.items():
        print(f"{name}: {_display_path(path)}")


if __name__ == "__main__":
    main()
