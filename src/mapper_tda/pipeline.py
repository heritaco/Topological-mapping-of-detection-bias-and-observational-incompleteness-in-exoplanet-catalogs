from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .cluster import estimate_dbscan_eps, make_clusterer
from .feature_sets import expand_space_selection, features_for_space, has_density_feature
from .lenses import make_lens_density, make_lens_domain, make_lens_pca2
from .metrics import build_edge_table, build_node_table, compare_mapper_graphs, compute_graph_metrics, mapper_graph_to_networkx


N_CUBES_GRID = [6, 8, 10, 12, 15]
OVERLAP_GRID = [0.20, 0.30, 0.35, 0.40, 0.50]


@dataclass
class MapperConfig:
    space: str = "joint"
    lens: str = "pca2"
    n_cubes: int = 10
    overlap: float = 0.35
    clusterer: str = "dbscan"
    min_samples: int = 4
    eps_percentile: float = 90
    k_density: int = 15
    random_state: int = 42
    input_method: str = "iterative"


def config_id(config: MapperConfig) -> str:
    overlap_slug = str(config.overlap).replace(".", "p")
    return f"{config.space}_{config.lens}_cubes{config.n_cubes}_overlap{overlap_slug}"


def _geometry_matrix(mapper_df: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    missing = [feature for feature in features if feature not in mapper_df.columns]
    if missing:
        raise ValueError(f"Faltan columnas geometricas para Mapper: {missing}")
    matrix = mapper_df.loc[:, features].apply(pd.to_numeric, errors="coerce")
    valid_mask = matrix.notna().all(axis=1)
    if not bool(valid_mask.any()):
        raise ValueError("No quedaron filas validas en la matriz geometrica para el espacio seleccionado.")
    return mapper_df.loc[valid_mask].reset_index(drop=True), matrix.loc[valid_mask].to_numpy(dtype=float)


def build_mapper_graph(
    mapper_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    config: MapperConfig,
) -> dict[str, Any]:
    try:
        import kmapper as km
    except ImportError as exc:
        raise ImportError("No se pudo importar 'kmapper'. Instala la dependencia con `pip install kmapper>=2.1`.") from exc

    used_features = features_for_space(config.space)
    mapper_used, Z = _geometry_matrix(mapper_df, used_features)
    physical_used = physical_df.iloc[mapper_used.index].reset_index(drop=True)
    mapper_used = mapper_used.reset_index(drop=True)

    if config.lens == "pca2":
        lens, lens_metadata = make_lens_pca2(Z, random_state=config.random_state)
    elif config.lens == "density":
        lens, lens_metadata = make_lens_density(Z, k_density=config.k_density, random_state=config.random_state)
    elif config.lens == "domain":
        lens, lens_metadata = make_lens_domain(physical_used, space=config.space)
    else:
        raise ValueError(f"Lens no soportado: {config.lens}")

    mapper = km.KeplerMapper()
    estimated_eps = estimate_dbscan_eps(Z, min_samples=config.min_samples, percentile=config.eps_percentile)
    clusterer = make_clusterer(
        clusterer=config.clusterer,
        Z=Z,
        min_samples=config.min_samples,
        eps_percentile=config.eps_percentile,
    )
    graph = mapper.map(
        lens,
        X=Z,
        clusterer=clusterer,
        cover=km.Cover(n_cubes=config.n_cubes, perc_overlap=config.overlap),
    )
    graph["sample_id_lookup"] = list(range(len(mapper_used)))

    nx_graph = mapper_graph_to_networkx(graph)
    graph_metrics = compute_graph_metrics(nx_graph, graph)
    node_table = build_node_table(
        graph=graph,
        nx_graph=nx_graph,
        lens=lens,
        physical_df=physical_used,
        used_features=used_features,
        config_id=config_id(config),
    )
    edge_table = build_edge_table(graph=graph, physical_df=physical_used, used_features=used_features, config_id=config_id(config))

    mapper_metadata = {
        "config_id": config_id(config),
        "space": config.space,
        "lens": config.lens,
        "n_rows_input": int(len(mapper_df)),
        "n_rows_used": int(len(mapper_used)),
        "used_features": used_features,
        "cover": {"n_cubes": int(config.n_cubes), "overlap": float(config.overlap)},
        "clusterer": {
            "name": config.clusterer,
            "min_samples": int(config.min_samples),
            "eps_percentile": float(config.eps_percentile),
            "estimated_eps": float(estimated_eps),
        },
        "lens_metadata": lens_metadata,
    }

    return {
        "config": config,
        "config_id": config_id(config),
        "graph": graph,
        "nx_graph": nx_graph,
        "lens": lens,
        "mapper_df": mapper_used,
        "physical_df": physical_used,
        "Z": Z,
        "used_features": used_features,
        "lens_metadata": lens_metadata,
        "mapper_metadata": mapper_metadata,
        "node_table": node_table,
        "edge_table": edge_table,
        "graph_metrics": graph_metrics,
    }


def expand_configs_from_cli(args) -> list[MapperConfig]:
    spaces = expand_space_selection(args.space)
    lenses = ["pca2", "density", "domain"] if args.lens == "all" else [args.lens]
    if args.fast:
        n_cubes_values = [10]
        overlap_values = [0.35]
    else:
        n_cubes_values = N_CUBES_GRID if args.grid else [args.n_cubes]
        overlap_values = OVERLAP_GRID if args.grid else [args.overlap]

    return [
        MapperConfig(
            space=space,
            lens=lens,
            n_cubes=n_cubes,
            overlap=overlap,
            clusterer=args.clusterer,
            min_samples=args.min_samples,
            eps_percentile=args.eps_percentile,
            k_density=args.k_density,
            random_state=args.random_state,
            input_method=args.input_method,
        )
        for space in spaces
        for lens in lenses
        for n_cubes in n_cubes_values
        for overlap in overlap_values
    ]


def _metrics_row(result: dict[str, Any]) -> dict[str, Any]:
    config: MapperConfig = result["config"]
    node_table = result["node_table"]
    mean_imputation = float(pd.to_numeric(node_table.get("mean_imputation_fraction"), errors="coerce").mean()) if not node_table.empty else 0.0
    max_imputation = float(pd.to_numeric(node_table.get("mean_imputation_fraction"), errors="coerce").max()) if not node_table.empty else 0.0
    mean_derived = float(pd.to_numeric(node_table.get("physically_derived_fraction"), errors="coerce").mean()) if not node_table.empty else 0.0
    frac_high_imputation = float((pd.to_numeric(node_table.get("mean_imputation_fraction"), errors="coerce") > 0.30).mean()) if not node_table.empty else 0.0
    frac_high_derived = float((pd.to_numeric(node_table.get("physically_derived_fraction"), errors="coerce") > 0.30).mean()) if not node_table.empty else 0.0
    return {
        "config_id": result["config_id"],
        "input_method": config.input_method,
        "space": config.space,
        "lens": config.lens,
        "n_cubes": int(config.n_cubes),
        "overlap": float(config.overlap),
        "clusterer": config.clusterer,
        "min_samples": int(config.min_samples),
        "eps_percentile": float(config.eps_percentile),
        "rows_used": int(len(result["mapper_df"])),
        "n_features": int(len(result["used_features"])),
        "mean_node_imputation_fraction": mean_imputation,
        "max_node_imputation_fraction": max_imputation,
        "mean_node_physically_derived_fraction": mean_derived,
        "frac_nodes_high_imputation": frac_high_imputation,
        "frac_nodes_high_derived": frac_high_derived,
        "config_has_density_feature": bool(has_density_feature(config.space)),
        **result["graph_metrics"],
    }


def run_mapper_batch(
    mapper_df: pd.DataFrame,
    physical_df: pd.DataFrame,
    configs: list[MapperConfig],
    mapper_features_path: Path,
    physical_csv_path: Path,
    alignment_summary: dict[str, Any],
) -> dict[str, Any]:
    results = [build_mapper_graph(mapper_df, physical_df, config) for config in configs]
    metrics_df = pd.DataFrame([_metrics_row(result) for result in results]).sort_values(
        ["space", "lens", "n_cubes", "overlap"],
        ignore_index=True,
    )
    distances_df = compare_mapper_graphs(metrics_df)
    return {
        "mapper_features_path": mapper_features_path,
        "physical_csv_path": physical_csv_path,
        "alignment_summary": alignment_summary,
        "results": results,
        "metrics_df": metrics_df,
        "distances_df": distances_df,
        "config_summary": {
            "dataset": {
                "mapper_features_path": str(mapper_features_path),
                "physical_csv_path": str(physical_csv_path),
                "rows_input": int(len(mapper_df)),
            },
            "configs": [{**asdict(config), "config_id": config_id(config)} for config in configs],
        },
    }
