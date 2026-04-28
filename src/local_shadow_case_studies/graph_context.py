from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import networkx as nx
import pandas as pd


@dataclass
class CaseNeighborhood:
    node_id: str
    n1_nodes: list[str]
    n2_nodes: list[str]
    component_nodes: list[str]
    graph_metrics: dict[str, Any]


def build_graph(edge_table: pd.DataFrame, node_table: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    for node_id in node_table["node_id"].astype(str).tolist():
        graph.add_node(node_id)
    if edge_table.empty:
        return graph
    for row in edge_table.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        if not graph.has_node(source):
            graph.add_node(source)
        if not graph.has_node(target):
            graph.add_node(target)
        weight = float(getattr(row, "n_shared_members", 1.0)) if hasattr(row, "n_shared_members") else 1.0
        graph.add_edge(source, target, weight=weight)
    return graph


def _component_member_size(component_nodes: list[str], membership: pd.DataFrame) -> int:
    subset = membership[membership["node_id"].astype(str).isin(component_nodes)].copy()
    if subset.empty:
        return 0
    return int(subset["row_index"].nunique() if "row_index" in subset.columns else len(subset))


def case_neighborhood(node_id: str, graph: nx.Graph, node_table: pd.DataFrame, membership: pd.DataFrame) -> CaseNeighborhood:
    node_id = str(node_id)
    n1_nodes = sorted(str(neighbor) for neighbor in graph.neighbors(node_id)) if graph.has_node(node_id) else []
    shortest = nx.single_source_shortest_path_length(graph, node_id, cutoff=2) if graph.has_node(node_id) else {}
    n2_nodes = sorted(str(other) for other, distance in shortest.items() if other != node_id and distance <= 2 and other not in n1_nodes)
    component_nodes: list[str] = []
    if graph.has_node(node_id):
        component_nodes = sorted(str(value) for value in nx.node_connected_component(graph, node_id))
    component_subgraph = graph.subgraph(component_nodes).copy() if component_nodes else nx.Graph()

    degree = int(graph.degree(node_id)) if graph.has_node(node_id) else 0
    weighted_degree = float(graph.degree(node_id, weight="weight")) if graph.has_node(node_id) else 0.0
    betweenness = nx.betweenness_centrality(component_subgraph).get(node_id, 0.0) if component_subgraph.number_of_nodes() > 0 else 0.0
    closeness = nx.closeness_centrality(component_subgraph).get(node_id, 0.0) if component_subgraph.number_of_nodes() > 1 else 0.0
    clustering = nx.clustering(component_subgraph, node_id) if component_subgraph.has_node(node_id) else 0.0
    articulation = node_id in set(nx.articulation_points(component_subgraph)) if component_subgraph.number_of_nodes() > 2 else False

    component_size_nodes = len(component_nodes)
    component_size_members = _component_member_size(component_nodes, membership)
    degree_map = dict(component_subgraph.degree()) if component_subgraph.number_of_nodes() > 0 else {}
    closeness_map = nx.closeness_centrality(component_subgraph) if component_subgraph.number_of_nodes() > 1 else {node_id: 0.0}
    node_sizes = (
        node_table.set_index(node_table["node_id"].astype(str))["n_members"].to_dict()
        if "n_members" in node_table.columns
        else {}
    )
    largest_node = None
    if component_nodes:
        largest_node = max(component_nodes, key=lambda value: float(node_sizes.get(value, 0.0)))
    component_core = None
    if component_nodes:
        component_core = max(component_nodes, key=lambda value: (float(closeness_map.get(value, 0.0)), float(degree_map.get(value, 0.0)), float(node_sizes.get(value, 0.0))))

    def _distance_to(target: str | None) -> float | None:
        if not target or not graph.has_node(node_id) or not graph.has_node(target):
            return None
        try:
            return float(nx.shortest_path_length(component_subgraph, node_id, target))
        except nx.NetworkXNoPath:
            return None

    eccentricity = None
    if component_subgraph.number_of_nodes() > 1:
        try:
            eccentricity = float(nx.eccentricity(component_subgraph, node_id))
        except Exception:
            eccentricity = None

    graph_metrics = {
        "degree": degree,
        "weighted_degree": weighted_degree,
        "betweenness_centrality": betweenness,
        "closeness_centrality": closeness,
        "clustering_coefficient": clustering,
        "component_size_nodes": component_size_nodes,
        "component_size_members": component_size_members,
        "distance_to_largest_node": _distance_to(largest_node),
        "distance_to_component_core": _distance_to(component_core),
        "is_articulation_point": articulation,
        "eccentricity": eccentricity,
    }
    return CaseNeighborhood(
        node_id=node_id,
        n1_nodes=n1_nodes,
        n2_nodes=n2_nodes,
        component_nodes=component_nodes,
        graph_metrics=graph_metrics,
    )

