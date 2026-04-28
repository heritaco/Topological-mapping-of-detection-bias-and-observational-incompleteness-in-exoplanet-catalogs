from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class NodeNeighborhood:
    node_id: str
    n1_nodes: list[str]
    n2_nodes: list[str]
    component_nodes: list[str]


def build_graph(edges: pd.DataFrame, all_nodes: list[str]) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(str(node_id) for node_id in all_nodes)
    if edges.empty:
        return graph
    for row in edges.itertuples(index=False):
        source = str(getattr(row, "source"))
        target = str(getattr(row, "target"))
        weight = float(getattr(row, "n_shared_members", 1.0)) if hasattr(row, "n_shared_members") else 1.0
        graph.add_edge(source, target, weight=weight)
    return graph


def node_neighborhood(graph: nx.Graph, node_id: str) -> NodeNeighborhood:
    node_id = str(node_id)
    if node_id not in graph:
        return NodeNeighborhood(node_id=node_id, n1_nodes=[], n2_nodes=[], component_nodes=[])
    shortest = nx.single_source_shortest_path_length(graph, node_id, cutoff=2)
    n1_nodes = sorted(str(other) for other, distance in shortest.items() if distance == 1)
    n2_nodes = sorted(str(other) for other, distance in shortest.items() if distance == 2)
    component_nodes = sorted(str(value) for value in nx.node_connected_component(graph, node_id))
    return NodeNeighborhood(node_id=node_id, n1_nodes=n1_nodes, n2_nodes=n2_nodes, component_nodes=component_nodes)


def _component_member_size(component_nodes: list[str], membership: pd.DataFrame) -> int:
    if membership.empty or not component_nodes:
        return 0
    subset = membership[membership["node_id"].astype(str).isin(component_nodes)].copy()
    if subset.empty:
        return 0
    if "row_index" in subset.columns:
        return int(subset["row_index"].nunique())
    return int(subset["pl_name"].astype(str).nunique()) if "pl_name" in subset.columns else int(len(subset))


def graph_metrics(graph: nx.Graph, membership: pd.DataFrame, epsilon: float = 1.0e-9) -> pd.DataFrame:
    if graph.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node_id"])
    components = list(nx.connected_components(graph))
    betweenness_full = nx.betweenness_centrality(graph) if graph.number_of_nodes() > 1 else {}
    clustering_full = nx.clustering(graph)
    degree_dict = dict(graph.degree())
    weighted_degree_dict = dict(graph.degree(weight="weight"))
    articulation_points = set(nx.articulation_points(graph)) if graph.number_of_nodes() > 2 else set()
    degree_max = max(degree_dict.values()) if degree_dict else 0
    rows: list[dict[str, object]] = []
    for component_index, component_nodes in enumerate(components):
        subgraph = graph.subgraph(component_nodes).copy()
        closeness_map = nx.closeness_centrality(subgraph) if subgraph.number_of_nodes() > 1 else {str(next(iter(component_nodes))): 0.0}
        eccentricity_map = nx.eccentricity(subgraph) if subgraph.number_of_nodes() > 1 else {}
        n_members_map = membership.groupby(membership["node_id"].astype(str)).size().to_dict() if not membership.empty else {}
        largest_node = max(component_nodes, key=lambda value: float(n_members_map.get(str(value), 0.0))) if component_nodes else None
        core_node = max(
            component_nodes,
            key=lambda value: (
                float(closeness_map.get(str(value), 0.0)),
                float(degree_dict.get(str(value), 0.0)),
                float(n_members_map.get(str(value), 0.0)),
            ),
        ) if component_nodes else None
        for node_id in subgraph.nodes():
            node_id = str(node_id)
            distance_to_largest = nx.shortest_path_length(subgraph, node_id, largest_node) if largest_node is not None else None
            distance_to_core = nx.shortest_path_length(subgraph, node_id, core_node) if core_node is not None else None
            degree = int(degree_dict.get(node_id, 0))
            rows.append(
                {
                    "node_id": node_id,
                    "degree": degree,
                    "weighted_degree": float(weighted_degree_dict.get(node_id, 0.0)),
                    "betweenness_centrality": float(betweenness_full.get(node_id, 0.0)),
                    "closeness_centrality": float(closeness_map.get(node_id, 0.0)),
                    "clustering_coefficient": float(clustering_full.get(node_id, 0.0)),
                    "component_id_graph": component_index,
                    "component_size_nodes": int(len(component_nodes)),
                    "component_size_members": _component_member_size([str(value) for value in component_nodes], membership),
                    "distance_to_largest_node": float(distance_to_largest) if distance_to_largest is not None else None,
                    "distance_to_component_core": float(distance_to_core) if distance_to_core is not None else None,
                    "is_articulation_point": bool(node_id in articulation_points),
                    "eccentricity": float(eccentricity_map.get(node_id)) if node_id in eccentricity_map else None,
                    "size_weight": None,
                    "degree_weight": float(np.log1p(degree) / max(np.log1p(degree_max), epsilon)) if degree_max > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows)


def add_network_support(frame: pd.DataFrame, epsilon: float = 1.0e-9) -> pd.DataFrame:
    out = frame.copy()
    n_members = pd.to_numeric(out.get("n_members"), errors="coerce").fillna(0).clip(lower=0)
    n_max = float(n_members.max()) if float(n_members.max()) > 0 else 1.0
    degree = pd.to_numeric(out.get("degree"), errors="coerce").fillna(0).clip(lower=0)
    degree_max = float(degree.max()) if float(degree.max()) > 0 else 1.0
    out["size_weight"] = np.log1p(n_members) / max(np.log1p(n_max), epsilon)
    out["degree_weight"] = np.log1p(degree) / max(np.log1p(degree_max), epsilon) if degree_max > 0 else 0.0
    out["S_net"] = np.sqrt(out["size_weight"] * out["degree_weight"])
    return out
