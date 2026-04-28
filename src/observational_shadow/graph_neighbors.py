from __future__ import annotations

import networkx as nx
import pandas as pd


def build_graph(node_ids: list[str], edge_table: pd.DataFrame) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from([str(node_id) for node_id in node_ids])
    if not edge_table.empty and {"source", "target"}.issubset(edge_table.columns):
        for row in edge_table.itertuples(index=False):
            source = str(getattr(row, "source"))
            target = str(getattr(row, "target"))
            if source in graph and target in graph:
                graph.add_edge(source, target)
    return graph


def neighbor_map(node_ids: list[str], edge_table: pd.DataFrame) -> dict[str, set[str]]:
    graph = build_graph(node_ids, edge_table)
    return {node_id: set(graph.neighbors(node_id)) for node_id in graph.nodes}


def component_lookup(node_ids: list[str], edge_table: pd.DataFrame) -> tuple[dict[str, int], dict[int, int]]:
    graph = build_graph(node_ids, edge_table)
    lookup: dict[str, int] = {}
    sizes: dict[int, int] = {}
    for component_id, nodes in enumerate(nx.connected_components(graph)):
        node_list = sorted(str(node) for node in nodes)
        sizes[component_id] = len(node_list)
        for node_id in node_list:
            lookup[node_id] = component_id
    return lookup, sizes

