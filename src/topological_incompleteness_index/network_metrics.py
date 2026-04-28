from __future__ import annotations
import networkx as nx
import pandas as pd

def build_graph(edges: pd.DataFrame, all_nodes: list[str]) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(map(str, all_nodes))
    for _, row in edges.iterrows():
        g.add_edge(str(row["source"]), str(row["target"]))
    return g

def graph_metrics(g: nx.Graph) -> pd.DataFrame:
    if g.number_of_nodes() == 0:
        return pd.DataFrame(columns=["node_id"])
    degree = dict(g.degree())
    bet = nx.betweenness_centrality(g)
    close = nx.closeness_centrality(g)
    clust = nx.clustering(g)
    articulation = set(nx.articulation_points(g))
    components = list(nx.connected_components(g))
    comp_id = {}
    comp_size = {}
    for i, comp in enumerate(components):
        for n in comp:
            comp_id[n] = i
            comp_size[n] = len(comp)
    rows = []
    for node in g.nodes():
        rows.append({
            "node_id": node,
            "degree": degree.get(node, 0),
            "betweenness_centrality": bet.get(node, 0.0),
            "closeness_centrality": close.get(node, 0.0),
            "clustering_coefficient": clust.get(node, 0.0),
            "component_id_graph": comp_id.get(node, -1),
            "component_size_nodes": comp_size.get(node, 1),
            "is_articulation_point": node in articulation,
        })
    return pd.DataFrame(rows)

def neighbors(g: nx.Graph, node_id: str, order: int = 1) -> set[str]:
    if node_id not in g:
        return set()
    lengths = nx.single_source_shortest_path_length(g, node_id, cutoff=order)
    return {n for n, d in lengths.items() if 0 < d <= order}
