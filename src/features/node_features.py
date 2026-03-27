from typing import Dict, List

import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)


def extract_node_features(G: nx.DiGraph, k_approx: int = 500) -> Dict[str, List[float]]:
    if G.number_of_nodes() == 0:
        return {"degree": [], "betweenness": [], "clustering": []}

    n = G.number_of_nodes()
    k = min(k_approx, n)

    degree    = nx.degree_centrality(G)
    between   = nx.betweenness_centrality(G, k=k, normalized=True, seed=42)
    UG        = G.to_undirected()
    clustering = nx.clustering(UG)

    result = {
        "degree":      [degree[node]     for node in G.nodes()],
        "betweenness": [between[node]    for node in G.nodes()],
        "clustering":  [clustering[node] for node in G.nodes()],
    }

    log.info(
        "node features extracted",
        n_nodes=n,
        k_approx=k,
        degree_mean=round(sum(result["degree"]) / n, 4),
    )
    return result
