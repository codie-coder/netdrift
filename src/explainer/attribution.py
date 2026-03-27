from typing import Dict, List

import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_top_nodes(
    G: nx.DiGraph,
    node_drift_scores: Dict[int, float],
    top_k: int = 5,
) -> List[str]:
    rev_map = G.graph.get("node_map", {})
    sorted_nodes = sorted(
        node_drift_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    top = []
    for node_id, _ in sorted_nodes[:top_k]:
        ip = rev_map.get(node_id, str(node_id))
        top.append(ip)
    return top


def get_top_communities(
    partition: Dict[int, int],
    comm_drift: float,
    top_k: int = 3,
) -> List[str]:
    if not partition:
        return []
    comm_sizes = {}
    for node, comm in partition.items():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
    top = sorted(comm_sizes.keys(), key=lambda c: comm_sizes[c], reverse=True)
    return [f"comm_{c}" for c in top[:top_k]]


def drift_contribution(
    node_d: float,
    community_d: float,
    path_d: float,
) -> Dict[str, float]:
    total = node_d + community_d + path_d + 1e-10
    return {
        "node":      round(node_d / total, 4),
        "community": round(community_d / total, 4),
        "path":      round(path_d / total, 4),
    }


def build_explanation(
    G: nx.DiGraph,
    node_feats: Dict,
    comm_feats: Dict,
    drift_scores: Dict,
) -> Dict:
    node_drift_map = {
        i: d * b
        for i, (d, b) in enumerate(
            zip(node_feats["betweenness"], node_feats["degree"])
        )
    }

    top_nodes  = get_top_nodes(G, node_drift_map)
    top_comms  = get_top_communities(
        comm_feats.get("partition", {}),
        drift_scores["community"],
    )
    contrib    = drift_contribution(
        drift_scores["node"],
        drift_scores["community"],
        drift_scores["path"],
    )

    log.info(
        "explanation built",
        top_nodes=top_nodes[:3],
        n_communities=len(top_comms),
    )

    return {
        "top_nodes":          top_nodes,
        "top_communities":    top_comms,
        "drift_contribution": contrib,
    }
