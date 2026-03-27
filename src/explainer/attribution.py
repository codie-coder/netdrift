from typing import Dict, List
import networkx as nx
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_top_nodes(G, node_drift_scores, top_k=5):
    rev_map = G.graph.get("node_map", {})
    sorted_nodes = sorted(node_drift_scores.items(), key=lambda x: x[1], reverse=True)
    top = []
    for node_id, _ in sorted_nodes[:top_k]:
        ip = rev_map.get(node_id, str(node_id))
        top.append(ip)
    return top


def get_top_communities(partition, comm_drift, top_k=3):
    if not partition:
        return []
    comm_sizes = {}
    for node, comm in partition.items():
        comm_sizes[comm] = comm_sizes.get(comm, 0) + 1
    top = sorted(comm_sizes.keys(), key=lambda c: comm_sizes[c], reverse=True)
    return [f"comm_{c}" for c in top[:top_k]]


def drift_contribution(node_d, community_d, path_d):
    total = node_d + community_d + path_d
    if total < 1e-10:
        return {"node": round(1/3, 4), "community": round(1/3, 4), "path": round(1/3, 4)}
    raw = {
        "node":      node_d / total,
        "community": community_d / total,
        "path":      path_d / total,
    }
    # round to 4dp then fix so they sum exactly to 1.0
    rounded = {k: round(v, 4) for k, v in raw.items()}
    diff = round(1.0 - sum(rounded.values()), 4)
    rounded["node"] = round(rounded["node"] + diff, 4)
    return rounded


def build_explanation(G, node_feats, comm_feats, drift_scores):
    node_drift_map = {
        i: d * b
        for i, (d, b) in enumerate(
            zip(node_feats["betweenness"], node_feats["degree"])
        )
    }
    top_nodes = get_top_nodes(G, node_drift_map)
    top_comms = get_top_communities(
        comm_feats.get("partition", {}),
        drift_scores["community"],
    )
    contrib = drift_contribution(
        drift_scores["node"],
        drift_scores["community"],
        drift_scores["path"],
    )
    log.info("explanation built", top_nodes=top_nodes[:3], n_communities=len(top_comms))
    return {
        "top_nodes":          top_nodes,
        "top_communities":    top_comms,
        "drift_contribution": contrib,
    }
