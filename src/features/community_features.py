from typing import Dict, Optional, Tuple

import networkx as nx
import community as community_louvain
from sklearn.metrics import normalized_mutual_info_score

from src.utils.logger import get_logger

log = get_logger(__name__)


def extract_community_features(
    G: nx.DiGraph,
    n_runs: int = 10,
    nmi_threshold: float = 0.7,
) -> Dict:
    if G.number_of_nodes() < 3:
        log.warning("graph too small for community detection")
        return {
            "modularity": 0.0,
            "n_communities": 0,
            "partition": {},
            "stable": False,
            "skipped": True,
        }

    UG = G.to_undirected()
    partitions = []
    for seed in range(n_runs):
        try:
            p = community_louvain.best_partition(UG, random_state=seed)
            partitions.append(p)
        except Exception as e:
            log.warning("louvain run failed", seed=seed, error=str(e))

    if not partitions:
        return {
            "modularity": 0.0,
            "n_communities": 0,
            "partition": {},
            "stable": False,
            "skipped": True,
        }

    nodes    = list(UG.nodes())
    label_sets = [[p[n] for n in nodes] for p in partitions]

    nmis = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            nmis.append(
                normalized_mutual_info_score(label_sets[i], label_sets[j])
            )

    mean_nmi = sum(nmis) / len(nmis) if nmis else 0.0
    stable   = mean_nmi >= nmi_threshold

    best = max(
        partitions,
        key=lambda p: community_louvain.modularity(p, UG),
    )
    modularity    = community_louvain.modularity(best, UG)
    n_communities = len(set(best.values()))

    comm_sizes = {}
    for node, comm_id in best.items():
        comm_sizes[comm_id] = comm_sizes.get(comm_id, 0) + 1

    log.info(
        "community features extracted",
        n_communities=n_communities,
        modularity=round(modularity, 4),
        mean_nmi=round(mean_nmi, 4),
        stable=stable,
    )

    return {
        "modularity":     modularity,
        "n_communities":  n_communities,
        "partition":      best,
        "comm_sizes":     list(comm_sizes.values()),
        "stable":         stable,
        "skipped":        False,
    }
