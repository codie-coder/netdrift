from typing import Dict

import networkx as nx
import numpy as np

from src.features.node_features      import extract_node_features
from src.features.community_features import extract_community_features
from src.features.path_features      import extract_path_features
from src.features.histogram          import to_histogram
from src.utils.logger                import get_logger

log = get_logger(__name__)


def extract_all_features(G: nx.DiGraph, cfg: dict) -> Dict:
    bins      = cfg["features"]["histogram_bins"]
    comm_cfg  = cfg["features"]["community"]
    path_cfg  = cfg["features"]["path"]

    node_feats = extract_node_features(G)
    comm_feats = extract_community_features(
        G,
        n_runs=comm_cfg["n_runs"],
        nmi_threshold=comm_cfg["nmi_threshold"],
    )
    path_feats = extract_path_features(
        G,
        min_nodes=path_cfg["min_nodes"],
        min_density=path_cfg["min_density"],
    )

    node_hist = to_histogram(node_feats["degree"], bins=bins)

    if comm_feats["skipped"]:
        comm_hist = np.ones(bins) / bins
    else:
        comm_hist = to_histogram(comm_feats["comm_sizes"], bins=bins)

    if path_feats["skipped"]:
        path_hist = np.ones(bins) / bins
    else:
        path_hist = to_histogram(path_feats["harmonic"], bins=bins)

    log.info("all features extracted")

    return {
        "node":      node_feats,
        "community": comm_feats,
        "path":      path_feats,
        "histograms": {
            "node":      node_hist,
            "community": comm_hist,
            "path":      path_hist,
        },
    }
