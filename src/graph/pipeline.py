from typing import Tuple

import networkx as nx
import pandas as pd

from src.graph.builder import build_graph, graph_stats
from src.graph.lcc import get_lcc
from src.utils.logger import get_logger

log = get_logger(__name__)


def window_to_graph(
    window_df: pd.DataFrame,
    cfg: dict,
) -> Tuple[nx.DiGraph, nx.Graph, dict]:
    """
    Convert a window DataFrame into:
    - G      : full directed graph
    - lcc    : largest connected component (undirected)
    - stats  : graph statistics dict
    """
    weight_col = cfg["graph"]["edge_weight"]
    G    = build_graph(window_df, weight_col=weight_col)
    lcc  = get_lcc(G)
    stats = graph_stats(G)
    stats["lcc_nodes"]    = lcc.number_of_nodes()
    stats["lcc_fraction"] = (
        round(lcc.number_of_nodes() / max(G.number_of_nodes(), 1), 3)
    )
    return G, lcc, stats
