from typing import Dict, Tuple

import networkx as nx
import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def build_graph(df: pd.DataFrame, weight_col: str = "bytes") -> nx.DiGraph:
    G = nx.DiGraph()

    if df.empty:
        log.warning("empty dataframe — returning empty graph")
        return G

    # intern IPs to integers for speed at 100K+ node scale
    all_ips = list(set(df["src_ip"].tolist() + df["dst_ip"].tolist()))
    node_map: Dict[str, int] = {ip: i for i, ip in enumerate(all_ips)}
    rev_map:  Dict[int, str] = {i: ip for ip, i in node_map.items()}

    G.graph["node_map"] = rev_map
    G.graph["n_original_ips"] = len(all_ips)

    for _, row in df.iterrows():
        u = node_map[row["src_ip"]]
        v = node_map[row["dst_ip"]]
        w = float(row[weight_col]) if weight_col in row else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
            G[u][v]["count"]  += 1
        else:
            G.add_edge(u, v, weight=w, count=1)

    log.info(
        "graph built",
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
    )
    return G


def graph_stats(G: nx.DiGraph) -> Dict:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    return {
        "n_nodes":  n,
        "n_edges":  e,
        "density":  nx.density(G) if n > 1 else 0.0,
        "is_empty": n == 0,
    }
