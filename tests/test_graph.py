import networkx as nx
import pandas as pd

from src.graph.builder import build_graph, graph_stats
from src.graph.lcc import get_lcc


def test_build_graph_basic(sample_df):
    G = build_graph(sample_df)
    assert G.number_of_nodes() > 0
    assert G.number_of_edges() > 0


def test_build_graph_empty():
    df = pd.DataFrame(columns=["src_ip", "dst_ip", "timestamp", "bytes"])
    G = build_graph(df)
    assert G.number_of_nodes() == 0


def test_graph_has_node_map(sample_df):
    G = build_graph(sample_df)
    assert "node_map" in G.graph
    assert len(G.graph["node_map"]) == G.number_of_nodes()


def test_graph_stats_keys(sample_df):
    G = build_graph(sample_df)
    stats = graph_stats(G)
    assert "n_nodes" in stats
    assert "n_edges" in stats
    assert "density" in stats


def test_lcc_subset_of_graph(sample_df):
    G = build_graph(sample_df)
    lcc = get_lcc(G)
    assert lcc.number_of_nodes() <= G.number_of_nodes()


def test_lcc_empty_graph():
    G = nx.DiGraph()
    lcc = get_lcc(G)
    assert lcc.number_of_nodes() == 0
