import networkx as nx

from src.utils.logger import get_logger

log = get_logger(__name__)


def get_lcc(G: nx.DiGraph) -> nx.Graph:
    """Return largest connected component as undirected graph."""
    if G.number_of_nodes() == 0:
        return nx.Graph()

    UG = G.to_undirected()
    components = list(nx.connected_components(UG))
    if not components:
        return nx.Graph()

    lcc_nodes = max(components, key=len)
    lcc = UG.subgraph(lcc_nodes).copy()

    log.info(
        "LCC extracted",
        lcc_nodes=lcc.number_of_nodes(),
        total_nodes=G.number_of_nodes(),
        lcc_fraction=round(lcc.number_of_nodes() / G.number_of_nodes(), 3),
    )
    return lcc
