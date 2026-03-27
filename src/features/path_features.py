from typing import Dict

import networkx as nx

from src.graph.lcc import get_lcc
from src.utils.logger import get_logger

log = get_logger(__name__)


def extract_path_features(
    G: nx.DiGraph,
    min_nodes: int = 10,
    min_density: float = 0.01,
) -> Dict:
    skipped_result = {
        "harmonic": [],
        "lcc_fraction": 0.0,
        "skipped": True,
    }

    if G.number_of_nodes() == 0:
        log.warning("empty graph — skipping path features")
        return skipped_result

    lcc = get_lcc(G)
    lcc_fraction = lcc.number_of_nodes() / max(G.number_of_nodes(), 1)

    if lcc.number_of_nodes() < min_nodes:
        log.warning(
            "LCC too small — skipping path features",
            lcc_nodes=lcc.number_of_nodes(),
            min_nodes=min_nodes,
        )
        return {**skipped_result, "lcc_fraction": lcc_fraction}

    density = nx.density(lcc)
    if density < min_density:
        log.warning(
            "LCC too sparse — skipping path features",
            density=round(density, 5),
            min_density=min_density,
        )
        return {**skipped_result, "lcc_fraction": lcc_fraction}

    hc = nx.harmonic_centrality(lcc)
    harmonic_values = list(hc.values())

    log.info(
        "path features extracted",
        lcc_nodes=lcc.number_of_nodes(),
        lcc_fraction=round(lcc_fraction, 3),
        harmonic_mean=round(sum(harmonic_values) / len(harmonic_values), 4),
    )

    return {
        "harmonic":     harmonic_values,
        "lcc_fraction": lcc_fraction,
        "skipped":      False,
    }
