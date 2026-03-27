import numpy as np
from scipy.stats import wasserstein_distance, entropy


def wasserstein(p: np.ndarray, q: np.ndarray) -> float:
    return float(wasserstein_distance(p, q))


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    return float(entropy(p, q))


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.array(p, dtype=float) + eps
    q = np.array(q, dtype=float) + eps
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * (entropy(p, m) + entropy(q, m)))


MEASURES = {
    "wasserstein": wasserstein,
    "kl":          kl_divergence,
    "js":          js_divergence,
}


def compute_drift(
    p: np.ndarray,
    q: np.ndarray,
    measures: list,
) -> dict:
    return {name: MEASURES[name](p, q) for name in measures}
