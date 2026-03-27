from typing import List

import numpy as np


def to_histogram(values: List[float], bins: int = 50) -> np.ndarray:
    if not values:
        return np.ones(bins) / bins

    arr  = np.array(values, dtype=float)
    hist, _ = np.histogram(arr, bins=bins, density=False)
    hist = hist.astype(float) + 1e-10
    return hist / hist.sum()
