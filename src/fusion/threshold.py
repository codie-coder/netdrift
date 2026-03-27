from typing import List, Tuple

import numpy as np

from src.utils.logger import get_logger

log = get_logger(__name__)


def percentile_threshold(
    scores: List[float],
    contamination: float = 0.05,
) -> float:
    threshold = float(np.percentile(scores, (1 - contamination) * 100))
    log.info(
        "threshold computed",
        threshold=round(threshold, 4),
        contamination=contamination,
        n_scores=len(scores),
    )
    return threshold


def apply_threshold(
    scores: List[float],
    threshold: float,
) -> List[bool]:
    return [s >= threshold for s in scores]
