from typing import Dict, Optional

import numpy as np

from src.drift.measures import compute_drift
from src.utils.logger import get_logger

log = get_logger(__name__)


class DriftEngine:
    def __init__(self, cfg: dict):
        self.primary  = cfg["drift"]["primary"]
        self.measures = cfg["drift"]["measures"]
        self.baseline: Optional[Dict[str, np.ndarray]] = None

    def set_baseline(self, histograms: Dict[str, np.ndarray]) -> None:
        self.baseline = {k: np.array(v) for k, v in histograms.items()}
        log.info("baseline set", scales=list(self.baseline.keys()))

    def compute(self, histograms: Dict[str, np.ndarray]) -> Dict:
        if self.baseline is None:
            raise RuntimeError("Baseline not set. Call set_baseline() first.")

        results = {}
        for scale in ["node", "community", "path"]:
            p = self.baseline[scale]
            q = np.array(histograms[scale])
            results[scale] = compute_drift(p, q, self.measures)

        primary_scores = {
            scale: results[scale][self.primary]
            for scale in results
        }

        log.info(
            "drift computed",
            node=round(primary_scores["node"], 4),
            community=round(primary_scores["community"], 4),
            path=round(primary_scores["path"], 4),
            measure=self.primary,
        )

        return {
            "primary":        self.primary,
            "scores":         primary_scores,
            "all_measures":   results,
        }

    def update_baseline(self, histograms: Dict[str, np.ndarray]) -> None:
        self.set_baseline(histograms)
