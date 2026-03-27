from typing import Dict, List, Tuple

import numpy as np
from sklearn.ensemble import IsolationForest

from src.utils.logger import get_logger

log = get_logger(__name__)


class IsoForestDetector:
    def __init__(self, cfg: dict):
        dc = cfg["detection"]
        self.model = IsolationForest(
            n_estimators=200,
            contamination=dc["contamination"],
            random_state=dc["random_state"],
            n_jobs=-1,
        )
        self.fitted = False

    def fit(self, X: np.ndarray) -> None:
        self.model.fit(X)
        self.fitted = True
        log.info("IsolationForest fitted", n_samples=len(X))

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("Model not fitted yet.")
        scores = self.model.decision_function(X)
        labels = self.model.predict(X)
        # convert: sklearn uses -1=anomaly, 1=normal
        # we use: True=anomaly, False=normal
        anomaly_flags = labels == -1
        # normalise scores to [0,1] — lower decision = more anomalous
        norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
        anomaly_scores = 1.0 - norm
        return anomaly_scores, anomaly_flags

    def fit_predict(
        self,
        X: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        self.fit(X)
        return self.predict(X)
