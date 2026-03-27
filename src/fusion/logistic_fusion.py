from pathlib import Path
from typing import List, Optional, Tuple

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

log = get_logger(__name__)


class DriftFusion:
    def __init__(self, cfg: dict):
        fc = cfg["fusion"]
        self.model = LogisticRegression(
            C=fc["C"],
            max_iter=fc["max_iter"],
            random_state=fc["random_state"],
        )
        self.scaler   = StandardScaler()
        self.fitted   = False
        self.cfg      = cfg

    def _to_vector(
        self,
        node_drift: float,
        community_drift: float,
        path_drift: float,
    ) -> np.ndarray:
        return np.array([[node_drift, community_drift, path_drift]])

    def fit(
        self,
        drift_vectors: List[List[float]],
        labels: List[int],
    ) -> None:
        X = np.array(drift_vectors)
        y = np.array(labels)
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self.fitted = True
        log.info(
            "fusion model fitted",
            n_samples=len(labels),
            n_anomalies=int(sum(labels)),
            classes=list(self.model.classes_),
        )

    def score(
        self,
        node_drift: float,
        community_drift: float,
        path_drift: float,
    ) -> float:
        if not self.fitted:
            # fallback: weighted average before training
            return float(
                0.4 * node_drift +
                0.2 * community_drift +
                0.4 * path_drift
            )
        x = self._to_vector(node_drift, community_drift, path_drift)
        x = self.scaler.transform(x)
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path: str = "models/fusion.pkl") -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, str(out))
        log.info("fusion model saved", path=str(out))

    def load(self, path: str = "models/fusion.pkl") -> None:
        data = joblib.load(path)
        self.model  = data["model"]
        self.scaler = data["scaler"]
        self.fitted = True
        log.info("fusion model loaded", path=path)
