from typing import Dict, List, Optional
from pydantic import BaseModel


class DriftScores(BaseModel):
    node: float
    community: float
    path: float


class DriftContribution(BaseModel):
    node: float
    community: float
    path: float


class Explanation(BaseModel):
    top_nodes: List[str]
    top_communities: List[str]
    drift_contribution: DriftContribution


class WindowOutput(BaseModel):
    window_id: int
    timestamp: float
    drift_scores: DriftScores
    final_score: float
    anomaly_label: bool
    explanation: Explanation
