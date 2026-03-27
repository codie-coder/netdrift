from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.drift.engine import DriftEngine
from src.explainer.attribution import build_explanation
from src.features.pipeline import extract_all_features
from src.fusion.logistic_fusion import DriftFusion
from src.fusion.threshold import apply_threshold, percentile_threshold
from src.graph.pipeline import window_to_graph
from src.ingestion.windower import sliding_windows
from src.utils.config import load_config
from src.utils.data_utils import ensure_dirs, save_json
from src.utils.logger import get_logger
from src.utils.schema import (DriftContribution, DriftScores, Explanation,
                               WindowOutput)
from src.utils.seed import fix_seeds

log = get_logger(__name__)


def run_full_pipeline(
    df: pd.DataFrame,
    cfg: dict,
    label_col: Optional[str] = None,
) -> Dict:
    fix_seeds(cfg["seed"])
    ensure_dirs([cfg["output"]["results_dir"], cfg["output"]["figures_dir"]])

    ws  = cfg["windowing"]["window_size_sec"]
    ss  = cfg["windowing"]["step_size_sec"]
    me  = cfg["windowing"]["min_edges"]

    drift_engine = DriftEngine(cfg)
    fusion       = DriftFusion(cfg)

    window_outputs: List[Dict]   = []
    drift_vectors:  List[List]   = []
    labels:         List[int]    = []
    baseline_set                 = False

    windows = list(sliding_windows(df, ws, ss, me))
    log.info("pipeline started", n_windows=len(windows))

    for window_id, t_start, t_end, window_df in windows:
        G, lcc, stats = window_to_graph(window_df, cfg)

        if stats["is_empty"]:
            log.warning("empty graph, skipping", window_id=window_id)
            continue

        feats = extract_all_features(G, cfg)
        hists = feats["histograms"]

        if not baseline_set:
            drift_engine.set_baseline(hists)
            baseline_set = True
            log.info("baseline window set", window_id=window_id)

        drift_result = drift_engine.compute(hists)
        scores       = drift_result["scores"]

        drift_vectors.append([
            scores["node"],
            scores["community"],
            scores["path"],
        ])

        true_label = 0
        if label_col and label_col in window_df.columns:
            true_label = int(window_df[label_col].max())
        labels.append(true_label)

        explanation = build_explanation(
            G,
            feats["node"],
            feats["community"],
            scores,
        )

        window_outputs.append({
            "window_id":    window_id,
            "timestamp":    float(t_start),
            "drift_scores": scores,
            "graph_stats":  stats,
            "explanation":  explanation,
            "true_label":   true_label,
        })

    # train fusion on collected drift vectors
    if len(drift_vectors) > 10 and sum(labels) > 0:
        fusion.fit(drift_vectors, labels)
        fusion.save()
    else:
        log.warning(
            "not enough labeled data for fusion training — using fallback scorer",
            n_windows=len(drift_vectors),
            n_anomalies=sum(labels),
        )

    # score all windows
    final_scores = [
        fusion.score(*dv) for dv in drift_vectors
    ]

    threshold = percentile_threshold(
        final_scores,
        contamination=cfg["detection"]["contamination"],
    )
    anomaly_flags = apply_threshold(final_scores, threshold)

    # assemble final outputs
    results = []
    for i, w in enumerate(window_outputs):
        if i >= len(final_scores):
            break

        expl = w["explanation"]
        out  = WindowOutput(
            window_id     = w["window_id"],
            timestamp     = w["timestamp"],
            drift_scores  = DriftScores(**w["drift_scores"]),
            final_score   = round(final_scores[i], 6),
            anomaly_label = bool(anomaly_flags[i]),
            explanation   = Explanation(
                top_nodes          = expl["top_nodes"],
                top_communities    = expl["top_communities"],
                drift_contribution = DriftContribution(
                    **expl["drift_contribution"]
                ),
            ),
        )
        results.append(out.model_dump())

    log.info(
        "pipeline complete",
        total_windows=len(results),
        anomalies=sum(r["anomaly_label"] for r in results),
    )

    return {
        "windows":       results,
        "drift_vectors": drift_vectors,
        "labels":        labels,
        "threshold":     threshold,
        "n_anomalies":   sum(r["anomaly_label"] for r in results),
    }
