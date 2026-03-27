from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)

from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_metrics(
    y_true: List[int],
    y_scores: List[float],
    threshold: float,
) -> Dict:
    y_pred = [1 if s >= threshold else 0 for s in y_scores]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        auc = 0.0

    report = classification_report(
        y_true, y_pred,
        target_names=["normal", "anomaly"],
        zero_division=0,
    )

    log.info(
        "metrics computed",
        accuracy=round(acc, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        f1=round(f1, 4),
        auc=round(auc, 4),
    )

    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1":        round(f1, 4),
        "auc":       round(auc, 4),
        "report":    report,
        "y_pred":    y_pred,
        "y_scores":  y_scores,
        "y_true":    y_true,
    }


def plot_roc_curve(
    y_true: List[int],
    y_scores: List[float],
    save_path: str = "figures/roc_curve.png",
    dataset: str = "synthetic",
) -> None:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
    except ValueError:
        log.warning("cannot plot ROC — only one class in y_true")
        return

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#4f8ef7", lw=2,
            label=f"NetDrift (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1,
            linestyle="--", label="Random")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title(f"ROC Curve — {dataset}", fontsize=12)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ROC curve saved", path=save_path)


def plot_drift_timeline(
    window_ids: List[int],
    drift_scores: List[float],
    anomaly_flags: List[bool],
    save_path: str = "figures/drift_timeline.png",
    dataset: str = "synthetic",
) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(window_ids, drift_scores, color="#4f8ef7",
            lw=1.5, label="Drift score", zorder=2)

    anomaly_ids    = [w for w, f in zip(window_ids, anomaly_flags) if f]
    anomaly_scores = [s for s, f in zip(drift_scores, anomaly_flags) if f]
    ax.scatter(anomaly_ids, anomaly_scores, color="#e24b4a",
               zorder=3, s=40, label="Anomaly", marker="x")

    if drift_scores:
        threshold = min(s for s, f in zip(drift_scores, anomaly_flags) if f)                     if any(anomaly_flags) else max(drift_scores)
        ax.axhline(y=threshold, color="#fb923c", lw=1,
                   linestyle="--", label="Threshold", zorder=1)

    ax.set_xlabel("Window ID", fontsize=11)
    ax.set_ylabel("Final drift score", fontsize=11)
    ax.set_title(f"Drift score timeline — {dataset}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("drift timeline saved", path=save_path)


def plot_ablation(
    ablation_results: Dict[str, Dict],
    save_path: str = "figures/ablation_bar.png",
) -> None:
    labels  = list(ablation_results.keys())
    f1s     = [ablation_results[k]["f1"]  for k in labels]
    aucs    = [ablation_results[k]["auc"] for k in labels]

    x    = np.arange(len(labels))
    w    = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, f1s,  w, label="F1",  color="#4f8ef7", alpha=0.85)
    ax.bar(x + w/2, aucs, w, label="AUC", color="#2dd4bf", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Ablation study — F1 and AUC by configuration", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("ablation bar chart saved", path=save_path)


def run_ablation(
    drift_vectors: List[List[float]],
    y_true: List[int],
    threshold: float,
) -> Dict:
    from src.fusion.logistic_fusion import DriftFusion
    from src.utils.config import load_config

    cfg = load_config()

    # Each config maps name -> (node_w, comm_w, path_w)
    # Zero weight = ablate that scale
    configs = {
        "full_model":         (1.0, 1.0, 1.0),
        "no_node_drift":      (0.0, 1.0, 1.0),
        "no_community_drift": (1.0, 0.0, 1.0),
        "no_path_drift":      (1.0, 1.0, 0.0),
        "node_only":          (1.0, 0.0, 0.0),
        "community_only":     (0.0, 1.0, 0.0),
        "path_only":          (0.0, 0.0, 1.0),
    }

    results = {}
    dv = np.array(drift_vectors)

    for name, (wn, wc, wp) in configs.items():
        # apply weights then re-normalise to [0,1] range
        weighted = dv * np.array([wn, wc, wp])
        row_sums = weighted.sum(axis=1, keepdims=True) + 1e-10
        normalised = (weighted / row_sums).tolist()

        fusion = DriftFusion(cfg)
        if sum(y_true) > 0:
            fusion.fit(normalised, y_true)
        scores = [fusion.score(r[0], r[1], r[2]) for r in normalised]

        metrics = compute_metrics(y_true, scores, threshold)
        results[name] = {"f1": metrics["f1"], "auc": metrics["auc"]}
        log.info("ablation", config=name, f1=metrics["f1"], auc=metrics["auc"])

    return results
