import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluation import (
    compute_metrics,
    plot_ablation,
    plot_drift_timeline,
    plot_roc_curve,
    run_ablation,
)
from src.pipeline import run_full_pipeline
from src.utils.config import load_config
from src.utils.data_utils import save_json
from src.utils.seed import fix_seeds

fix_seeds(42)

parser = argparse.ArgumentParser(description="NetDrift — reproduce paper results")
parser.add_argument("--config",    default="configs/default.yaml")
parser.add_argument("--dataset",   default="synthetic",
                    choices=["synthetic", "cicids2017", "unswnb15"])
parser.add_argument("--data_path", default=None)
args = parser.parse_args()

cfg     = load_config(args.config)
dataset = args.dataset

if dataset == "synthetic":
    print("Running on synthetic dataset...")
    np.random.seed(42)
    n = 2000
    df = pd.DataFrame({
        "src_ip":    [f"192.168.{np.random.randint(1,5)}.{np.random.randint(1,50)}"
                      for _ in range(n)],
        "dst_ip":    [f"10.0.{np.random.randint(0,3)}.{np.random.randint(1,20)}"
                      for _ in range(n)],
        "timestamp": np.sort(np.random.uniform(0, 7200, n)),
        "bytes":     np.random.randint(64, 65535, n),
        "label":     ([0] * 1800) + ([1] * 200),
    })
else:
    if not args.data_path:
        raise ValueError(f"--data_path required for dataset: {dataset}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows from {args.data_path}")

# run pipeline
results   = run_full_pipeline(df, cfg, label_col="label")
windows   = results["windows"]
dv        = results["drift_vectors"]
y_true    = results["labels"]
threshold = results["threshold"]

# collect scores and flags
y_scores      = [w["final_score"]   for w in windows]
anomaly_flags = [w["anomaly_label"] for w in windows]
window_ids    = [w["window_id"]     for w in windows]
y_true_w      = y_true[:len(windows)]

# metrics
metrics = compute_metrics(y_true_w, y_scores, threshold)

print(f"\n--- Metrics ({dataset}) ---")
print(f"Accuracy  : {metrics['accuracy']}")
print(f"Precision : {metrics['precision']}")
print(f"Recall    : {metrics['recall']}")
print(f"F1        : {metrics['f1']}")
print(f"AUC       : {metrics['auc']}")
print(f"\n{metrics['report']}")

# plots
plot_roc_curve(y_true_w, y_scores,
               save_path=f"figures/roc_{dataset}.png",
               dataset=dataset)

plot_drift_timeline(window_ids, y_scores, anomaly_flags,
                    save_path=f"figures/drift_timeline_{dataset}.png",
                    dataset=dataset)

# ablation
if sum(y_true) > 0 and len(dv) > 10:
    ablation = run_ablation(dv, y_true[:len(dv)], threshold)
    plot_ablation(ablation, save_path=f"figures/ablation_{dataset}.png")
    print("\n--- Ablation ---")
    for k, v in ablation.items():
        print(f"  {k:<25} F1={v['f1']:.4f}  AUC={v['auc']:.4f}")
else:
    ablation = {}

# save full results
out = {
    "dataset":          dataset,
    "seed":             cfg["seed"],
    "config":           args.config,
    "metrics":          {k: v for k, v in metrics.items()
                         if k not in ("report", "y_pred", "y_scores", "y_true")},
    "ablation":         ablation,
    "n_windows":        len(windows),
    "n_anomalies":      results["n_anomalies"],
    "threshold":        threshold,
}
out_path = Path(cfg["output"]["results_dir"]) / f"{dataset}_results.json"
save_json(out, str(out_path))

print(f"\nResults saved to : {out_path}")
print(f"Figures saved to : figures/")
