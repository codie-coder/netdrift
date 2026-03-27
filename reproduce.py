import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipeline import run_full_pipeline
from src.utils.config import load_config
from src.utils.data_utils import save_json
from src.utils.seed import fix_seeds

fix_seeds(42)

parser = argparse.ArgumentParser(description="NetDrift — reproduce paper results")
parser.add_argument("--config",   default="configs/default.yaml")
parser.add_argument("--dataset",  default="synthetic",
                    choices=["synthetic", "cicids2017", "unswnb15"])
parser.add_argument("--data_path", default=None)
args = parser.parse_args()

cfg = load_config(args.config)

if args.dataset == "synthetic":
    print("Running on synthetic dataset (no real data needed)...")
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
        raise ValueError(f"--data_path required for dataset: {args.dataset}")
    df = pd.read_csv(args.data_path)
    print(f"Loaded {len(df)} rows from {args.data_path}")

results = run_full_pipeline(df, cfg, label_col="label")

out_path = Path(cfg["output"]["results_dir"]) / f"{args.dataset}_results.json"
save_json(results, str(out_path))

print(f"\n--- Results ---")
print(f"Total windows  : {len(results['windows'])}")
print(f"Anomalies found: {results['n_anomalies']}")
print(f"Threshold      : {results['threshold']:.4f}")
print(f"Saved to       : {out_path}")
