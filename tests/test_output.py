import numpy as np
import pandas as pd
import pytest

from src.pipeline import run_full_pipeline
from src.utils.config import load_config
from src.utils.schema import WindowOutput


def test_pipeline_output_schema():
    cfg = load_config()
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "src_ip":    [f"192.168.1.{np.random.randint(1,20)}" for _ in range(n)],
        "dst_ip":    [f"10.0.0.{np.random.randint(1,10)}" for _ in range(n)],
        "timestamp": np.sort(np.random.uniform(0, 600, n)),
        "bytes":     np.random.randint(100, 50000, n),
        "label":     [0] * 450 + [1] * 50,
    })
    results = run_full_pipeline(df, cfg, label_col="label")
    assert "windows" in results
    assert "n_anomalies" in results
    assert isinstance(results["n_anomalies"], int)
    for w in results["windows"]:
        WindowOutput(**w)


def test_drift_contribution_sums_to_one():
    cfg = load_config()
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        "src_ip":    [f"192.168.1.{np.random.randint(1,20)}" for _ in range(n)],
        "dst_ip":    [f"10.0.0.{np.random.randint(1,10)}" for _ in range(n)],
        "timestamp": np.sort(np.random.uniform(0, 600, n)),
        "bytes":     np.random.randint(100, 50000, n),
    })
    results = run_full_pipeline(df, cfg)
    for w in results["windows"]:
        contrib = w["explanation"]["drift_contribution"]
        total = round(contrib["node"] + contrib["community"] + contrib["path"], 2)
        assert total == 1.0
