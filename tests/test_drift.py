import numpy as np
import pytest

from src.drift.measures import js_divergence, kl_divergence, wasserstein
from src.drift.engine import DriftEngine


def test_wasserstein_identical():
    p = np.ones(50) / 50
    assert wasserstein(p, p) < 1e-6


def test_wasserstein_increases_with_drift():
    p = np.ones(50) / 50
    q = np.ones(50) / 50
    q[0] = 10.0
    q /= q.sum()
    assert wasserstein(p, q) > wasserstein(p, p)


def test_kl_identical():
    p = np.ones(50) / 50
    assert kl_divergence(p, p) < 1e-4


def test_js_symmetric():
    p = np.random.dirichlet(np.ones(50))
    q = np.random.dirichlet(np.ones(50))
    assert abs(js_divergence(p, q) - js_divergence(q, p)) < 1e-6


def test_js_bounded():
    p = np.random.dirichlet(np.ones(50))
    q = np.random.dirichlet(np.ones(50))
    result = js_divergence(p, q)
    assert 0.0 <= result <= 1.0


def test_drift_engine_baseline_required(cfg):
    engine = DriftEngine(cfg)
    hists = {
        "node": np.ones(50) / 50,
        "community": np.ones(50) / 50,
        "path": np.ones(50) / 50,
    }
    with pytest.raises(RuntimeError):
        engine.compute(hists)


def test_drift_engine_identical_windows(cfg):
    engine = DriftEngine(cfg)
    hists = {
        "node": np.ones(50) / 50,
        "community": np.ones(50) / 50,
        "path": np.ones(50) / 50,
    }
    engine.set_baseline(hists)
    result = engine.compute(hists)
    assert result["scores"]["node"] < 1e-5
    assert result["scores"]["community"] < 1e-5
