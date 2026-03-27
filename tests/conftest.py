import numpy as np
import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.seed import fix_seeds


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture(scope="session")
def sample_df():
    fix_seeds(42)
    np.random.seed(42)
    n = 300
    return pd.DataFrame({
        "src_ip":    [f"192.168.1.{np.random.randint(1,30)}" for _ in range(n)],
        "dst_ip":    [f"10.0.0.{np.random.randint(1,15)}" for _ in range(n)],
        "timestamp": np.sort(np.random.uniform(0, 300, n)),
        "bytes":     np.random.randint(100, 50000, n),
    })
