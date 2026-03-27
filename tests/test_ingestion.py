import numpy as np
import pandas as pd
import pytest

from src.ingestion.windower import sliding_windows


def test_windower_yields_correct_count(sample_df, cfg):
    windows = list(sliding_windows(
        sample_df,
        window_size_sec=cfg["windowing"]["window_size_sec"],
        step_size_sec=cfg["windowing"]["step_size_sec"],
        min_edges=cfg["windowing"]["min_edges"],
    ))
    assert len(windows) > 0


def test_windower_empty_df():
    df = pd.DataFrame(columns=["src_ip", "dst_ip", "timestamp", "bytes"])
    windows = list(sliding_windows(df, 300, 60, 10))
    assert windows == []


def test_windower_window_ids_sequential(sample_df, cfg):
    windows = list(sliding_windows(
        sample_df,
        window_size_sec=cfg["windowing"]["window_size_sec"],
        step_size_sec=cfg["windowing"]["step_size_sec"],
        min_edges=cfg["windowing"]["min_edges"],
    ))
    ids = [w[0] for w in windows]
    assert ids == sorted(ids)


def test_windower_min_edges_respected(sample_df, cfg):
    windows = list(sliding_windows(sample_df, 300, 60, min_edges=99999))
    assert windows == []
