from typing import Generator, Tuple

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)


def sliding_windows(
    df: pd.DataFrame,
    window_size_sec: int,
    step_size_sec: int,
    min_edges: int = 10,
) -> Generator[Tuple[int, float, float, pd.DataFrame], None, None]:
    if df.empty:
        log.warning("empty dataframe passed to windower")
        return

    t_min = df["timestamp"].min()
    t_max = df["timestamp"].max()
    window_id = 0
    t_start = t_min

    while t_start < t_max:
        t_end = t_start + window_size_sec
        mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)
        window_df = df[mask].copy()

        if len(window_df) < min_edges:
            log.warning(
                "window too sparse, skipping",
                window_id=window_id,
                edges=len(window_df),
                min_edges=min_edges,
            )
        else:
            yield window_id, t_start, t_end, window_df

        window_id += 1
        t_start += step_size_sec
