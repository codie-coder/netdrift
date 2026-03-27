import ipaddress
from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

log = get_logger(__name__)

REQUIRED_COLS = ["src_ip", "dst_ip", "timestamp", "bytes"]


def _validate_ip(ip: str) -> str:
    return str(ipaddress.ip_address(str(ip).strip()))


def load_csv(path: str, cfg: dict) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    log.info("loading CSV", path=str(p))

    col_map = {
        cfg["ingestion"]["src_col"]:       "src_ip",
        cfg["ingestion"]["dst_col"]:       "dst_ip",
        cfg["ingestion"]["timestamp_col"]: "timestamp",
        cfg["ingestion"]["bytes_col"]:     "bytes",
    }

    df = pd.read_csv(p)
    df = df.rename(columns=col_map)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS)
    dropped = before - len(df)
    if dropped:
        log.warning("dropped null rows", count=dropped)

    try:
        df["src_ip"] = df["src_ip"].apply(_validate_ip)
        df["dst_ip"] = df["dst_ip"].apply(_validate_ip)
    except ValueError as e:
        raise ValueError(f"Invalid IP address found: {e}")

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].astype(float)
    df["bytes"] = pd.to_numeric(df["bytes"], errors="coerce").fillna(0).astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    log.info("CSV loaded", rows=len(df), cols=list(df.columns))
    return df
