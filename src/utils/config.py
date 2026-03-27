from pathlib import Path

import yaml


def load_config(path: str = "configs/default.yaml") -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
