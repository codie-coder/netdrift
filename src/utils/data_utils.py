from pathlib import Path
import json
from typing import Any


def save_json(data: Any, path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(Path(path), "r") as f:
        return json.load(f)


def ensure_dirs(dirs: list) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
