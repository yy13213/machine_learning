from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def save_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def resolve_training_device(device: str = "auto", gpu_id: int = 0) -> str:
    value = (device or "auto").strip().lower()
    if value in {"cpu"}:
        return "cpu"
    if value in {"gpu", "cuda"}:
        return f"cuda:{gpu_id}"
    if value.startswith("gpu"):
        suffix = value[3:]
        return f"cuda:{suffix}" if suffix else f"cuda:{gpu_id}"
    if value.startswith("cuda"):
        if value == "cuda":
            return f"cuda:{gpu_id}"
        return value
    if value == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                return f"cuda:{gpu_id}"
        except Exception:
            pass
        return "cpu"
    return value


class ProgressTimer:
    def __init__(self) -> None:
        self.start_time = time.perf_counter()

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time
