from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def make_windows(
    feature_table: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "load",
    memory_length: int = 96,
    horizon: int = 96,
    stride: int = 1,
    max_windows: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sequence-to-sequence samples: past memory_length -> future horizon."""
    if memory_length <= 0 or horizon <= 0:
        raise ValueError("memory_length 和 horizon 必须为正数。")
    if len(feature_table) < memory_length + horizon + 1:
        raise ValueError(
            f"数据量不足，至少需要 {memory_length + horizon + 1} 行，当前 {len(feature_table)} 行。"
        )
    arr_x = feature_table[feature_cols].to_numpy(dtype=np.float32)
    arr_y = feature_table[target_col].to_numpy(dtype=np.float32)
    times = feature_table.index.to_numpy()

    starts = np.arange(0, len(feature_table) - memory_length - horizon + 1, stride, dtype=int)
    if max_windows is not None and len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = starts[idx]

    X = np.empty((len(starts), memory_length, len(feature_cols)), dtype=np.float32)
    y = np.empty((len(starts), horizon), dtype=np.float32)
    pred_start_times = np.empty((len(starts),), dtype=times.dtype)
    for i, start in enumerate(starts):
        x_start, x_end = start, start + memory_length
        y_end = x_end + horizon
        X[i] = arr_x[x_start:x_end]
        y[i] = arr_y[x_end:y_end]
        pred_start_times[i] = times[x_end]
    return X, y, pred_start_times


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Split without shuffling to avoid time-series leakage."""
    n = len(X)
    if n < 10:
        raise ValueError("窗口样本太少，无法划分训练/验证/测试集。")
    test_n = max(1, int(n * test_size))
    val_n = max(1, int(n * val_size))
    train_n = n - test_n - val_n
    if train_n <= 0:
        raise ValueError("训练集大小不足，请减少 test_size/val_size 或增加 max_windows。")
    return {
        "X_train": X[:train_n],
        "y_train": y[:train_n],
        "time_train": times[:train_n],
        "X_val": X[train_n:train_n + val_n],
        "y_val": y[train_n:train_n + val_n],
        "time_val": times[train_n:train_n + val_n],
        "X_test": X[train_n + val_n:],
        "y_test": y[train_n + val_n:],
        "time_test": times[train_n + val_n:],
    }


def flatten_sequences(X: np.ndarray) -> np.ndarray:
    return X.reshape(X.shape[0], -1)
