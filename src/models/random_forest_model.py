from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from ..data_windowing import flatten_sequences
from ..metrics import evaluate_predictions


class RandomForestPVModel:
    """Random Forest multi-output day-ahead PV forecasting model."""

    name = "RandomForest"

    def __init__(
        self,
        n_estimators: int = 120,
        max_depth: Optional[int] = 18,
        random_state: int = 42,
        n_jobs: int = -1,
        verbose: int = 0,
    ):
        base = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs,
            min_samples_leaf=2,
            verbose=verbose,
        )
        self.model = base

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val=None,
        y_val=None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        if progress_callback:
            progress_callback(0, 1, {"stage": "fit_start"})
        self.model.fit(flatten_sequences(X_train), y_train)
        train_pred = self.predict(X_train)
        metrics = {f"train_{k}": v for k, v in evaluate_predictions(y_train, train_pred).items()}
        if progress_callback:
            progress_callback(1, 1, metrics)
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(flatten_sequences(X)), dtype=np.float32)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        return evaluate_predictions(y_test, self.predict(X_test))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestPVModel":
        obj = cls()
        obj.model = joblib.load(path)
        return obj
