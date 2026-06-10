from __future__ import annotations

from pathlib import Path
import time
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np

from ..data_windowing import flatten_sequences
from ..metrics import evaluate_predictions
from ..utils import format_seconds

try:
    import cupy as cp
except ImportError:  # pragma: no cover
    cp = None


class XGBoostPVModel:
    """XGBoost multi-output day-ahead PV forecasting model."""

    name = "XGBoost"

    def __init__(
        self,
        n_estimators: int = 220,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.9,
        colsample_bytree: float = 0.9,
        max_bin: int = 256,
        random_state: int = 42,
        n_jobs: int = -1,
        device: str = "cpu",
        progress_interval: int = 8,
        early_stopping_rounds: Optional[int] = 30,
    ):
        try:
            from xgboost import XGBRegressor
        except ImportError as exc:
            raise ImportError("请先安装 xgboost：pip install xgboost") from exc
        self.regressor_cls = XGBRegressor
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            max_bin=max_bin,
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self.device = device
        if device.startswith("cuda"):
            self.params["device"] = device
            self.params["sampling_method"] = "gradient_based"
        self.progress_interval = max(1, int(progress_interval))
        self.early_stopping_rounds = early_stopping_rounds
        if early_stopping_rounds is not None and early_stopping_rounds > 0:
            self.params["early_stopping_rounds"] = int(early_stopping_rounds)
        self.models: List[object] = []

    def _build_estimator(self):
        return self.regressor_cls(**self.params)

    def _gpu_predict(self, est, X_flat: np.ndarray) -> np.ndarray:
        if not self.device.startswith("cuda") or cp is None:
            return est.predict(X_flat)
        device_id = int(self.device.split(":")[1]) if ":" in self.device else 0
        with cp.cuda.Device(device_id):
            preds = est.predict(cp.asarray(X_flat))
            return cp.asnumpy(preds)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val=None,
        y_val=None,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None,
    ) -> Dict[str, float]:
        X_train_flat = flatten_sequences(X_train)
        X_val_flat = flatten_sequences(X_val) if X_val is not None else None
        self.models = []
        total_outputs = y_train.shape[1]
        start = time.perf_counter()
        for idx in range(total_outputs):
            est = self._build_estimator()
            fit_kwargs = {}
            if X_val_flat is not None and y_val is not None:
                fit_kwargs["eval_set"] = [(X_val_flat, y_val[:, idx])]
                fit_kwargs["verbose"] = False
            est.fit(X_train_flat, y_train[:, idx], **fit_kwargs)
            self.models.append(est)
            if progress_callback and (
                idx == 0 or idx + 1 == total_outputs or (idx + 1) % self.progress_interval == 0
            ):
                elapsed = time.perf_counter() - start
                per_target = elapsed / float(idx + 1)
                eta = per_target * max(0, total_outputs - idx - 1)
                progress_callback(
                    idx + 1,
                    total_outputs,
                    {
                        "elapsed": elapsed,
                        "eta": eta,
                        "message": f"target {idx + 1}/{total_outputs}, elapsed={format_seconds(elapsed)}, eta={format_seconds(eta)}",
                    },
                )
        train_pred = self.predict(X_train)
        return {f"train_{k}": v for k, v in evaluate_predictions(y_train, train_pred).items()}

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_flat = flatten_sequences(X)
        if not self.models:
            raise RuntimeError("模型尚未训练。")
        preds = [self._gpu_predict(est, X_flat) for est in self.models]
        return np.asarray(np.column_stack(preds), dtype=np.float32)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        return evaluate_predictions(y_test, self.predict(X_test))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "params": self.params,
                "device": self.device,
                "early_stopping_rounds": self.early_stopping_rounds,
                "models": self.models,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostPVModel":
        payload = joblib.load(path)
        params = payload.get("params", {})
        device = payload.get("device", "cpu")
        obj = cls(
            n_estimators=params.get("n_estimators", 1),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            subsample=params.get("subsample", 1.0),
            colsample_bytree=params.get("colsample_bytree", 1.0),
            max_bin=params.get("max_bin", 256),
            random_state=params.get("random_state", 42),
            n_jobs=params.get("n_jobs", -1),
            device=device,
            early_stopping_rounds=payload.get("early_stopping_rounds", 30),
        )
        obj.models = payload["models"]
        return obj
