from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_arrays(y_true, y_pred):
    yt = np.asarray(y_true, dtype=np.float64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(yt) & np.isfinite(yp)
    return yt[mask], yp[mask]


def evaluate_predictions(y_true_scaled, y_pred_scaled) -> Dict[str, float]:
    """Compute nRMSE(%), nMAE(%), R2 on normalized target values."""
    yt, yp = _safe_arrays(y_true_scaled, y_pred_scaled)
    yp = np.clip(yp, 0.0, 1.0)
    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    try:
        r2 = float(r2_score(yt, yp))
    except Exception:
        r2 = float("nan")
    return {
        "nRMSE(%)": rmse * 100.0,
        "nMAE(%)": mae * 100.0,
        "R2": r2,
        "RMSE_scaled": rmse,
        "MAE_scaled": mae,
    }
