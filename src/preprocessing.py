from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .utils import ensure_dir
from .visualization import plot_cleaning_effect, plot_missing_values, plot_raw_overview

DEFAULT_TIME_COL = "time"
DEFAULT_TARGET_COL = "load"


@dataclass
class PreprocessResult:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    scaled: pd.DataFrame
    feature_scaler: object
    target_scaler: object
    numeric_columns: List[str]
    report: Dict[str, float | int | str]


def infer_columns(df: pd.DataFrame, time_col: Optional[str] = None, target_col: Optional[str] = None) -> Tuple[str, str]:
    """Infer time and target columns using robust defaults."""
    if time_col is None:
        candidates = [c for c in df.columns if c.lower() in {"time", "timestamp", "date", "datetime"}]
        time_col = candidates[0] if candidates else df.columns[0]
    if target_col is None:
        candidates = [c for c in df.columns if c.lower() in {"load", "power", "pv_power", "active_power"}]
        target_col = candidates[0] if candidates else df.columns[-1]
    return time_col, target_col


def load_raw_data(csv_path: str | Path, time_col: Optional[str] = None, target_col: Optional[str] = None) -> Tuple[pd.DataFrame, str, str]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"数据文件不存在：{csv_path}")
    df = pd.read_csv(csv_path)
    time_col, target_col = infer_columns(df, time_col, target_col)
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(subset=[time_col])
    return df.reset_index(drop=True), time_col, target_col


def resample_to_interval(
    df: pd.DataFrame,
    time_col: str,
    freq: str = "15min",
) -> pd.DataFrame:
    """Resample irregular or high-frequency data to the required interval."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("没有可用于建模的数值列。")
    resampled = (
        df.set_index(time_col)[numeric_cols]
        .resample(freq)
        .mean()
        .sort_index()
    )
    return resampled


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing timestamps/features using time interpolation plus safe fallbacks."""
    out = df.copy()
    out = out.interpolate(method="time", limit_direction="both")
    out = out.ffill().bfill()
    return out


def clip_outliers_iqr(df: pd.DataFrame, columns: List[str], target_col: str, factor: float = 3.0) -> pd.DataFrame:
    """Clip outliers by IQR and clamp negative PV output to zero."""
    out = df.copy()
    for col in columns:
        series = out[col]
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if np.isfinite(iqr) and iqr > 0:
            lower, upper = q1 - factor * iqr, q3 + factor * iqr
            out[col] = series.clip(lower, upper)
    if target_col in out.columns:
        out[target_col] = out[target_col].clip(lower=0)
    return out


def scale_dataframe(
    df: pd.DataFrame,
    target_col: str,
    scaler_type: str = "minmax",
) -> Tuple[pd.DataFrame, object, object]:
    """Scale feature columns and target column. Metrics are computed on scaled load."""
    if target_col not in df.columns:
        raise ValueError(f"目标列 {target_col} 不存在。")
    feature_cols = [c for c in df.columns if c != target_col]
    scaler_cls = MinMaxScaler if scaler_type == "minmax" else StandardScaler
    feature_scaler = scaler_cls()
    target_scaler = MinMaxScaler()  # target must be in [0, 1] for nRMSE/nMAE by task definition.
    scaled = df.copy()
    if feature_cols:
        scaled[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
    scaled[[target_col]] = target_scaler.fit_transform(df[[target_col]])
    return scaled, feature_scaler, target_scaler


def run_preprocessing(
    csv_path: str | Path,
    output_dir: str | Path,
    time_col: Optional[str] = None,
    target_col: Optional[str] = None,
    freq: str = "15min",
    scaler_type: str = "minmax",
    save_processed: bool = True,
) -> PreprocessResult:
    """Full preprocessing pipeline with required figures."""
    output_dir = ensure_dir(output_dir)
    fig_dir = ensure_dir(Path(output_dir) / "figures")
    scaler_dir = ensure_dir(Path(output_dir) / "scalers")

    raw, time_col, target_col = load_raw_data(csv_path, time_col, target_col)
    plot_raw_overview(raw, time_col, target_col, fig_dir / "01_raw_load_overview.png")
    plot_missing_values(raw, fig_dir / "02_missing_values_raw.png")

    resampled = resample_to_interval(raw, time_col=time_col, freq=freq)
    numeric_columns = resampled.columns.tolist()
    missing_before = int(resampled.isna().sum().sum())
    negatives_before = int((resampled[target_col] < 0).sum()) if target_col in resampled.columns else 0

    filled = fill_missing_values(resampled)
    cleaned = clip_outliers_iqr(filled, numeric_columns, target_col)
    missing_after = int(cleaned.isna().sum().sum())
    negatives_after = int((cleaned[target_col] < 0).sum()) if target_col in cleaned.columns else 0

    plot_cleaning_effect(resampled, cleaned, target_col, fig_dir / "03_cleaning_before_after.png")
    plot_missing_values(cleaned.reset_index(), fig_dir / "04_missing_values_cleaned.png")

    scaled, feature_scaler, target_scaler = scale_dataframe(cleaned, target_col=target_col, scaler_type=scaler_type)
    if save_processed:
        ensure_dir(Path(output_dir) / "processed")
        cleaned.to_csv(Path(output_dir) / "processed" / "cleaned_15min.csv")
        scaled.to_csv(Path(output_dir) / "processed" / "scaled_15min.csv")
    joblib.dump(feature_scaler, scaler_dir / "feature_scaler.joblib")
    joblib.dump(target_scaler, scaler_dir / "target_scaler.joblib")

    report = {
        "time_col": time_col,
        "target_col": target_col,
        "raw_rows": int(len(raw)),
        "resampled_rows": int(len(resampled)),
        "freq": freq,
        "missing_before": missing_before,
        "missing_after": missing_after,
        "negative_target_before": negatives_before,
        "negative_target_after": negatives_after,
        "start_time": str(cleaned.index.min()),
        "end_time": str(cleaned.index.max()),
    }

    return PreprocessResult(
        raw=raw,
        cleaned=cleaned,
        scaled=scaled,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        numeric_columns=numeric_columns,
        report=report,
    )
