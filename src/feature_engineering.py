from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression

from .utils import ensure_dir
from .visualization import plot_correlation_heatmap, plot_feature_scores, plot_pca_variance


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time features from DateTimeIndex."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("特征工程要求 DataFrame 使用 DatetimeIndex。")
    out = df.copy()
    minute_of_day = out.index.hour * 60 + out.index.minute
    out["hour_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    out["hour_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    out["dayofyear_sin"] = np.sin(2 * np.pi * out.index.dayofyear / 366)
    out["dayofyear_cos"] = np.cos(2 * np.pi * out.index.dayofyear / 366)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out.index.dayofweek / 7)
    out["month"] = out.index.month.astype(float)
    return out


def add_lag_rolling_features(
    df: pd.DataFrame,
    target_col: str = "load",
    lags: Iterable[int] = (1, 4, 8, 96, 192),
    rolling_windows: Iterable[int] = (4, 16, 96),
) -> pd.DataFrame:
    """Add load lag and rolling statistics. All features are shifted to avoid leakage."""
    out = df.copy()
    if target_col not in out.columns:
        raise ValueError(f"目标列 {target_col} 不存在。")
    for lag in lags:
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    shifted = out[target_col].shift(1)
    for win in rolling_windows:
        out[f"{target_col}_roll_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
        out[f"{target_col}_roll_std_{win}"] = shifted.rolling(win, min_periods=2).std().fillna(0)
        out[f"{target_col}_roll_max_{win}"] = shifted.rolling(win, min_periods=1).max()
    return out


def infer_exogenous_feature_cols(df: pd.DataFrame, target_col: str = "load") -> List[str]:
    """Pick weather/radiation related columns that are worth lagging."""
    keywords = (
        "radiation",
        "irradiance",
        "weather",
        "wind",
        "humidity",
        "temperature",
        "temp",
    )
    cols: List[str] = []
    for col in df.columns:
        if col == target_col:
            continue
        lower = col.lower()
        if any(key in lower for key in keywords):
            cols.append(col)
    return cols


def add_exogenous_lag_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    lags: Iterable[int] = (1, 4, 96),
    rolling_windows: Iterable[int] = (4, 16),
) -> pd.DataFrame:
    """Add lagged weather/radiation features using only historical values."""
    out = df.copy()
    for col in columns:
        shifted = out[col].shift(1)
        for lag in lags:
            out[f"{col}_lag_{lag}"] = out[col].shift(lag)
        for win in rolling_windows:
            out[f"{col}_roll_mean_{win}"] = shifted.rolling(win, min_periods=1).mean()
            out[f"{col}_roll_std_{win}"] = shifted.rolling(win, min_periods=2).std().fillna(0)
    return out


def create_feature_table(
    base_df: pd.DataFrame,
    target_col: str = "load",
    output_dir: str | Path | None = None,
    enable_pca: bool = True,
    pca_components: int = 8,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """Create final feature table and save feature-engineering plots."""
    out = add_time_features(base_df)
    out = add_lag_rolling_features(out, target_col=target_col)
    exogenous_cols = infer_exogenous_feature_cols(base_df, target_col=target_col)
    out = add_exogenous_lag_features(out, exogenous_cols)
    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    feature_cols = [c for c in out.columns if c != target_col]
    artifacts: Dict[str, object] = {"feature_cols": feature_cols, "exogenous_cols": exogenous_cols}

    if output_dir is not None:
        fig_dir = ensure_dir(Path(output_dir) / "figures")
        plot_correlation_heatmap(out, fig_dir / "05_correlation_heatmap.png", max_cols=25)

        # Mutual information on a subsample for speed.
        sample = out.sample(min(5000, len(out)), random_state=42) if len(out) > 5000 else out
        try:
            scores = mutual_info_regression(sample[feature_cols], sample[target_col], random_state=42)
            score_series = pd.Series(scores, index=feature_cols).sort_values(ascending=False)
            artifacts["mutual_info"] = score_series
            plot_feature_scores(score_series.head(20), fig_dir / "06_feature_mutual_info_top20.png")
        except Exception:
            artifacts["mutual_info"] = None

        if enable_pca and len(feature_cols) >= 2:
            n_components = min(pca_components, len(feature_cols))
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(sample[feature_cols])
            artifacts["pca"] = pca
            plot_pca_variance(pca, fig_dir / "07_pca_explained_variance.png")

    return out, artifacts


def select_top_features(
    feature_table: pd.DataFrame,
    target_col: str,
    k: int | None = None,
    reference_df: pd.DataFrame | None = None,
) -> List[str]:
    """Return k most correlated features by absolute Pearson correlation."""
    feature_cols = [c for c in feature_table.columns if c != target_col]
    if not k or k >= len(feature_cols):
        return feature_cols
    ref = reference_df if reference_df is not None else feature_table
    corr = ref[feature_cols + [target_col]].corr(numeric_only=True)[target_col].drop(target_col)
    return corr.abs().sort_values(ascending=False).head(k).index.tolist()
