from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import ensure_dir


def _save(fig, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_raw_overview(df: pd.DataFrame, time_col: str, target_col: str, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    sample = df[[time_col, target_col]].dropna()
    if len(sample) > 8000:
        sample = sample.iloc[np.linspace(0, len(sample) - 1, 8000).astype(int)]
    ax.plot(sample[time_col], sample[target_col], linewidth=0.7)
    ax.set_title("Raw PV output overview")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_missing_values(df: pd.DataFrame, path: str | Path) -> str:
    miss = df.isna().sum().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(miss))))
    ax.barh(miss.index.astype(str), miss.values)
    ax.set_title("Missing values by column")
    ax.set_xlabel("Count")
    ax.invert_yaxis()
    return _save(fig, path)


def plot_cleaning_effect(raw_resampled: pd.DataFrame, cleaned: pd.DataFrame, target_col: str, path: str | Path) -> str:
    fig, ax = plt.subplots(figsize=(12, 4))
    a = raw_resampled[target_col].copy()
    b = cleaned[target_col].copy()
    if len(a) > 3000:
        idx = np.linspace(0, len(a) - 1, 3000).astype(int)
        a = a.iloc[idx]
        b = b.iloc[idx]
    ax.plot(a.index, a.values, linewidth=0.8, label="Before")
    ax.plot(b.index, b.values, linewidth=0.8, label="After")
    ax.set_title("Target cleaning effect")
    ax.set_xlabel("Time")
    ax.set_ylabel(target_col)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_correlation_heatmap(df: pd.DataFrame, path: str | Path, max_cols: int = 25) -> str:
    numeric = df.select_dtypes(include=[np.number])
    # Keep columns with highest variance to avoid unreadable giant heatmaps.
    if numeric.shape[1] > max_cols:
        cols = numeric.var().sort_values(ascending=False).head(max_cols).index
        numeric = numeric[cols]
    corr = numeric.corr().fillna(0)
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Feature correlation heatmap")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=80, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    return _save(fig, path)


def plot_feature_scores(scores: pd.Series, path: str | Path) -> str:
    scores = scores.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(scores) * 0.32)))
    ax.barh(scores.index.astype(str), scores.values)
    ax.set_title("Top feature scores")
    ax.set_xlabel("Mutual information")
    return _save(fig, path)


def plot_pca_variance(pca, path: str | Path) -> str:
    ratio = np.asarray(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(1, len(ratio) + 1), ratio)
    ax.plot(np.arange(1, len(ratio) + 1), np.cumsum(ratio), marker="o", label="Cumulative")
    ax.set_title("PCA explained variance")
    ax.set_xlabel("Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_ylim(0, min(1.05, max(0.2, np.cumsum(ratio)[-1] + 0.05)))
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_prediction_curve(y_true, y_pred, path: str | Path, title: str = "Day-ahead prediction curve") -> str:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(yt, marker="o", markersize=2, linewidth=1.0, label="Actual")
    ax.plot(np.clip(yp, 0, 1), marker="o", markersize=2, linewidth=1.0, label="Predicted")
    ax.set_title(title)
    ax.set_xlabel("15-min point index")
    ax.set_ylabel("Normalized PV output")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_error_distribution(y_true, y_pred, path: str | Path, title: str = "Prediction error distribution") -> str:
    err = np.asarray(y_pred).reshape(-1) - np.asarray(y_true).reshape(-1)
    err = err[np.isfinite(err)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(err, bins=40)
    ax.set_title(title)
    ax.set_xlabel("Prediction error")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_scatter(y_true, y_pred, path: str | Path, title: str = "Actual vs predicted") -> str:
    yt = np.asarray(y_true).reshape(-1)
    yp = np.asarray(y_pred).reshape(-1)
    if len(yt) > 5000:
        idx = np.linspace(0, len(yt) - 1, 5000).astype(int)
        yt, yp = yt[idx], yp[idx]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, np.clip(yp, 0, 1), s=5, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_metrics_comparison(metrics_df: pd.DataFrame, path: str | Path) -> str:
    if metrics_df.empty:
        return ""
    primary_cols = [c for c in ["nRMSE(%)", "nMAE(%)"] if c in metrics_df.columns]
    secondary_col = "R2" if "R2" in metrics_df.columns else None
    x = np.arange(len(metrics_df))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, col in enumerate(primary_cols):
        vals = pd.to_numeric(metrics_df[col], errors="coerce").values
        ax.bar(x + (i - len(primary_cols) / 2) * width + width / 2, vals, width, label=col)
    labels = [f"{r['model']}\nM={r.get('memory_length', '')}" for _, r in metrics_df.iterrows()]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=8)
    ax.set_title("Model metrics comparison")
    ax.set_ylabel("Error (%)")
    ax.grid(True, axis="y", alpha=0.3)
    handles, labels_ = ax.get_legend_handles_labels()
    if secondary_col:
        ax2 = ax.twinx()
        vals = pd.to_numeric(metrics_df[secondary_col], errors="coerce").values
        line = ax2.plot(x, vals, color="black", marker="o", linewidth=1.5, label=secondary_col)
        ax2.set_ylabel("R2")
        ax2.set_ylim(0, max(1.0, np.nanmax(vals) * 1.05 if len(vals) else 1.0))
        handles += line
        labels_ += [secondary_col]
    ax.legend(handles, labels_, loc="best")
    return _save(fig, path)


def plot_training_loss(history: Iterable[dict] | pd.DataFrame, path: str | Path, title: str = "Training curve") -> str:
    df = pd.DataFrame(history)
    if df.empty or "epoch" not in df.columns:
        return ""
    fig, ax = plt.subplots(figsize=(8, 4))
    if "train_loss" in df.columns:
        ax.plot(df["epoch"], df["train_loss"], marker="o", label="Train")
    if "val_loss" in df.columns:
        ax.plot(df["epoch"], df["val_loss"], marker="o", label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_history_dashboard(history_df: pd.DataFrame, path: str | Path) -> str:
    if history_df.empty or "model" not in history_df.columns:
        return ""
    latest = history_df.copy()
    latest["run_time"] = latest.get("run_time", "")
    return plot_metrics_comparison(latest.tail(20), path)
