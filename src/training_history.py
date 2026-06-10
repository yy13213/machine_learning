from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .utils import ensure_dir


def append_history(
    history_path: str | Path,
    record: Dict[str, object],
) -> pd.DataFrame:
    history_path = Path(history_path)
    ensure_dir(history_path.parent)
    df_new = pd.DataFrame([record])
    if history_path.exists():
        old = pd.read_csv(history_path)
        hist = pd.concat([old, df_new], ignore_index=True)
    else:
        hist = df_new
    hist.to_csv(history_path, index=False)
    return hist


def load_history(history_path: str | Path) -> pd.DataFrame:
    path = Path(history_path)
    if not path.exists():
        return pd.DataFrame()
    return normalize_history(pd.read_csv(path), source=str(path.parent))


def normalize_history(df: pd.DataFrame, source: str | None = None) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "R2" not in out.columns and "R2(%)" in out.columns:
        out["R2"] = pd.to_numeric(out["R2(%)"], errors="coerce") / 100.0
    if "R2(%)" not in out.columns and "R2" in out.columns:
        out["R2(%)"] = pd.to_numeric(out["R2"], errors="coerce") * 100.0
    if "run_time" in out.columns:
        out["run_time"] = pd.to_datetime(out["run_time"], errors="coerce")
    for col in ["memory_length", "horizon", "n_train", "n_val", "n_test", "n_features"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    if "quick" in out.columns:
        out["quick"] = out["quick"].astype(str).str.lower().map({"true": True, "false": False}).fillna(out["quick"])
    if source is not None:
        out["source"] = source
    return out


def load_all_histories(root_dir: str | Path) -> pd.DataFrame:
    root = Path(root_dir)
    frames: List[pd.DataFrame] = []
    for path in sorted(root.glob("**/history/training_history.csv")):
        df = load_history(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates()
    if "run_time" in merged.columns:
        merged = merged.sort_values("run_time")
    return merged.reset_index(drop=True)


def summarize_recent_experiments(history_df: pd.DataFrame) -> List[str]:
    if history_df.empty:
        return ["暂无实验记录。"]
    df = history_df.copy()
    df = normalize_history(df)
    quick_series = df["quick"] if "quick" in df.columns else False
    horizon_series = df["horizon"] if "horizon" in df.columns else 0
    full = df[(quick_series == False) & (horizon_series == 96)]  # noqa: E712
    if full.empty:
        full = df
    full = full.sort_values("run_time").tail(12)
    insights: List[str] = []

    valid = full.dropna(subset=["nRMSE(%)", "nMAE(%)", "R2"])
    if not valid.empty:
        best = valid.sort_values(["nRMSE(%)", "nMAE(%)", "R2"], ascending=[True, True, False]).iloc[0]
        insights.append(
            f"最近记录里表现最好的模型是 {best['model']} (M={int(best['memory_length'])})，"
            f"nRMSE={best['nRMSE(%)']:.3f}%，nMAE={best['nMAE(%)']:.3f}%，R2={best['R2']:.4f}。"
        )

    for model_name in ["RandomForest", "XGBoost", "Transformer", "LSTM"]:
        sub = valid[valid["model"] == model_name]
        if len(sub) < 2:
            continue
        by_mem = sub.sort_values("run_time").drop_duplicates(subset=["memory_length"], keep="last")
        if {96, 192}.issubset(set(by_mem["memory_length"].dropna().astype(int).tolist())):
            m96 = by_mem[by_mem["memory_length"] == 96].iloc[-1]
            m192 = by_mem[by_mem["memory_length"] == 192].iloc[-1]
            delta = m192["nRMSE(%)"] - m96["nRMSE(%)"]
            direction = "更优" if delta < 0 else "略差"
            insights.append(
                f"{model_name} 在 memory_length=192 相比 96 的 nRMSE 变化为 {delta:+.3f} 个百分点，说明 192 {direction}。"
            )

    tree = valid[valid["model"].isin(["RandomForest", "XGBoost"])]
    deep = valid[valid["model"].isin(["LSTM", "Transformer"])]
    if not tree.empty and not deep.empty:
        best_tree = tree.sort_values("nRMSE(%)").iloc[0]
        best_deep = deep.sort_values("nRMSE(%)").iloc[0]
        insights.append(
            f"当前树模型仍领先深度模型：最佳树模型 {best_tree['model']} 的 nRMSE={best_tree['nRMSE(%)']:.3f}%，"
            f"优于最佳深度模型 {best_deep['model']} 的 {best_deep['nRMSE(%)']:.3f}%。"
        )

    if "n_features" in valid.columns:
        latest_by_setting = valid.sort_values("run_time").drop_duplicates(subset=["model", "memory_length", "n_features"], keep="last")
        if latest_by_setting["n_features"].nunique() > 1:
            compact = latest_by_setting.groupby("n_features")["nRMSE(%)"].mean().sort_index()
            if len(compact) >= 2:
                best_nf = compact.idxmin()
                insights.append(f"从最近记录看，特征数控制在 {int(best_nf)} 左右时平均 nRMSE 更低，说明继续做特征裁剪仍有价值。")

    return insights[:6]
