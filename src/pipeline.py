from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .data_windowing import chronological_split, make_windows
from .feature_engineering import create_feature_table, select_top_features
from .preprocessing import run_preprocessing
from .training_history import append_history, load_history
from .utils import ensure_dir, format_seconds, resolve_training_device, save_json, set_seed
from .visualization import (
    plot_error_distribution,
    plot_history_dashboard,
    plot_metrics_comparison,
    plot_prediction_curve,
    plot_scatter,
    plot_training_loss,
)
from .models import LSTMPVModel, RandomForestPVModel, SAMFormerPVModel, TransformerPVModel, XGBoostPVModel

MODEL_ALIASES = {
    "rf": "random_forest",
    "random_forest": "random_forest",
    "xgb": "xgboost",
    "xgboost": "xgboost",
    "lstm": "lstm",
    "transformer": "transformer",
    "samformer": "samformer",
}


def build_model(
    name: str,
    input_dim: int,
    memory_length: int,
    horizon: int,
    quick: bool = False,
    device: str = "cpu",
    show_progress: bool = True,
    early_stopping_patience: int = 3,
    model_params: Optional[Dict[str, object]] = None,
):
    name = MODEL_ALIASES.get(name.lower(), name.lower())
    params = dict(model_params or {})
    if name == "random_forest":
        return RandomForestPVModel(
            n_estimators=params.pop("n_estimators", 20 if quick else 120),
            max_depth=params.pop("max_depth", 8 if quick else 18),
            verbose=1 if show_progress else 0,
            **params,
        )
    if name == "xgboost":
        return XGBoostPVModel(
            n_estimators=params.pop("n_estimators", 30 if quick else 220),
            max_depth=params.pop("max_depth", 3 if quick else 5),
            learning_rate=params.pop("learning_rate", 0.08 if quick else 0.05),
            device=device,
            **params,
        )
    if name == "lstm":
        return LSTMPVModel(
            input_dim=input_dim,
            horizon=horizon,
            hidden_dim=params.pop("hidden_dim", 32 if quick else 96),
            num_layers=params.pop("num_layers", 1 if quick else 2),
            epochs=params.pop("epochs", 1 if quick else 10),
            batch_size=params.pop("batch_size", 32 if quick else 64),
            patience=params.pop("patience", early_stopping_patience),
            device=device,
            **params,
        )
    if name == "transformer":
        return TransformerPVModel(
            input_dim=input_dim,
            memory_length=memory_length,
            horizon=horizon,
            d_model=params.pop("d_model", 32 if quick else 96),
            nhead=params.pop("nhead", 4),
            num_layers=params.pop("num_layers", 1 if quick else 2),
            epochs=params.pop("epochs", 1 if quick else 10),
            batch_size=params.pop("batch_size", 32 if quick else 64),
            patience=params.pop("patience", early_stopping_patience),
            device=device,
            **params,
        )
    if name == "samformer":
        return SAMFormerPVModel(
            input_dim=input_dim,
            memory_length=memory_length,
            horizon=horizon,
            hid_dim=params.pop("hid_dim", 16 if quick else 24),
            dropout=params.pop("dropout", 0.1),
            lr=params.pop("lr", 1e-3),
            weight_decay=params.pop("weight_decay", 1e-5),
            rho=params.pop("rho", 0.5),
            batch_size=params.pop("batch_size", 32 if quick else 64),
            epochs=params.pop("epochs", 2 if quick else 12),
            patience=params.pop("patience", early_stopping_patience),
            use_revin=params.pop("use_revin", True),
            device=device,
            **params,
        )
    raise ValueError(f"未知模型：{name}")


def select_midnight_curve_index(times: np.ndarray) -> int:
    if len(times) == 0:
        return 0
    dt_index = pd.to_datetime(times)
    midnight_idx = np.where((dt_index.hour == 0) & (dt_index.minute == 0))[0]
    return int(midnight_idx[0]) if len(midnight_idx) else 0


def default_progress_printer(message: str) -> None:
    print(message, flush=True)


def maybe_reduce_sequence_features(
    splits: Dict[str, np.ndarray],
    output_dir: str | Path,
    n_components: Optional[int],
    file_stem: str = "feature_pca",
) -> tuple[Dict[str, np.ndarray], Optional[PCA]]:
    if n_components is None:
        return splits, None
    X_train = splits["X_train"]
    input_dim = X_train.shape[-1]
    if n_components <= 0 or n_components >= input_dim:
        return splits, None

    pca = PCA(n_components=n_components, random_state=42)
    pca.fit(X_train.reshape(-1, input_dim))
    reduced = dict(splits)
    for key in ("X_train", "X_val", "X_test"):
        arr = splits[key]
        reduced[key] = pca.transform(arr.reshape(-1, input_dim)).reshape(arr.shape[0], arr.shape[1], n_components).astype(np.float32)

    try:
        import joblib

        scaler_dir = ensure_dir(Path(output_dir) / "scalers")
        joblib.dump(pca, scaler_dir / f"{file_stem}.joblib")
    except Exception as exc:
        print(f"[WARN] PCA 保存失败: {exc}")
    return reduced, pca


def prepare_experiment_data(
    data_path: str | Path,
    output_dir: str | Path,
    memory_length: int,
    horizon: int,
    stride: int,
    max_windows: int,
    top_k_features: Optional[int],
    quick: bool,
    target_col: str,
    time_col: str,
) -> Dict[str, object]:
    output_dir = ensure_dir(output_dir)
    pre = run_preprocessing(
        data_path,
        output_dir=output_dir,
        time_col=time_col,
        target_col=target_col,
        freq="15min",
        save_processed=not quick,
    )
    save_json(pre.report, Path(output_dir) / "preprocess_report.json")

    feature_table, feature_artifacts = create_feature_table(pre.cleaned, target_col=target_col, output_dir=output_dir)
    reference_rows = max(1, int(len(feature_table) * 0.7))
    feature_cols = select_top_features(
        feature_table,
        target_col=target_col,
        k=top_k_features,
        reference_df=feature_table.iloc[:reference_rows],
    )
    save_json(
        {
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "exogenous_cols": feature_artifacts.get("exogenous_cols", []),
        },
        Path(output_dir) / "feature_report.json",
    )

    X, y, times = make_windows(
        feature_table,
        feature_cols=feature_cols,
        target_col=target_col,
        memory_length=memory_length,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
    )
    raw_splits = chronological_split(X, y, times, test_size=0.2, val_size=0.1)
    splits = scale_window_splits(raw_splits, output_dir=output_dir)
    return {
        "preprocess_report": pre.report,
        "feature_cols": feature_cols,
        "feature_artifacts": feature_artifacts,
        "splits": splits,
    }


def scale_window_splits(
    splits: Dict[str, np.ndarray],
    output_dir: str | Path,
) -> Dict[str, np.ndarray]:
    """Fit scalers on train windows only to avoid temporal leakage."""
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    n_features = X_train.shape[-1]

    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(X_train.reshape(-1, n_features))
    target_scaler.fit(y_train.reshape(-1, 1))

    scaled = dict(splits)
    for key in ("X_train", "X_val", "X_test"):
        arr = splits[key]
        scaled[key] = feature_scaler.transform(arr.reshape(-1, n_features)).reshape(arr.shape).astype(np.float32)
    for key in ("y_train", "y_val", "y_test"):
        arr = splits[key]
        scaled[key] = target_scaler.transform(arr.reshape(-1, 1)).reshape(arr.shape).astype(np.float32)

    scaler_dir = ensure_dir(Path(output_dir) / "scalers")
    try:
        import joblib

        joblib.dump(feature_scaler, scaler_dir / "feature_scaler.joblib")
        joblib.dump(target_scaler, scaler_dir / "target_scaler.joblib")
    except Exception as exc:
        print(f"[WARN] scaler 保存失败: {exc}")
    return scaled


def run_experiment(
    data_path: str | Path,
    output_dir: str | Path = "results",
    models: Iterable[str] = ("random_forest", "xgboost", "lstm", "transformer"),
    memory_length: int = 96,
    horizon: int = 96,
    stride: int = 4,
    max_windows: int = 5000,
    top_k_features: Optional[int] = None,
    quick: bool = False,
    target_col: str = "load",
    time_col: str = "time",
    device: str = "auto",
    gpu_id: int = 0,
    show_progress: bool = True,
    progress_callback: Optional[Callable[[str], None]] = None,
    feature_pca_dim: Optional[int] = None,
    deep_feature_dim: Optional[int] = None,
    early_stopping_patience: int = 3,
    model_params: Optional[Dict[str, Dict[str, object]]] = None,
) -> Dict[str, object]:
    """End-to-end experiment used by CLI and Streamlit."""
    set_seed(42)
    model_names = list(models)
    progress = progress_callback or default_progress_printer
    resolved_device = resolve_training_device(device=device, gpu_id=gpu_id)
    experiment_start = perf_counter()

    def log(message: str) -> None:
        if show_progress:
            progress(message)

    output_dir = ensure_dir(output_dir)
    fig_dir = ensure_dir(Path(output_dir) / "figures")
    model_dir = ensure_dir(Path(output_dir) / "models")
    history_path = Path(output_dir) / "history" / "training_history.csv"

    log(f"[Init] device={resolved_device}, models={model_names}, memory_length={memory_length}, horizon={horizon}")

    stage_start = perf_counter()
    prepared = prepare_experiment_data(
        data_path=data_path,
        output_dir=output_dir,
        memory_length=memory_length,
        horizon=horizon,
        stride=stride,
        max_windows=max_windows,
        top_k_features=top_k_features,
        quick=quick,
        target_col=target_col,
        time_col=time_col,
    )
    pre_report = prepared["preprocess_report"]
    feature_cols = prepared["feature_cols"]
    splits = prepared["splits"]
    log(f"[1/5] 预处理完成，用时 {format_seconds(perf_counter() - stage_start)}")
    stage_start = perf_counter()
    feature_artifacts = prepared["feature_artifacts"]
    feature_report = Path(output_dir) / "feature_report.json"
    feature_meta = {
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "device": resolved_device,
        "exogenous_cols": feature_artifacts.get("exogenous_cols", []),
    }
    save_json(feature_meta, feature_report)
    log(f"[2/5] 特征工程完成，用时 {format_seconds(perf_counter() - stage_start)}，n_features={len(feature_cols)}")

    stage_start = perf_counter()
    log(
        f"[3/5] 窗口构建完成，用时 {format_seconds(perf_counter() - stage_start)}，"
        f"train/val/test={len(splits['X_train'])}/{len(splits['X_val'])}/{len(splits['X_test'])}"
    )

    all_metrics: List[Dict[str, object]] = []
    predictions: Dict[str, np.ndarray] = {}
    for model_idx, raw_name in enumerate(model_names, start=1):
        canonical = MODEL_ALIASES.get(raw_name.lower(), raw_name.lower())
        current_splits = splits
        input_dim = current_splits["X_train"].shape[-1]
        pca_used = None
        target_pca_dim = feature_pca_dim
        if target_pca_dim is None and canonical in {"lstm", "transformer", "samformer"}:
            target_pca_dim = deep_feature_dim
        if target_pca_dim is not None:
            current_splits, pca_used = maybe_reduce_sequence_features(
                splits,
                output_dir=output_dir,
                n_components=target_pca_dim,
                file_stem=f"{canonical}_feature_pca",
            )
            input_dim = current_splits["X_train"].shape[-1]
            if pca_used is not None:
                log(f"    [{canonical}] PCA 降维: {splits['X_train'].shape[-1]} -> {input_dim}")
        model = build_model(
            canonical,
            input_dim=input_dim,
            memory_length=memory_length,
            horizon=horizon,
            quick=quick,
            device=resolved_device,
            show_progress=show_progress,
            early_stopping_patience=early_stopping_patience,
            model_params=(model_params or {}).get(canonical),
        )

        model_start = perf_counter()
        log(f"[4/5][{model_idx}/{len(model_names)}] 开始训练 {model.name}，device={resolved_device}")

        def model_progress(current: int, total: int, payload: Dict[str, float]) -> None:
            if not show_progress:
                return
            elapsed = perf_counter() - model_start
            eta = (elapsed / max(current, 1)) * max(0, total - current) if current else 0.0
            details = []
            for key in ("train_loss", "val_loss"):
                if key in payload:
                    details.append(f"{key}={payload[key]:.6f}")
            if "message" in payload:
                details.append(str(payload["message"]))
            suffix = f" | {'; '.join(details)}" if details else ""
            log(
                f"    [{model.name}] progress {current}/{total} | "
                f"elapsed={format_seconds(elapsed)} | eta={format_seconds(eta)}{suffix}"
            )

        fit_info = model.fit(
            current_splits["X_train"],
            current_splits["y_train"],
            current_splits["X_val"],
            current_splits["y_val"],
            progress_callback=model_progress,
        )
        y_pred = model.predict(current_splits["X_test"])
        metrics = model.evaluate(current_splits["X_test"], current_splits["y_test"])
        record = {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model.name,
            "memory_length": memory_length,
            "horizon": horizon,
            "n_train": int(len(current_splits["X_train"])),
            "n_val": int(len(current_splits["X_val"])),
            "n_test": int(len(current_splits["X_test"])),
            "n_features": int(input_dim),
            "feature_pca_dim": int(input_dim) if pca_used is not None else None,
            "quick": bool(quick),
            "device": resolved_device,
            **metrics,
        }
        all_metrics.append(record)
        append_history(history_path, record)
        predictions[model.name] = y_pred
        log(
            f"[{model_idx}/{len(model_names)}] {model.name} 完成，用时 {format_seconds(perf_counter() - model_start)} | "
            f"nRMSE={metrics['nRMSE(%)']:.4f}% | nMAE={metrics['nMAE(%)']:.4f}% | R2={metrics['R2']:.4f}"
        )

        suffix = model.name.lower()
        curve_idx = select_midnight_curve_index(current_splits["time_test"])
        curve_time = pd.to_datetime(current_splits["time_test"][curve_idx]).strftime("%Y-%m-%d %H:%M")
        plot_prediction_curve(
            current_splits["y_test"][curve_idx],
            y_pred[curve_idx],
            fig_dir / f"08_prediction_curve_{suffix}.png",
            title=f"{model.name} 24h prediction from {curve_time}",
        )
        plot_error_distribution(current_splits["y_test"], y_pred, fig_dir / f"09_error_distribution_{suffix}.png", title=f"{model.name} error distribution")
        plot_scatter(current_splits["y_test"], y_pred, fig_dir / f"10_actual_vs_pred_{suffix}.png", title=f"{model.name}: actual vs predicted")
        if hasattr(model, "history") and getattr(model, "history"):
            plot_training_loss(getattr(model, "history"), fig_dir / f"11_training_loss_{suffix}.png", title=f"{model.name} training curve")
            pd.DataFrame(getattr(model, "history")).to_csv(Path(output_dir) / "history" / f"{suffix}_epoch_history.csv", index=False)
        try:
            ext = ".pt" if canonical in {"lstm", "transformer", "samformer"} else ".joblib"
            model.save(model_dir / f"{suffix}_M{memory_length}_H{horizon}{ext}")
        except Exception as exc:
            print(f"[WARN] 模型保存失败 {model.name}: {exc}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(Path(output_dir) / "metrics_latest.csv", index=False)
    plot_metrics_comparison(metrics_df, fig_dir / "12_model_metrics_comparison_latest.png")
    hist_df = load_history(history_path)
    plot_history_dashboard(hist_df, fig_dir / "13_training_history_dashboard.png")
    log(f"[5/5] 实验完成，总用时 {format_seconds(perf_counter() - experiment_start)}")

    return {
        "preprocess_report": pre_report,
        "feature_cols": feature_cols,
        "metrics": metrics_df,
        "splits": splits,
        "predictions": predictions,
        "output_dir": str(output_dir),
    }
