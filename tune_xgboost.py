from __future__ import annotations

import argparse
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.pipeline import build_model, prepare_experiment_data
from src.utils import format_seconds, resolve_training_device


def parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]

def parse_args():
    parser = argparse.ArgumentParser(description="XGBoost 自动调参")
    parser.add_argument("--data", type=str, default="data/Austrailapvdataset.csv")
    parser.add_argument("--output-dir", type=str, default="results_tuning_xgb")
    parser.add_argument("--memory-length", type=int, default=96)
    parser.add_argument("--horizon", type=int, default=96)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-windows", type=int, default=3500)
    parser.add_argument("--top-k-features", type=int, default=48)
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--gpu-id", type=int, default=1)
    parser.add_argument("--n-estimators", type=str, default="200,300")
    parser.add_argument("--max-depth", type=str, default="4,5")
    parser.add_argument("--learning-rate", type=str, default="0.03,0.05")
    parser.add_argument("--subsample", type=str, default="0.8,1.0")
    parser.add_argument("--colsample-bytree", type=str, default="0.8,1.0")
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--coarse-target-step", type=int, default=4, help="第一阶段每隔多少个 horizon 点采样一个目标点")
    parser.add_argument("--refine-top-k", type=int, default=8, help="第一阶段保留前 K 组进入全 horizon 复赛")
    return parser.parse_args()


def subset_splits_by_target_stride(splits: Dict[str, object], step: int) -> Dict[str, object]:
    if step <= 1:
        return splits
    target_idx = list(range(0, splits["y_train"].shape[1], step))
    reduced = dict(splits)
    for key in ("y_train", "y_val", "y_test"):
        reduced[key] = splits[key][:, target_idx]
    reduced["target_idx"] = target_idx
    return reduced


def make_grid(args) -> List[Dict[str, float | int]]:
    return [
        {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "max_bin": args.max_bin,
            "early_stopping_rounds": args.early_stopping_rounds,
            "progress_interval": 16,
        }
        for n_estimators, max_depth, learning_rate, subsample, colsample_bytree in product(
            parse_int_list(args.n_estimators),
            parse_int_list(args.max_depth),
            parse_float_list(args.learning_rate),
            parse_float_list(args.subsample),
            parse_float_list(args.colsample_bytree),
        )
    ]


def train_and_score(
    params: Dict[str, float | int],
    splits: Dict[str, object],
    input_dim: int,
    memory_length: int,
    horizon: int,
    device: str,
) -> Dict[str, object]:
    model = build_model(
        "xgboost",
        input_dim=input_dim,
        memory_length=memory_length,
        horizon=horizon,
        quick=False,
        device=device,
        show_progress=False,
        model_params=params,
    )
    model.fit(splits["X_train"], splits["y_train"], splits["X_val"], splits["y_val"])
    metrics = model.evaluate(splits["X_val"], splits["y_val"])
    return {**params, **metrics}


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_training_device(args.device, gpu_id=args.gpu_id)

    prepared = prepare_experiment_data(
        data_path=args.data,
        output_dir=output_dir,
        memory_length=args.memory_length,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
        top_k_features=args.top_k_features,
        quick=False,
        target_col="load",
        time_col="time",
    )
    splits = prepared["splits"]
    input_dim = splits["X_train"].shape[-1]
    flat_dim = splits["X_train"].shape[1] * splits["X_train"].shape[2]
    full_horizon = splits["y_train"].shape[1]
    grid = make_grid(args)

    print(
        f"[Info] train_shape={splits['X_train'].shape}, val_shape={splits['X_val'].shape}, "
        f"flat_input_dim={flat_dim}, horizon={full_horizon}, grid={len(grid)}",
        flush=True,
    )
    print(
        f"[Info] 原始全量需要训练约 {len(grid) * full_horizon} 个 XGBoost 子模型，"
        f"现改为两阶段：粗筛 stride={args.coarse_target_step} + 全量复赛 top-{args.refine_top_k}",
        flush=True,
    )

    coarse_splits = subset_splits_by_target_stride(splits, args.coarse_target_step)
    coarse_horizon = coarse_splits["y_train"].shape[1]
    coarse_rows: List[Dict[str, object]] = []
    stage1_start = time.perf_counter()
    total = len(grid)
    for idx, params in enumerate(grid, start=1):
        print(
            f"[Stage1 {idx}/{total}] M={args.memory_length}, horizon_subset={coarse_horizon}, "
            f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"lr={params['learning_rate']}, subsample={params['subsample']}, "
            f"colsample={params['colsample_bytree']}",
            flush=True,
        )
        row = train_and_score(
            params=params,
            splits=coarse_splits,
            input_dim=input_dim,
            memory_length=args.memory_length,
            horizon=coarse_horizon,
            device=resolved_device,
        )
        row["stage"] = "coarse"
        row["target_count"] = coarse_horizon
        coarse_rows.append(row)
        pd.DataFrame(coarse_rows).sort_values(["nRMSE(%)", "nMAE(%)", "R2"], ascending=[True, True, False]).to_csv(
            output_dir / "xgboost_tuning_stage1.csv", index=False
        )

    coarse_results = pd.DataFrame(coarse_rows).sort_values(["nRMSE(%)", "nMAE(%)", "R2"], ascending=[True, True, False])
    survivors = coarse_results.head(min(args.refine_top_k, len(coarse_results))).to_dict("records")
    print(
        f"[Stage1] 完成，用时 {format_seconds(time.perf_counter() - stage1_start)}，"
        f"保留 {len(survivors)} 组进入全 horizon 复赛",
        flush=True,
    )

    final_rows: List[Dict[str, object]] = []
    stage2_start = time.perf_counter()
    for idx, row in enumerate(survivors, start=1):
        params = {
            "n_estimators": int(row["n_estimators"]),
            "max_depth": int(row["max_depth"]),
            "learning_rate": float(row["learning_rate"]),
            "subsample": float(row["subsample"]),
            "colsample_bytree": float(row["colsample_bytree"]),
            "max_bin": int(row["max_bin"]),
            "early_stopping_rounds": int(row["early_stopping_rounds"]),
            "progress_interval": 16,
        }
        print(
            f"[Stage2 {idx}/{len(survivors)}] full_horizon={full_horizon}, "
            f"n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
            f"lr={params['learning_rate']}, subsample={params['subsample']}, "
            f"colsample={params['colsample_bytree']}",
            flush=True,
        )
        result = train_and_score(
            params=params,
            splits=splits,
            input_dim=input_dim,
            memory_length=args.memory_length,
            horizon=args.horizon,
            device=resolved_device,
        )
        result["stage"] = "full"
        result["target_count"] = full_horizon
        final_rows.append(result)
        pd.DataFrame(final_rows).sort_values(["nRMSE(%)", "nMAE(%)", "R2"], ascending=[True, True, False]).to_csv(
            output_dir / "xgboost_tuning_results.csv", index=False
        )

    results = pd.DataFrame(final_rows).sort_values(["nRMSE(%)", "nMAE(%)", "R2"], ascending=[True, True, False])
    best = results.iloc[0]
    (output_dir / "best_xgboost_params.txt").write_text(
        "\n".join(
            [
                f"device={resolved_device}",
                f"memory_length={args.memory_length}",
                f"horizon={args.horizon}",
                f"top_k_features={args.top_k_features}",
                f"coarse_target_step={args.coarse_target_step}",
                f"refine_top_k={args.refine_top_k}",
                f"n_estimators={int(best['n_estimators'])}",
                f"max_depth={int(best['max_depth'])}",
                f"learning_rate={best['learning_rate']}",
                f"subsample={best['subsample']}",
                f"colsample_bytree={best['colsample_bytree']}",
                f"max_bin={int(best['max_bin'])}",
                f"early_stopping_rounds={int(best['early_stopping_rounds'])}",
                f"nRMSE(%)={best['nRMSE(%)']:.6f}",
                f"nMAE(%)={best['nMAE(%)']:.6f}",
                f"R2={best['R2']:.6f}",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"[Stage2] 完成，用时 {format_seconds(time.perf_counter() - stage2_start)}，"
        f"最佳结果 nRMSE={best['nRMSE(%)']:.4f}% nMAE={best['nMAE(%)']:.4f}% R2={best['R2']:.4f}",
        flush=True,
    )
    print("\n===== Best Params =====")
    print(best.to_string())
    print(f"结果目录：{output_dir.resolve()}")


if __name__ == "__main__":
    main()
