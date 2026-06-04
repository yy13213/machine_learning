from __future__ import annotations

import argparse
from pathlib import Path

from src.pipeline import run_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="PV day-ahead power forecasting course project")
    parser.add_argument("--data", type=str, default="data/Austrailapvdataset.csv", help="CSV 数据路径")
    parser.add_argument("--output-dir", type=str, default="results", help="输出目录")
    parser.add_argument("--models", nargs="+", default=["random_forest", "xgboost", "lstm", "transformer"], help="模型列表：random_forest xgboost lstm transformer samformer")
    parser.add_argument("--memory-length", type=int, default=96, help="输入历史长度，课程建议 96 或 192")
    parser.add_argument("--horizon", type=int, default=96, help="预测长度，日前 24h=96 个 15min 点")
    parser.add_argument("--stride", type=int, default=4, help="窗口滑动步长，越大越快")
    parser.add_argument("--max-windows", type=int, default=5000, help="最大训练窗口数，避免课程机训练过慢")
    parser.add_argument("--top-k-features", type=int, default=None, help="只保留相关性最高的 K 个特征，可留空")
    parser.add_argument("--target-col", type=str, default="load", help="目标列名")
    parser.add_argument("--time-col", type=str, default="time", help="时间列名")
    parser.add_argument("--device", type=str, default="auto", help="训练设备：auto/cpu/gpu/cuda/cuda:0/cuda:1")
    parser.add_argument("--gpu-id", type=int, default=0, help="当 device=auto/gpu/cuda 时使用的 GPU 编号")
    parser.add_argument("--feature-pca-dim", type=int, default=None, help="所有模型统一使用的输入 PCA 维度，例如 33/49；留空表示不启用")
    parser.add_argument("--deep-feature-dim", type=int, default=None, help="深度模型输入 PCA 降维维度，例如 24/32；树模型不受影响")
    parser.add_argument("--early-stopping-patience", type=int, default=3, help="深度模型早停耐心轮数")
    parser.add_argument("--quick", action="store_true", help="快速 smoke test 模式，只训练很小模型")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_experiment(
        data_path=args.data,
        output_dir=args.output_dir,
        models=args.models,
        memory_length=args.memory_length,
        horizon=args.horizon,
        stride=args.stride,
        max_windows=args.max_windows,
        top_k_features=args.top_k_features,
        quick=args.quick,
        target_col=args.target_col,
        time_col=args.time_col,
        device=args.device,
        gpu_id=args.gpu_id,
        show_progress=True,
        feature_pca_dim=args.feature_pca_dim,
        deep_feature_dim=args.deep_feature_dim,
        early_stopping_patience=args.early_stopping_patience,
    )
    print("\n===== 实验完成 =====")
    print(result["metrics"].to_string(index=False))
    print(f"结果目录：{Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
