# 基于历史数据的光伏电站日前输出功率预测实验报告

本目录为课程设计最终实验报告的独立工作区，包含报告正文、配套图表与数据表。

## 目录结构

- [01_background_dataset.md](01_background_dataset.md)：课题背景与数据集总览
- [02_preprocessing.md](02_preprocessing.md)：数据预处理过程与发现
- [03_feature_engineering.md](03_feature_engineering.md)：特征工程设计、重要特征分析与特征数对性能的影响
- [04_model_experiments.md](04_model_experiments.md)：四类模型在 `memory_length=96/192` 条件下的实验结果与现象分析
- [05_xgboost_tuning.md](05_xgboost_tuning.md)：XGBoost 超参数调优方法、结果与原因分析

## 说明

1. 报告仅采用正式实验记录与可复现实验结果。
2. 快速 smoke 测试、异常中间结果与明显不合理记录未纳入正文分析。
3. 所有图表与表格均位于 `assets/` 目录，可直接用于排版或导出。
