# 基于历史数据的光伏电站日前输出功率预测

本项目面向《机器学习》课程设计：使用澳大利亚 Alice Springs 光伏示范设施历史数据，预测未来 24 小时光伏输出功率。默认将原始 5 分钟数据重采样为 15 分钟，并采用 `memory_length=96/192` 的历史序列预测未来 `horizon=96` 个点。

## 1. 项目结构

```text
pv_power_forecasting_project/
├── main.py                         # 命令行入口
├── app.py                          # Streamlit 图形化前端
├── requirements.txt                # 依赖
├── data/sample_pv_7days.csv         # 小样例数据，可用于快速测试
├── src/
│   ├── preprocessing.py             # 缺失值、异常值、重采样、归一化
│   ├── feature_engineering.py        # 时间特征、滞后特征、滚动特征、相关性/PCA 图
│   ├── data_windowing.py            # 96/192 -> 96 窗口构建
│   ├── metrics.py                   # nRMSE、nMAE、R²
│   ├── visualization.py             # 预处理/特征/预测/对比图表生成
│   ├── pipeline.py                  # 端到端实验流水线
│   └── models/
│       ├── xgboost_model.py
│       ├── random_forest_model.py
│       ├── lstm_model.py
│       └── transformer_model.py
├── tests/test_pipeline.py           # 轻量测试
└── docs/
    ├── report_outline.md            # 课程报告撰写提纲
    └── experiment_notes.md          # 实验说明与调参建议
```

## 2. 安装依赖

建议新建虚拟环境：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> 如果暂时不跑前端，可不安装 streamlit；如果暂时不跑 XGBoost，可先不安装 xgboost。

## 3. 命令行运行

把完整数据放到：

```text
data/Austrailapvdataset.csv
```

然后运行快速测试：

```bash
python main.py --data data/sample_pv_7days.csv --models random_forest --memory-length 16 --horizon 8 --max-windows 120 --quick
```

正式实验示例：

```bash
python main.py --data data/Austrailapvdataset.csv --models random_forest xgboost --memory-length 96 --horizon 96 --max-windows 5000 --device gpu --gpu-id 1
```

深度学习模型示例：

```bash
python main.py --data data/Austrailapvdataset.csv --models lstm transformer --memory-length 96 --horizon 96 --max-windows 8000 --device gpu --gpu-id 0
```

常用设备参数：

- `--device auto`：自动选择，优先 GPU；
- `--device cpu`：强制使用 CPU；
- `--device gpu --gpu-id 1`：显式使用 `GPU 1`；
- `--device cuda:0`：直接指定 `GPU 0`。

## 4. Streamlit 前端

```bash
streamlit run app.py
```

前端功能：

- 上传 CSV 或填写本地 CSV 路径；
- 图形化选择模型、Memory length、horizon、窗口数量；
- 图形化选择 CPU / GPU 0 / GPU 1；
- 查看训练进度；
- 控制台与前端状态区显示训练进度和预计剩余时间；
- 自动展示预处理图、特征工程图、预测曲线、误差分布图、模型指标对比图；
- 查看历史训练记录与模型比较看板。

## 5. 输出目录

默认输出在 `results/`：

```text
results/
├── figures/                         # 所有图表
├── models/                          # 模型权重/模型文件
├── scalers/                         # 归一化器
├── processed/                       # 清洗/归一化后的 CSV
├── history/training_history.csv      # 历史实验记录
├── metrics_latest.csv                # 最近一次实验结果
├── preprocess_report.json
└── feature_report.json
```

## 6. 已实现图表

数据预处理阶段：

- 原始负荷时序图；
- 缺失值统计图；
- 清洗前后对比图；
- 清洗后缺失值统计图。

特征工程阶段：

- 相关系数热力图；
- 互信息 Top-20 特征图；
- PCA 解释方差图。

模型预测阶段：

- 未来 24 小时实际值 vs 预测值曲线；
- 误差分布直方图；
- Actual vs Predicted 散点图；
- 深度模型训练损失曲线；
- 多模型指标对比图；
- 历史实验看板。

## 7. 测试

```bash
pytest -q
```

测试不会训练大模型，只会生成合成小数据并验证预处理、特征工程、窗口构建、指标计算和 Random Forest 快速流水线可运行。
