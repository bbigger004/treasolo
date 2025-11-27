# 小区未来y值预测系统

## 项目概述

本系统用于预测社区未来六个月的y值，基于历史数据和机器学习算法。系统采用单一模型架构，使用XGBoost算法进行预测，支持多小区同时预测。

## 功能特点

- **单一模型架构**：使用一个统一的模型预测所有小区，无需为每个小区单独训练
- **时间序列预测**：支持基于历史数据的未来六个月预测
- **特征工程**：自动生成滞后特征、滚动统计特征等
- **模型优化**：使用网格搜索和时间序列交叉验证优化模型参数
- **可视化评估**：生成预测结果可视化图表和评估报告
- **批量预测**：支持所有小区的批量预测

## 安装步骤

### 1. 环境要求

- Python 3.8+ 
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn

### 2. 安装依赖

```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn streamlit   plotly   scikit-learn joblib
```

## 使用方法

### 1. 数据准备

将原始数据文件 `originData.csv` 放置在项目根目录下，数据格式要求：

| 列名 | 类型 | 说明 |
|------|------|------|
| 年月 | int | 格式：YYYYMM，如202401 |
| 小区ID | string | 小区唯一标识 |
| y | float | 目标预测值 |
| 小区年限 | float | 小区建成年限 |
| 是否老旧小区 | int | 0/1，1表示是老旧小区 |
| 是否增长停滞 | int | 0/1，1表示增长停滞 |
| 饱和度 | float | 小区饱和度 |
| 变压器容量 | float | 变压器容量 |
| 变压器数量 | int | 变压器数量 |
| 用户数量 | int | 用户数量 |
| 均价 | float | 小区均价 |
| 建成年份 | int | 建成年份 |
| 小区类型 | string | 小区类型（可选，不参与模型训练） |

### 2. 数据预处理

运行数据预处理脚本，生成训练集和测试集：

```bash
python data_preprocessing.py
```

输出：
- `train_data.csv`：训练集数据
- `test_data.csv`：测试集数据

### 3. 模型训练

运行模型训练脚本，训练XGBoost模型：

```bash
python model_training.py
```

输出：
- `models/single_model.pkl`：训练好的单一模型
- `models/evaluation_results.csv`：模型评估结果

### 4. 模型评估

运行模型评估脚本，生成评估报告和可视化图表：

```bash
python model_evaluation.py
```

输出：
- `evaluation/all_predictions.csv`：测试集预测结果
- `evaluation/evaluation_results.csv`：详细评估指标
- `evaluation/all_communities_predictions.png`：预测结果可视化
- `evaluation/all_communities_residuals.png`：残差分布可视化
- `evaluation/overall_evaluation.png`：总体评估报告
- `evaluation/stats_summary.csv`：评估指标统计摘要

### 5. 未来预测

运行预测脚本，预测所有小区未来六个月的y值：

```bash
python prediction.py
```

输出：
- `predictions/all_communities_predictions.csv`：所有小区未来六个月预测结果
- `predictions/{小区ID}_predictions.csv`：每个小区的单独预测结果

## 文件结构

```
treasolo/
├── originData.csv          # 原始数据文件
├── data_preprocessing.py   # 数据预处理脚本
├── model_training.py       # 模型训练脚本
├── model_evaluation.py     # 模型评估脚本
├── prediction.py           # 未来预测脚本
├── train_data.csv          # 预处理后的训练集
├── test_data.csv           # 预处理后的测试集
├── models/                 # 模型保存目录
│   ├── single_model.pkl    # 训练好的单一模型
│   └── evaluation_results.csv  # 模型评估结果
├── evaluation/             # 评估结果目录
│   ├── all_predictions.csv     # 测试集预测结果
│   ├── evaluation_results.csv  # 详细评估指标
│   ├── all_communities_predictions.png  # 预测结果可视化
│   ├── all_communities_residuals.png     # 残差分布可视化
│   ├── overall_evaluation.png   # 总体评估报告
│   └── stats_summary.csv        # 评估指标统计摘要
└── predictions/            # 预测结果目录
    ├── all_communities_predictions.csv  # 所有小区未来预测结果
    └── {小区ID}_predictions.csv         # 每个小区的单独预测结果
```

## 模型训练流程

1. **数据加载**：从`originData.csv`加载原始数据
2. **数据清洗**：处理缺失值和异常值，移除小区类型列
3. **特征工程**：
   - 提取时间特征（年、月、季度）
   - 生成滞后特征（前1-6个月的y值）
   - 生成滚动统计特征（过去3个月的均值、最大值、最小值）
4. **数据划分**：按时间顺序划分为训练集和测试集
5. **模型训练**：
   - 使用网格搜索优化超参数
   - 使用时间序列交叉验证
   - 训练XGBoost回归模型
6. **模型评估**：计算MAE、MSE、RMSE、MAPE等评估指标
7. **未来预测**：生成所有小区未来六个月的预测结果

## 结果解释

### 评估指标

- **MAE（平均绝对误差）**：预测值与真实值的平均绝对差异，值越小越好
- **MSE（均方误差）**：预测值与真实值的平均平方差异，值越小越好
- **RMSE（均方根误差）**：MSE的平方根，值越小越好，单位与y值相同
- **MAPE（平均绝对百分比误差）**：预测值与真实值的平均百分比差异，值越小越好

### 预测结果文件

- `all_communities_predictions.csv`：包含所有小区未来六个月的预测结果
  - 小区ID：小区唯一标识
  - 年月：预测的年月，格式YYYYMM
  - 预测y值：模型预测的y值

## 注意事项

1. 原始数据中的`年月`列必须为YYYYMM格式
2. 小区ID仅用于标识不同小区，不参与模型训练
3. 小区类型列不参与模型训练，仅作为参考信息
4. 模型训练时间取决于数据量大小和参数网格规模
5. 预测结果应结合实际业务情况进行分析和调整

## 扩展建议

1. 尝试其他机器学习算法，如LightGBM、CatBoost等
2. 增加更多特征工程，如节假日特征、季节性特征等
3. 实现模型监控和更新机制
4. 开发Web界面，方便用户交互和查看结果
5. 增加异常值检测和处理机制

## 联系方式

如有问题或建议，请联系系统管理员。