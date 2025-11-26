import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载测试数据
def load_test_data(test_file):
    """加载测试数据"""
    test_data = pd.read_csv(test_file)
    return test_data

# 加载模型
def load_models(model_dir):
    """加载单一模型"""
    model_path = os.path.join(model_dir, 'single_model.pkl')
    return joblib.load(model_path)

# 评估模型
def evaluate_models(model, test_data):
    """评估模型"""
    # 创建评估结果目录
    if not os.path.exists('evaluation'):
        os.makedirs('evaluation')
    
    print("正在评估单一模型...")
    
    # 准备模型输入数据
    X_test, y_test = prepare_model_data(test_data)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    # 存储评估结果
    evaluation_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    # 保存预测结果
    test_data['预测值'] = y_pred
    test_data.to_csv('evaluation/all_predictions.csv', index=False)
    
    # 生成可视化图表
    plot_predictions(test_data, 'all_communities')
    plot_residuals(y_test, y_pred, 'all_communities')
    plot_test_prediction_scatter(y_test, y_pred, 'all_communities')
    
    # 保存评估结果
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv('evaluation/evaluation_results.csv', index=False)
    
    # 生成总体评估报告
    generate_evaluation_report(evaluation_df)
    
    return evaluation_results

# 准备模型输入数据
def prepare_model_data(data, target_col='y'):
    """准备模型输入数据，分离特征和目标变量"""
    # 选择特征列
    feature_cols = [col for col in data.columns if col not in ['年月', '小区ID', target_col]]
    
    # 分离特征和目标变量
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y

# 绘制预测值与实际值对比图
def plot_predictions(test_community, community_id):
    """绘制预测值与实际值对比图"""
    plt.figure(figsize=(12, 6))
    plt.plot(pd.to_datetime(test_community['年月']), test_community['y'], label='实际值', marker='o')
    plt.plot(pd.to_datetime(test_community['年月']), test_community['预测值'], label='预测值', marker='x')
    plt.title(f'{community_id} 实际值与预测值对比')
    plt.xlabel('时间')
    plt.ylabel('y值')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'evaluation/{community_id}_predictions.png')
    plt.close()

# 绘制残差分布图
def plot_residuals(y_test, y_pred, community_id):
    """绘制残差分布图"""
    residuals = y_test - y_pred
    
    plt.figure(figsize=(12, 6))
    
    # 残差直方图
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, kde=True)
    plt.title(f'{community_id} 残差分布')
    plt.xlabel('残差')
    plt.ylabel('频率')
    
    # 残差散点图
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title(f'{community_id} 残差与预测值关系')
    plt.xlabel('预测值')
    plt.ylabel('残差')
    
    plt.tight_layout()
    plt.savefig(f'evaluation/{community_id}_residuals.png')
    plt.close()

# 绘制测试数据与预测结果散点图
def plot_test_prediction_scatter(y_test, y_pred, community_id):
    """绘制测试数据与预测结果散点图"""
    plt.figure(figsize=(10, 8))
    
    # 绘制散点图
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # 添加理想线（y=x）
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想线 (y=x)')
    
    # 设置图表属性
    plt.title(f'{community_id} 测试数据与预测结果对比')
    plt.xlabel('测试数据')
    plt.ylabel('模型预测值')
    plt.legend()
    plt.grid(True)
    
    # 保存图表
    plt.tight_layout()
    plt.savefig(f'evaluation/{community_id}_test_prediction_scatter.png')
    plt.close()

# 生成总体评估报告
def generate_evaluation_report(evaluation_df):
    """生成总体评估报告"""
    plt.figure(figsize=(16, 12))
    
    # 1. 各小区MAE对比
    plt.subplot(2, 2, 1)
    evaluation_df['MAE'].plot(kind='bar')
    plt.title('MAE指标')
    plt.xlabel('模型')
    plt.ylabel('MAE')
    plt.xticks(rotation=90)
    
    # 2. 各小区RMSE对比
    plt.subplot(2, 2, 2)
    evaluation_df['RMSE'].plot(kind='bar')
    plt.title('RMSE指标')
    plt.xlabel('模型')
    plt.ylabel('RMSE')
    plt.xticks(rotation=90)
    
    # 3. 各小区MAPE对比
    plt.subplot(2, 2, 3)
    evaluation_df['MAPE'].plot(kind='bar')
    plt.title('MAPE指标')
    plt.xlabel('模型')
    plt.ylabel('MAPE (%)')
    plt.xticks(rotation=90)
    
    # 4. 评估指标相关性热力图
    plt.subplot(2, 2, 4)
    correlation = evaluation_df.corr()
    sns.heatmap(correlation, annot=True, cmap='coolwarm')
    plt.title('评估指标相关性热力图')
    
    plt.tight_layout()
    plt.savefig('evaluation/overall_evaluation.png')
    plt.close()
    
    # 生成统计摘要
    stats_summary = evaluation_df.describe()
    stats_summary.to_csv('evaluation/stats_summary.csv')
    
    print("\n模型评估完成！")
    print("\n评估指标统计摘要：")
    print(stats_summary)

if __name__ == '__main__':
    # 加载测试数据
    test_data = load_test_data('test_data.csv')
    
    # 加载模型
    model = load_models('models')
    
    # 评估模型
    evaluate_models(model, test_data)