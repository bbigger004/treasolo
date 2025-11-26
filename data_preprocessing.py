import pandas as pd
import numpy as np

# 读取数据
def load_data(file_path):
    """加载原始数据"""
    data = pd.read_csv(file_path)
    return data

# 数据清洗
def clean_data(data):
    """处理缺失值和异常值"""
    # 删除所有列都为空的行
    data = data.dropna(how='all')
    # 填充数值列缺失值为0
    numeric_cols = ['y', '小区年限', '是否老旧小区', '是否增长停滞', '饱和度', '变压器容量', '变压器数量', '用户数量', '均价', '建成年份']
    data[numeric_cols] = data[numeric_cols].fillna(0)
    # 移除小区类型列，因为不需要作为特征
    if '小区类型' in data.columns:
        data = data.drop('小区类型', axis=1)
    return data

# 特征工程
def feature_engineering(data):
    """进行特征工程"""
    # 将年月转换为datetime类型
    data['年月'] = pd.to_datetime(data['年月'], format='%Y%m')
    
    # 提取年、月、季度特征
    data['年'] = data['年月'].dt.year
    data['月'] = data['年月'].dt.month
    data['季度'] = data['年月'].dt.quarter
    
    # 生成滞后特征（前1-6个月的y值）
    for i in range(1, 7):
        data[f'y_lag_{i}'] = data['y'].shift(i)
    
    # 生成滚动统计特征（过去3个月的均值、最大值、最小值）
    data['y_rolling_mean_3'] = data['y'].rolling(window=3).mean()
    data['y_rolling_max_3'] = data['y'].rolling(window=3).max()
    data['y_rolling_min_3'] = data['y'].rolling(window=3).min()
    
    # 填充滞后特征和滚动统计特征的缺失值
    lag_cols = [f'y_lag_{i}' for i in range(1, 7)]
    rolling_cols = ['y_rolling_mean_3', 'y_rolling_max_3', 'y_rolling_min_3']
    data[lag_cols + rolling_cols] = data[lag_cols + rolling_cols].fillna(0)
    
    return data

# 数据划分
def split_data(data, split_date='2025-01-01'):
    """按时间顺序划分训练集和测试集"""
    train_data = data[data['年月'] < split_date]
    test_data = data[data['年月'] >= split_date]
    return train_data, test_data

# 准备模型输入数据
def prepare_model_data(data, target_col='y'):
    """准备模型输入数据，分离特征和目标变量"""
    # 选择特征列，排除小区ID、小区类型等不需要的列
    feature_cols = [col for col in data.columns if col not in ['年月', '小区ID', '小区类型', target_col]]
    
    # 分离特征和目标变量
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y

if __name__ == '__main__':
    print("开始数据预处理...")
    # 加载数据
    print("正在加载原始数据...")
    data = load_data('originData.csv')
    print(f"原始数据加载完成，共 {len(data)} 行")
    
    # 数据清洗
    print("正在进行数据清洗...")
    data = clean_data(data)
    print(f"数据清洗完成，共 {len(data)} 行")
    
    # 特征工程
    print("正在进行特征工程...")
    data = feature_engineering(data)
    print(f"特征工程完成，共 {len(data)} 行")
    
    # 数据划分
    print("正在进行数据划分...")
    train_data, test_data = split_data(data)
    print(f"数据划分完成，训练集 {len(train_data)} 行，测试集 {len(test_data)} 行")
    
    # 保存处理后的数据
    print("正在保存处理后的数据...")
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)
    print("数据保存完成！")
    
    print(f"数据预处理完成！")
    print(f"训练集大小：{len(train_data)}")
    print(f"测试集大小：{len(test_data)}")