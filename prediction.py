import pandas as pd
import numpy as np
import joblib
import os

# 加载原始数据
def load_original_data(data_file):
    """加载原始数据"""
    data = pd.read_csv(data_file)
    data['年月'] = pd.to_datetime(data['年月'], format='%Y%m')
    return data

# 加载模型
def load_models(model_dir):
    """加载单一模型"""
    model_path = os.path.join(model_dir, 'single_model.pkl')
    return joblib.load(model_path)

# 生成未来六个月的日期
def generate_future_dates(last_date, months=6):
    """生成未来六个月的日期"""
    future_dates = []
    for i in range(1, months + 1):
        # 计算未来月份
        year = last_date.year
        month = last_date.month + i
        
        # 处理月份超过12的情况
        while month > 12:
            year += 1
            month -= 12
        
        # 格式化为YYYYMM字符串
        future_date = pd.to_datetime(f'{year}{month:02d}', format='%Y%m')
        future_dates.append(future_date)
    
    return future_dates

# 为小区生成未来六个月的特征数据
def generate_future_features(community_data, future_dates):
    """为小区生成未来六个月的特征数据"""
    # 获取小区的基本信息
    community_info = community_data.iloc[-1].copy()
    
    # 生成未来六个月的特征数据
    future_data = []
    for date in future_dates:
        # 复制小区基本信息
        future_row = community_info.copy()
        
        # 更新年月
        future_row['年月'] = date
        future_row['年'] = date.year
        future_row['月'] = date.month
        future_row['季度'] = date.quarter
        
        # 移除y值（因为这是我们要预测的）
        future_row['y'] = np.nan
        
        # 添加到未来数据列表
        future_data.append(future_row)
    
    # 转换为DataFrame
    future_df = pd.DataFrame(future_data)
    
    # 生成滞后特征和滚动统计特征
    future_df = generate_lag_features(future_df, community_data)
    
    return future_df

# 生成滞后特征和滚动统计特征
def generate_lag_features(future_df, historical_data):
    """生成滞后特征和滚动统计特征"""
    # 获取历史数据的y值
    historical_y = historical_data['y'].values
    
    # 生成滞后特征（前1-6个月的y值）
    for i in range(1, 7):
        if len(historical_y) >= i:
            # 使用历史数据的最后i个值作为未来数据的滞后特征
            future_df[f'y_lag_{i}'] = historical_y[-i]
        else:
            # 如果历史数据不足，使用0填充
            future_df[f'y_lag_{i}'] = 0
    
    # 计算历史数据的滚动统计特征
    if len(historical_y) >= 3:
        rolling_mean = historical_y[-3:].mean()
        rolling_max = historical_y[-3:].max()
        rolling_min = historical_y[-3:].min()
    else:
        rolling_mean = historical_y.mean() if len(historical_y) > 0 else 0
        rolling_max = historical_y.max() if len(historical_y) > 0 else 0
        rolling_min = historical_y.min() if len(historical_y) > 0 else 0
    
    # 填充滚动统计特征
    future_df['y_rolling_mean_3'] = rolling_mean
    future_df['y_rolling_max_3'] = rolling_max
    future_df['y_rolling_min_3'] = rolling_min
    
    return future_df

# 预测未来六个月的y值
def predict_future(model, original_data, months=6):
    """预测未来六个月的y值"""
    # 创建预测结果目录
    if not os.path.exists('predictions'):
        os.makedirs('predictions')
    
    # 获取所有小区ID
    community_ids = original_data['小区ID'].unique()
    
    # 存储所有预测结果
    all_predictions = []
    
    # 对每个小区进行预测
    for community_id in community_ids:
        print(f"正在预测小区 {community_id} 未来六个月的y值...")
        
        # 获取该小区的历史数据
        community_data = original_data[original_data['小区ID'] == community_id].copy()
        
        # 进行特征工程
        from data_preprocessing import feature_engineering
        community_data = feature_engineering(community_data)
        
        # 获取最后一个日期
        last_date = community_data['年月'].max()
        
        # 生成未来六个月的日期
        future_dates = generate_future_dates(last_date, months)
        
        # 生成未来六个月的特征数据
        future_features = generate_future_features(community_data, future_dates)
        
        # 准备模型输入数据
        X_future, _ = prepare_model_data(future_features)
        
        # 预测
        y_pred = model.predict(X_future)
        
        # 保存预测结果
        future_features['预测y值'] = y_pred
        
        # 提取需要的列
        prediction_result = future_features[['小区ID', '年月', '预测y值']]
        prediction_result['年月'] = prediction_result['年月'].dt.strftime('%Y%m')
        
        # 添加到所有预测结果
        all_predictions.append(prediction_result)
        
        # 保存单个小区的预测结果
        prediction_result.to_csv(f'predictions/{community_id}_predictions.csv', index=False)
        
        print(f"小区 {community_id} 未来六个月的y值预测完成！")
    
    # 合并所有预测结果
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_df.to_csv('predictions/all_communities_predictions.csv', index=False)
        print("\n所有小区的预测结果已保存到 predictions/all_communities_predictions.csv")
    
    return all_predictions

# 准备模型输入数据
def prepare_model_data(data, target_col='y'):
    """准备模型输入数据，分离特征和目标变量"""
    # 选择特征列，排除小区ID、小区类型等不需要的列
    feature_cols = [col for col in data.columns if col not in ['年月', '小区ID', '小区类型', target_col]]
    
    # 分离特征和目标变量
    X = data[feature_cols]
    y = data[target_col] if target_col in data.columns else None
    
    return X, y

if __name__ == '__main__':
    # 加载原始数据
    original_data = load_original_data('originData.csv')
    
    # 加载模型
    model = load_models('models')
    
    # 预测未来六个月的y值
    predict_future(model, original_data, months=6)