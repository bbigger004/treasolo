import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import os

# 加载预处理后的数据
def load_processed_data(train_file, test_file):
    """加载预处理后的数据"""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data

# 训练单一模型
def train_single_model(train_data, test_data):
    """训练单一模型"""
    # 创建模型保存目录
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # 定义XGBoost参数网格
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    print("正在训练单一模型...")
    
    # 准备模型输入数据
    X_train, y_train = prepare_model_data(train_data)
    X_test, y_test = prepare_model_data(test_data)
    
    # 使用时间序列交叉验证
    tscv = TimeSeriesSplit(n_splits=3)
    
    # 初始化XGBoost回归器
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # 网格搜索优化超参数
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    # 获取最佳模型
    best_model = grid_search.best_estimator_
    
    # 评估模型
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    
    # 存储评估结果
    evaluation_results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        '最佳参数': grid_search.best_params_
    }
    
    # 保存模型
    joblib.dump(best_model, 'models/single_model.pkl')
    
    # 保存评估结果
    evaluation_df = pd.DataFrame([evaluation_results])
    evaluation_df.to_csv('models/evaluation_results.csv', index=False)
    
    print(f"单一模型训练完成！MAE: {mae:.2f}, RMSE: {rmse:.2f}")
    
    return best_model, evaluation_results

# 准备模型输入数据
def prepare_model_data(data, target_col='y'):
    """准备模型输入数据，分离特征和目标变量"""
    # 选择特征列
    feature_cols = [col for col in data.columns if col not in ['年月', '小区ID', target_col]]
    
    # 分离特征和目标变量
    X = data[feature_cols]
    y = data[target_col]
    
    return X, y

if __name__ == '__main__':
    # 加载预处理后的数据
    train_data, test_data = load_processed_data('train_data.csv', 'test_data.csv')
    
    # 训练单一模型
    model, evaluation_results = train_single_model(train_data, test_data)
    
    print("\n模型训练完成！")
    print(f"MAE: {evaluation_results['MAE']:.2f}")
    print(f"RMSE: {evaluation_results['RMSE']:.2f}")