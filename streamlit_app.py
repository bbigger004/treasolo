import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# 设置页面配置
st.set_page_config(
    page_title="小区y值预测系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data(file_path):
    """加载数据"""
    data = pd.read_csv(file_path)
    return data

# 加载模型
def load_models(model_dir):
    """加载单一模型"""
    model_path = os.path.join(model_dir, 'single_model.pkl')
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

# 页面标题
st.title("小区y值预测系统")

# 侧边栏
st.sidebar.title("导航")
page = st.sidebar.radio(
    "选择页面",
    ["数据概览", "模型评估", "预测结果"]
)

# 加载数据
@st.cache_data
def load_all_data():
    """加载所有数据"""
    # 加载原始数据
    original_data = pd.read_csv('originData.csv') if os.path.exists('originData.csv') else None
    if original_data is not None:
        original_data['年月'] = pd.to_datetime(original_data['年月'], format='%Y%m')
    
    # 加载训练数据和测试数据
    train_data = pd.read_csv('train_data.csv') if os.path.exists('train_data.csv') else None
    test_data = pd.read_csv('test_data.csv') if os.path.exists('test_data.csv') else None
    
    # 加载评估结果
    evaluation_results = pd.read_csv('evaluation/evaluation_results.csv') if os.path.exists('evaluation/evaluation_results.csv') else None
    
    # 加载预测结果
    all_predictions = pd.read_csv('predictions/all_communities_predictions.csv') if os.path.exists('predictions/all_communities_predictions.csv') else None
    
    # 加载模型
    model = load_models('models')
    
    return original_data, train_data, test_data, evaluation_results, all_predictions, model

original_data, train_data, test_data, evaluation_results, all_predictions, model = load_all_data()

# 数据概览页面
if page == "数据概览":
    st.header("数据概览")
    
    if original_data is not None:
        # 显示原始数据
        st.subheader("原始数据")
        st.dataframe(original_data)  # 显示所有原始数据
        
        # 小区数量统计
        st.subheader("小区数量统计")
        community_count = original_data['小区ID'].nunique()
        st.metric("小区总数", community_count)
        
        # 显示前10个小区的y值统计
        top_communities = original_data.groupby('小区ID')['y'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=top_communities.index, y=top_communities.values, title="前10个小区平均y值")
        fig.update_layout(xaxis_title="小区ID", yaxis_title="平均y值")
        st.plotly_chart(fig, use_container_width=True)
        
        # 小区y值时间序列
        st.subheader("小区y值时间序列")
        # 选择小区
        community_ids = original_data['小区ID'].unique()
        selected_community = st.selectbox("选择小区", community_ids)
        
        # 筛选数据
        community_data = original_data[original_data['小区ID'] == selected_community]
        
        # 绘制时间序列图
        fig = px.line(community_data, x='年月', y='y', title=f"{selected_community} y值变化趋势")
        st.plotly_chart(fig, use_container_width=True)
    
    # 特征相关性热力图
    st.subheader("特征相关性热力图")
    if train_data is not None:
        # 选择数值特征
        numeric_cols = ['y', '小区年限', '是否老旧小区', '是否增长停滞', '饱和度', '变压器容量', '变压器数量', '用户数量', '均价', '建成年份']
        # 确保所有列都存在
        available_cols = [col for col in numeric_cols if col in train_data.columns]
        if available_cols:
            correlation = train_data[available_cols].corr()
            
            # 绘制热力图
            fig = px.imshow(correlation, text_auto=True, title="特征相关性热力图")
            st.plotly_chart(fig, use_container_width=True)

# 模型评估页面
elif page == "模型评估":
    st.header("模型评估")
    
    if evaluation_results is not None:
        # 显示评估结果
        st.subheader("模型评估结果")
        st.dataframe(evaluation_results)
        
        # 评估指标可视化
        st.subheader("评估指标")
        
        # 绘制评估指标条形图
        metrics = ['MAE', 'MSE', 'RMSE', 'MAPE']
        # 确保所有指标都存在
        available_metrics = [metric for metric in metrics if metric in evaluation_results.columns]
        if available_metrics:
            metric_values = [evaluation_results[metric].values[0] for metric in available_metrics]
            
            fig = px.bar(x=available_metrics, y=metric_values, title="模型评估指标")
            fig.update_layout(xaxis_title="指标", yaxis_title="值")
            st.plotly_chart(fig, use_container_width=True)
        
        # 实际值与预测值对比
        st.subheader("实际值与预测值对比")
        
        # 加载所有测试数据的预测结果
        all_test_predictions = pd.read_csv('evaluation/all_predictions.csv') if os.path.exists('evaluation/all_predictions.csv') else None
        
        if all_test_predictions is not None:
            # 选择小区
            community_ids = all_test_predictions['小区ID'].unique()
            selected_community = st.selectbox("选择小区", community_ids)
            
            # 筛选该小区的数据
            community_data = all_test_predictions[all_test_predictions['小区ID'] == selected_community]
            
            if not community_data.empty:
                # 绘制对比图
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=community_data['年月'], y=community_data['y'], name='实际值', mode='lines+markers'))
                fig.add_trace(go.Scatter(x=community_data['年月'], y=community_data['预测值'], name='预测值', mode='lines+markers'))
                fig.update_layout(title=f"{selected_community} 实际值与预测值对比", xaxis_title="时间", yaxis_title="y值")
                st.plotly_chart(fig, use_container_width=True)
            
            # 残差分布
            st.subheader("残差分布")
            if not community_data.empty:
                residuals = community_data['y'] - community_data['预测值']
                
                # 绘制残差直方图
                fig = px.histogram(residuals, nbins=20, title=f"{selected_community} 残差分布")
                fig.update_layout(xaxis_title="残差", yaxis_title="频率")
                st.plotly_chart(fig, use_container_width=True)
            
            # 添加功能：所有测试数据与模型执行结果的散点图
            st.subheader("所有测试数据与模型执行结果对比")
            fig = px.scatter(all_test_predictions, x='y', y='预测值', title="测试数据与模型执行结果对比", hover_data=['小区ID', '年月'])
            fig.add_trace(go.Scatter(x=[all_test_predictions['y'].min(), all_test_predictions['y'].max()], 
                                   y=[all_test_predictions['y'].min(), all_test_predictions['y'].max()], 
                                   mode='lines', name='理想线', line=dict(color='red', dash='dash')))
            fig.update_layout(xaxis_title="测试数据", yaxis_title="模型预测值")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("尚未生成评估结果，请先运行模型评估脚本。")

# 预测结果页面
elif page == "预测结果":
    st.header("预测结果")
    
    if all_predictions is not None:
        # 显示所有预测结果
        st.subheader("所有小区预测结果")
        st.dataframe(all_predictions)
        
        # 下载预测结果
        csv = all_predictions.to_csv(index=False)
        st.download_button(
            label="下载预测结果",
            data=csv,
            file_name="all_communities_predictions.csv",
            mime="text/csv"
        )
        
        # 单个小区预测结果
        st.subheader("单个小区预测结果")
        
        # 选择小区
        community_ids = all_predictions['小区ID'].unique()
        selected_community = st.selectbox("选择小区", community_ids)
        
        # 筛选该小区的预测结果
        community_prediction = all_predictions[all_predictions['小区ID'] == selected_community]
        
        # 绘制预测结果折线图
        fig = px.line(community_prediction, x='年月', y='预测y值', title=f"{selected_community} 未来六个月y值预测")
        fig.update_layout(xaxis_title="时间", yaxis_title="预测y值")
        st.plotly_chart(fig, use_container_width=True)
        
        # 显示该小区的预测结果
        st.dataframe(community_prediction)
        
        # 各小区预测值对比
        st.subheader("各小区预测值对比")
        
        # 选择月份
        months = all_predictions['年月'].unique()
        selected_month = st.selectbox("选择月份", months)
        
        # 筛选该月份的预测结果
        month_prediction = all_predictions[all_predictions['年月'] == selected_month]
        
        # 绘制柱状图
        fig = px.bar(month_prediction, x='小区ID', y='预测y值', title=f"{selected_month} 各小区预测y值对比")
        fig.update_layout(xaxis_title="小区ID", yaxis_title="预测y值")
        st.plotly_chart(fig, use_container_width=True)
        
        # 预测趋势热力图
        st.subheader("预测趋势热力图")
        
        # 转换数据格式
        heatmap_data = all_predictions.pivot(index='小区ID', columns='年月', values='预测y值')
        
        # 绘制热力图
        fig = px.imshow(heatmap_data, text_auto=True, title="各小区未来六个月预测y值热力图")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("尚未生成预测结果，请先运行预测脚本。")

# 页脚
st.sidebar.markdown("---")
st.sidebar.info("小区y值预测")