import pandas as pd
print('Pandas版本:', pd.__version__)

try:
    data = pd.read_csv('originData.csv')
    print('数据加载成功，共', len(data), '行')
    print('数据列:', data.columns.tolist())
    print('前5行数据:')
    print(data.head())
except Exception as e:
    print('数据加载失败:', str(e))
