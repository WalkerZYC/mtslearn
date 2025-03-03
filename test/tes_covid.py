import pandas as pd
import os
import mtslearn.feature_extraction as fe
import mtslearn as ls

# 加载Excel文件
file_path = '/Users/zhaoyuechao/Documents/GitHub/mtslearn/tests/test_data/375_patients_example.xlsx'
# file_path = './test_data/375_patients.xlsx'  # 请替换为你的文件路径
df = pd.read_excel(file_path)

# 使用向前填充的方法填充PATIENT_ID列中的NaN值
df['PATIENT_ID'].fillna(method='ffill', inplace=True)

# 打印前40行数据
print(df.head(40))
df.to_csv('./test_data/375prep.csv')


fe = fe.FeModEvaluator(df,
                       'PATIENT_ID',
                       'RE_DATE',
                       'outcome',
                       {'eGFR': ['mean', 'max'], 'creatinine': ['mean']},
                        include_duration=True)

fe.run(model_type='xgboost', fill=True, fill_method='mean', test_size=0.3, balance_data= True,plot_importance=True)

fe.run(model_type='logit', fill=True, fill_method='mean', cross_val=True,plot_importance=True)

# Example calls to describe_data
# fe.describe_data(plot_type='boxplot', value_col='eGFR')
#fe.describe_data(plot_type='violinplot', value_col='eGFR')
# fe.describe_data(plot_type='correlation_matrix', feature1='eGFR', feature2='creatinine')

# 我们假设 'eGFR' 是我们要预测的特征值列，'outcome' 是预测目标列

# lstm_model_pipeline 函数的参数:
# df: 输入的数据集
# id_col: 病人的唯一标识符列 (如 'PATIENT_ID')
# time_col: 时间列 (如 'RE_DATE')
# value_col: 需要预测的特征值列 (如 'eGFR')
# outcome_col: 要预测的结果列 (如 'outcome')
# sequence_length: 使用的时间序列的长度 (如 10)

# 调用LSTM模型
# clstm_model_pipeline(df,
#                    id_col='PATIENT_ID',
 #                   time_col='RE_DATE',
#                    value_col='eGFR',
#                    outcome_col='outcome',
#                    sequence_length=10)
