import pandas as pd
import os
import mtslearn.feature_extraction as fe

# 加载Excel文件
# file_path = '/Users/zhaoyuechao/Desktop/academic/Projects/ISMTS/test_data/data/covid19_data/train/375_patients.xlsx'
file_path = 'test_data/data/covid19_data/train/375_patients.xlsx'  # 请替换为你的文件路径
df = pd.read_excel(file_path)

# 使用向前填充的方法填充PATIENT_ID列中的NaN值
df['PATIENT_ID'].fillna(method='ffill', inplace=True)

# 打印前40行数据
# print(df.head(40))

# 确保数据按时间顺序排序
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)
df.to_csv('../test_data/data/covid19_data/train/375prep.csv')


fe = fe.FeModEvaluator(df,
                       'PATIENT_ID',
                       'RE_DATE',
                       'outcome',
                       {'eGFR':['mean','max'],'creatinine':['mean']},
                        include_duration=True)
fe.run(model_type='xgboost', fill=True, fill_method='mean', test_size=0.3, balance_data= True,plot_importance=True)
#fe.run(model_type='logit', fill=True, fill_method='mean', cross_val=True,plot_importance=True)

# Example calls to describe_data
fe.describe_data(plot_type='boxplot', value_col='eGFR')
#fe.describe_data(plot_type='violinplot', value_col='eGFR')
fe.describe_data(plot_type='correlation_matrix', feature1='eGFR', feature2='creatinine')

