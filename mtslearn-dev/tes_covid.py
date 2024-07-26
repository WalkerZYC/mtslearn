import pandas as pd
import os
import mtslearn.feature_extraction as fe

# 加载Excel文件
# file_path = '/Users/zhaoyuechao/Desktop/academic/Projects/ISMTS/test_data/data/covid19_data/train/375_patients.xlsx'
file_path = '../test_data/data/covid19_data/train/375_patients.xlsx'  # 请替换为你的文件路径
df = pd.read_excel(file_path)

# 使用向前填充的方法填充PATIENT_ID列中的NaN值
df['PATIENT_ID'].fillna(method='ffill', inplace=True)
df.to_csv('../test_data/data/covid19_data/train/375prep.csv')
# 打印前40行数据
# print(df.head(40))

# 确保数据按时间顺序排序
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)
fe = fe.FeatureExtractorAndModelEvaluator(df, 'PATIENT_ID', 'RE_DATE',  'outcome', ['eGFR', 'creatinine'],['mean','max'],include_duration=True)
# evaluator.run(model_type='lasso', fill=True, fill_method='mean', test_size=0.3, balance_data= True)
fe.run(model_type='logit', fill=True, fill_method='mean', cross_val=True)
fe.plot_time_series()
fe.plot_heatmap()
fe.plot_boxplot()
fe.plot_violinplot()
fe.plot_correlation_matrix()
