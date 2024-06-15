import pandas as pd
import os
import mtslearn.feature_extraction as fe

# 加载Excel文件
file_path = '/Users/zhaoyuechao/Desktop/academic/Projects/ISMTS/test_data/data/covid19_data/train/375_patients.xlsx'  # 请替换为你的文件路径
df = pd.read_excel(file_path)
# 使用向前填充的方法填充PATIENT_ID列中的NaN值
df['PATIENT_ID'].fillna(method='ffill', inplace=True)

# 打印前40行数据
print(df.head(40))

# 确保数据按时间顺序排序
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)

# Call the function
features_df = fe.extract_features_from_dataframe(df, 'RE_DATE', ['eGFR', 'creatinine'], 'PATIENT_ID',fill=True, fill_method='mean')
print(features_df)
features_df.to_csv('../test_data/data/covid19_data/train/375_patients.csv', index=False)

