import pandas as pd
import os
import mtslearn.feature_extraction as fe
import mtslearn as ls

# 示例数据
data = {
    'PATIENT_ID': [1, 1, 1, 2, 2, 3, 3],
    'RE_DATE': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01', '2020-01-02', '2020-01-01', '2020-01-02'],
    'eGFR': [50, 55, 60, 45, 50, 70, None],
    'creatinine': [1.1, 1.2, 1.3, 1.0, 1.1, 1.4, None],
    'outcome': [1, 0, 1, 0, 0, 1, 0]  # 目标变量
}

df = pd.DataFrame(data)
df['RE_DATE'] = pd.to_datetime(df['RE_DATE'])

fe = fe.FeModEvaluator(df,
                       'PATIENT_ID',
                       'RE_DATE',
                       'outcome',
                       {'eGFR': ['mean', 'max'], 'creatinine': ['mean']},
                        include_duration=True)

fe.run(model_type='logit', fill=True, fill_method='mean', cross_val=True, plot_importance=True)
