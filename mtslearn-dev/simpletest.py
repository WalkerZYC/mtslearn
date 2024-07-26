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

# 创建并运行特征提取和模型评估器
evaluator = FeatureExtractorAndModelEvaluator(df, 'RE_DATE', ['eGFR', 'creatinine'], 'PATIENT_ID', 'outcome')
evaluator.run(model_type='logit', fill=True, fill_method='mean', test_size=0.2)