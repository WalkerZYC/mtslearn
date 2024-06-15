import pandas as pd

def extract_basic_features(values, fill_method='mean', fill=True):
    """Extract basic statistical features from a series of values."""
    # 计算缺失值数量和比例
    missing_count = values.isna().sum()
    missing_ratio = values.isna().mean()

    # 如果 fill 参数为 True，则进行缺失值填补
    if fill:
        if fill_method == 'mean':
            filled_values = values.fillna(values.mean())
        elif fill_method == 'median':
            filled_values = values.fillna(values.median())
        elif fill_method == 'zero':
            filled_values = values.fillna(0)
        else:
            raise ValueError(f"Unknown fill method: {fill_method}")
    else:
        filled_values = values

    # 计算统计特征
    features = {
        'mean': filled_values.mean(),
        'median': filled_values.median(),
        'std': filled_values.std(),
        'min': filled_values.min(),
        'max': filled_values.max(),
        'missing_count': missing_count,
        'missing_ratio': missing_ratio
    }
    return features

def extract_features_from_dataframe(df, time_col, value_cols, group_col, fill=True, fill_method='mean'):
    """Extract features from a dataframe with time series data."""
    grouped = df.groupby(group_col)
    feature_dict = {}

    for name, group in grouped:
        features = {'ID': name}  # 保留ID信息
        for value_col in value_cols:
            values = group.sort_values(by=time_col)[value_col]
            extracted_features = extract_basic_features(values, fill_method=fill_method, fill=fill)
            for feature_name, feature_value in extracted_features.items():
                features[f"{value_col}_{feature_name}"] = feature_value
        feature_dict[name] = features

    return pd.DataFrame.from_dict(feature_dict, orient='index')

# Example usage
data = {
    'PATIENT_ID': [1, 1, 1, 2, 2, 3, 3],
    'RE_DATE': ['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-01', '2020-01-02', '2020-01-01', '2020-01-02'],
    'eGFR': [50, 55, 60, 45, 50, 70, None],
    'creatinine': [1.1, 1.2, 1.3, 1.0, 1.1, 1.4, None]
}

df = pd.DataFrame(data)
df['RE_DATE'] = pd.to_datetime(df['RE_DATE'])

# Call the function without filling missing values
features_df_no_fill = extract_features_from_dataframe(df, 'RE_DATE', ['eGFR', 'creatinine'], 'PATIENT_ID', fill=False)
print("Without filling missing values:")
print(features_df_no_fill)

# Call the function with mean filling method
features_df_mean_fill = extract_features_from_dataframe(df, 'RE_DATE', ['eGFR', 'creatinine'], 'PATIENT_ID', fill=True, fill_method='mean')
print("\nWith mean filling method:")
print(features_df_mean_fill)
