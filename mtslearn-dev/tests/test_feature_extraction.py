import pandas as pd
import unittest


# 定义 extract_basic_features 函数
def extract_basic_features(values, fill_method='mean'):
    """Extract basic statistical features from a series of values."""
    if fill_method == 'mean':
        filled_values = values.fillna(values.mean())
    elif fill_method == 'median':
        filled_values = values.fillna(values.median())
    else:
        filled_values = values.fillna(0)

    features = {
        'mean': filled_values.mean(),
        'median': filled_values.median(),
        'std': filled_values.std(),
        'min': filled_values.min(),
        'max': filled_values.max(),
    }
    return features


# 定义 extract_features_from_dataframe 函数
def extract_features_from_dataframe(df, time_col, value_col, group_col, fill_method='mean'):
    """Extract features from a dataframe with time series data."""
    grouped = df.groupby(group_col)
    feature_dict = {}

    for name, group in grouped:
        values = group.sort_values(by=time_col)[value_col]
        features = extract_basic_features(values, fill_method=fill_method)
        feature_dict[name] = features

    return pd.DataFrame.from_dict(feature_dict, orient='index')


# 加载Excel文件
file_path = '/Users/zhaoyuechao/Desktop/academic/Projects/ISMTS/test_data/data/covid19_data/train/375_patients.xlsx'
df = pd.read_excel(file_path)

# 确保数据按时间顺序排序
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)


# 单元测试
class TestFeatureExtraction(unittest.TestCase):
    def test_extract_features(self):
        # 测试 extract_features_from_dataframe 函数
        features_df = extract_features_from_dataframe(df, 'RE_DATE', 'eGFR', 'PATIENT_ID', fill_method='mean')

        # 检查返回的DataFrame是否包含预期的列
        expected_columns = ['mean', 'median', 'std', 'min', 'max']
        self.assertTrue(all(col in features_df.columns for col in expected_columns))

        # 检查返回的DataFrame是否包含预期的行
        self.assertTrue(len(features_df) > 0)


if __name__ == '__main__':
    unittest.main()
