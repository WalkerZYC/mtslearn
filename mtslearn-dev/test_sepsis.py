import mtslearn.feature_extraction as fe

import pandas as pd
import os
import glob


def load_psv_files(directory):
    """
    从指定目录加载所有的 .psv 文件并合并为一个 DataFrame

    Parameters:
    directory (str): .psv 文件所在的目录路径

    Returns:
    pd.DataFrame: 合并后的 DataFrame
    """
    all_files = glob.glob(os.path.join(directory, "*.psv"))
    li = []

    for filename in all_files:
        df = pd.read_csv(filename, sep='|')
        li.append(df)

    frame = pd.concat(li, axis=0, ignore_index=True)
    return frame


# 使用示例
directory = '../test_data/data/sepsis_data/set_A'
combined_df = load_psv_files(directory)

print("合并后的数据:")
print(combined_df.head(40))

# 保存为 CSV 文件
combined_df.to_csv('../test_data/data/sepsis_data/setA.csv', index=False)







# 加载数据
df = pd.read_csv('data.csv')

print("原始数据帧:")
print(df)

# 从数据帧中提取特征并处理缺失值
features_df = fe.extract_features_from_dataframe(df, 'time', 'value', 'id', fill_method='mean')

print("\n提取的特征数据帧:")
print(features_df)
