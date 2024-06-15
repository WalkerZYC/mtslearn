
#### `docs/usage.md`
```markdown
# Usage

```python
import mtslearn.feature_extraction as fe
import pandas as pd

# 提取基本统计特征并处理缺失值
data = pd.Series([1, 2, None, 4, 5])
features = fe.extract_basic_features(data, fill_method='mean')
print(features)

# 从数据帧中提取特征并处理缺失值
df = pd.DataFrame({
    'time': [1, 2, 3, 4, 5, 1, 2, 3],
    'value': [1, None, 3, 4, 5, 2, 3, 4],
    'id': [1, 1, 1, 1, 1, 2, 2, 2]
})
features_df = fe.extract_features_from_dataframe(df, 'time', 'value', 'id', fill_method='mean')
print(features_df)
