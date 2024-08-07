### Input Data Format Guide for Feature Extraction Package

To use the `FeModEvaluator` class effectively, your input data should adhere to the following format. Below is a guide based on a provided sample dataset.

#### Sample Data Format

The dataset should be structured as a pandas DataFrame with the following columns:

- **Patient_ID**: A unique identifier for each patient, `PATIENT_ID` in the example.
- **Record_Time**: The date and time of the record, `RE_DATE` in the example, in a format that can be parsed by `pandas.to_datetime` (e.g., `YYYY-MM-DD HH:MM:SS`).
- **Outcome**: The outcome variable, indicating the result of the treatment or condition.
- **Clinical Measurements**: Clinical data, such as lab values or measurements (e.g., `eGFR`,`creatinine` in the example, or hemoglobin, serum sodium, etc.).

##### Example Rows

| PATIENT_ID | RE_DATE               | outcome | eGFR | creatinine | ... (other measurements) |
|------------|-----------------------|---------|------|------------|--------------------------|
| 1          | 2020-01-31 01:09:00   | 0       | NaN  | NaN        | ...                      |
| 1          | 2020-01-31 01:25:00   | 0       | NaN  | 136.0      | ...                      |
| 1          | 2020-01-31 01:44:00   | 0       | 46.6 | 130.0      | ...                      |
| ...        | ...                   | ...     | ...  | ...        | ...                      |

### Usage

To utilize this data structure with the `FeModEvaluator` class, you need to initialize the class as follows:

```python

import mtslearn.feature_extraction as fe
fe = fe.FeModEvaluator(df, 'PATIENT_ID', 'RE_DATE',  'outcome', ['eGFR', 'creatinine'],['mean','max'],include_duration=True)
```

#### Parameters:

- **df**: The DataFrame containing your patient data.
- **group_col**: Column name to group the data by (e.g., patient ID).
- **time_col**: Column name representing the time of each record. This column should be in a format that can be parsed by `pandas.to_datetime`. 
  - Common formats include:
    - `YYYY-MM-DD`
    - `YYYY-MM-DD HH:MM:SS`
    - `MM/DD/YYYY`
    - `MM/DD/YYYY HH:MM:SS`
    
- **outcome_col**: Column name representing the outcome variable.
- **value_cols**: List of columns to extract features from.
- **selected_features**: List of selected features for model training (optional).
- **include_duration**: Boolean indicating whether to include the duration feature.


### Example of Initialization

You can  download our example data at https://github.com/WalkerZYC/mtslearn/blob/main/test_data/data/covid19_data/train/375_patients_example.xlsx

```python
import pandas as pd
import mtslearn.feature_extraction as fe

# Uplode Excel Data
file_path = '.../375_patients_example.xlsx' # Replece with your file path
df = pd.read_excel(file_path)

# Class Initialization
fe = FeModEvaluator(df, 'PATIENT_ID', 'RE_DATE', 'outcome', ['eGFR', 'creatinine'], ['mean', 'max'], include_duration=True)
```

This guide provides the essential format and usage details for integrating your dataset with the `FeModEvaluator` class. 