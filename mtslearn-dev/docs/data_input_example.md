### Input Data Format Guide for Feature Extraction Package

**Properly importing the data is essential for you to use this package.** 

To use `FeModEvaluator` class effectively, your input data should adhere to the following format. Below is a guide based on a provided sample dataset.

#### Sample Data Format

The dataset should be imported as a **pandas DataFrame** with at least the following columns:

- **Patient_ID**: A unique identifier for each patient, `PATIENT_ID` in the example.
- **Record_Time**: The date and time of the record, `RE_DATE` in the example, in a format that can be parsed by `pandas.to_datetime` (e.g., `YYYY-MM-DD HH:MM:SS`).
- **Outcome**: The outcome variable, indicating the result of the treatment or condition.
- **Clinical Measurements**: Clinical data, such as lab values or measurements (e.g., `eGFR`, `creatinine` in the example, or hemoglobin, serum sodium, etc.).

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

# Class Initialization
fe = fe.FeModEvaluator(
    df=df,
    group_col='PATIENT_ID',
    time_col='RE_DATE',
    outcome_col='outcome',
    features_to_extract={
        'eGFR': ['mean', 'max'],
        'creatinine': ['mean']
    },
    include_duration=True
)
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
- **features_to_extract**: A dictionary where keys are column names to extract features from, and values are lists of features to calculate for each column.
- **include_duration**: Boolean indicating whether to include the duration feature.

### Example of Initialization

You can download our example data at [this link](https://github.com/WalkerZYC/mtslearn/blob/main/test_data/data/covid19_data/train/375_patients_example.xlsx).

```python
import pandas as pd
import mtslearn.feature_extraction as fe

# Upload Excel Data
file_path = '.../375_patients_example.xlsx'  # Replace with your file path
df = pd.read_excel(file_path)

# Class Initialization
fe = FeModEvaluator(
    df=df,
    group_col='PATIENT_ID',
    time_col='RE_DATE',
    outcome_col='outcome',
    features_to_extract={
        'eGFR': ['mean', 'max'],
        'creatinine': ['mean']
    },
    include_duration=True
)
```

This guide provides the essential format and usage details for integrating your dataset with the `FeModEvaluator` class. 