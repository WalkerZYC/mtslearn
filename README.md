# **Medical Irregular Time-Series Data Analysis Toolkit**
## **Overview**

The Medical Time-Series Data Analysis Toolkit `mtslearn` is designed to empower healthcare professionals and researchers with tools to analyze and interpret time-series medical data. It offers a comprehensive set of features for extracting key clinical metrics, preparing data for modeling, evaluating predictive models, and visualizing the results. The toolkit is specifically tailored to handle complex datasets, such as longitudinal irregular sampled patient records, and provides meaningful insights to support informed clinical decision-making.

## **Features**

- **Feature Extraction**: Automatically extract meaningful features from time-series data, including statistical measures and temporal dynamics.
- **Data Preparation**: Handle missing data, balance datasets, and split data into training and testing sets with ease.
- **Model Evaluation**: Supports multiple model types (Logistic Regression, Cox Proportional Hazards, XGBoost, Lasso) and evaluates model performance with key metrics.
- **Visualization**: Generate visualizations such as boxplots and correlation matrices to help interpret clinical data and model outcomes.
## **Installation**
### **Clone the Repository**
To download and use the toolkit from GitHub, start by cloning the repository:
```
git clone https://github.com/WalkerZYC/mtslearn.git
cd mtslearn
```
### **Install Dependencies**
Next, install the required dependencies:
```
pip install -r requirements.txt
```
Alternatively, you can manually install the necessary Python packages:
```
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lifelines imbalanced-learn
```
## **Quickstart**
### **1. Prepare Your Data**
Ensure your data is in a pandas DataFrame with the following structure:

- `Patient_ID`: Unique identifier for each patient.
- `Record_Time`: Timestamp of the record.
- `Outcome`: Outcome variable, indicating the result of treatment or condition.
- `Clinical Measurements`: Relevant clinical data (e.g., lab values, vital signs).

Example:
```python
import pandas as pd

# Load your data
df = pd.read_excel('path/to/your/375_patients_example.xlsx')

# Sort by patient ID and timestamp
df.sort_values(by=['PATIENT_ID', 'RE_DATE'], inplace=True)
```
### **2. Initialize the Toolkit**
```python
import mtslearn.feature_extraction as fe

# Initialize the feature extraction and evaluation tool
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
### **3. Run the Analysis Pipeline**
```python
# Run the pipeline with XGBoost
fe.run(
    model_type='xgboost',
    fill=True,
    fill_method='mean',
    test_size=0.3,
    balance_data=True
)
```
### **4. Visualize Results**
```python
# Boxplot for a specific clinical measurement
fe.describe_data(plot_type='boxplot', value_col='eGFR')

# Correlation matrix between two clinical measurements
fe.describe_data(plot_type='correlation_matrix', feature1='eGFR', feature2='creatinine')
```
## **Documentation**
For detailed documentation, including advanced usage, customization options, and examples, refer to the [User Guide](./User Guide.md).
## **License**
This project is licensed under the MIT License. See the [LICENSE](mtslearn-dev/LICENSE) file for details.
## **Contact**
For any questions or issues, please open an issue on GitHub or contact us at zycwalker11@gmail.com.
