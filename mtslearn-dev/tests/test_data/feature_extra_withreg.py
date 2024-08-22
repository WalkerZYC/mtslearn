import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, \
    confusion_matrix
from statsmodels.api import OLS, Logit, add_constant
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from imblearn.over_sampling import SMOTE


class FeatureExtractorAndModelEvaluator:
    def __init__(self, df, time_col, value_cols, group_col, outcome_col):
        self.df = df
        self.time_col = time_col
        self.value_cols = value_cols
        self.group_col = group_col
        self.outcome_col = outcome_col

    def extract_basic_features(self, values, fill_method='mean', fill=True):
        """Extract basic statistical features from a series of values."""
        missing_count = values.isna().sum()
        total_count = len(values)
        missing_ratio = missing_count / total_count

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

    def extract_features_from_dataframe(self, fill=True, fill_method='mean'):
        grouped = self.df.groupby(self.group_col)
        feature_dict = {}

        for name, group in grouped:
            features = {'ID': name}
            outcome_value = group[self.outcome_col].iloc[0]  # 假设每组的 outcome 值是相同的
            features[self.outcome_col] = outcome_value
            for value_col in self.value_cols:
                values = group.sort_values(by=self.time_col)[value_col]
                extracted_features = self.extract_basic_features(values, fill_method=fill_method, fill=fill)
                for feature_name, feature_value in extracted_features.items():
                    features[f"{value_col}_{feature_name}"] = feature_value
            feature_dict[name] = features

        return pd.DataFrame.from_dict(feature_dict, orient='index')

    def prepare_data(self, fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False):
        features_df = self.extract_features_from_dataframe(fill=fill, fill_method=fill_method)
        X = features_df.drop(columns=[self.outcome_col])
        y = features_df[self.outcome_col]

        # 使用SimpleImputer填充缺失值
        imputer = SimpleImputer(strategy=fill_method)
        X = imputer.fit_transform(X)

        if cross_val:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return X, y, skf
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if balance_data:
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test, y_prob):
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        if len(set(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
            print(f"AUC: {auc}")

            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.2f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()
            plt.close()
        else:
            print("Only one class present in y_test. ROC AUC score is not defined.")

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        plt.close()

    def run(self, task='prediction', model_type='logit', fill=True, fill_method='mean', test_size=0.2,
            balance_data=True, cross_val=False):
        if task == 'prediction':
            if cross_val:
                X, y, skf = self.prepare_data(fill=fill, fill_method=fill_method, balance_data=balance_data,
                                              cross_val=True)
                if model_type == 'logit':
                    model = LogisticRegression(max_iter=1000)
                    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                    print(f'Cross-Validation Accuracy Scores: {scores}')
                    print(f'Mean Accuracy: {scores.mean()}')
                elif model_type == 'cox':
                    print(
                        "Cox model is not typically used with cross-validation in the same way as other models. Consider using prediction or regression task with train/test split for Cox model.")
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
            else:
                X_train, X_test, y_train, y_test = self.prepare_data(fill=fill, fill_method=fill_method,
                                                                     test_size=test_size, balance_data=balance_data)

                if model_type == 'logit':
                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)
                    y_prob = model.predict_proba(X_test)[:, 1]
                    self.evaluate_model(model, X_test, y_test, y_prob)
                elif model_type == 'cox':
                    cox_model = CoxPHFitter()
                    X_train['outcome'] = y_train
                    cox_model.fit(X_train, duration_col='duration', event_col='event')
                    predicted_survival_prob = cox_model.predict_survival_function(X_test).loc[0.5]
                    y_prob = 1 - predicted_survival_prob
                    y_pred = (predicted_survival_prob > 0.5).astype(int)
                    self.evaluate_model(cox_model, X_test, y_test, y_prob)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
        elif task == 'regression':
            features_df = self.extract_features_from_dataframe(fill=fill, fill_method=fill_method)
            X = features_df.drop(columns=[self.outcome_col])
            y = features_df[self.outcome_col]
            X = add_constant(X)  # 添加常量项

            if model_type == 'ols':
                model = OLS(y, X).fit()
            elif model_type == 'logit':
                model = Logit(y, X).fit()
            elif model_type == 'cox':
                cox_model = CoxPHFitter()
                cox_model.fit(features_df, duration_col='duration', event_col='event')
                model = cox_model
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            if model_type != 'cox':
                print(model.summary())
            else:
                print(model.summary)
        else:
            raise ValueError(f"Unknown task: {task}")




