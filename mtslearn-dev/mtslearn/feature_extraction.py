import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import shap
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error, r2_score
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LassoCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import warnings
import random

warnings.filterwarnings('ignore')

class FeatureExtractorAndModelEvaluator:
    def __init__(self, df, group_col, time_col, outcome_col, value_cols, selected_features=None, include_duration=True):
        """
        Initialize the FeatureExtractorAndModelEvaluator class.

        Parameters:
        - df: DataFrame containing the data.
        - group_col: Column name to group the data by (e.g., patient ID).
        - time_col: Column name representing the time of each record.
        - outcome_col: Column name representing the outcome variable.
        - value_cols: List of columns to extract features from.
        - selected_features: List of selected features for model training (optional).
        - include_duration: Boolean indicating whether to include the duration feature.
        """
        self.df = df
        self.group_col = group_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.value_cols = value_cols
        self.selected_features = selected_features if selected_features else value_cols
        self.include_duration = include_duration

    def extract_basic_features(self, values, fill_method='mean', fill=True):
        """
        Extract basic statistical features from a series of values.

        Parameters:
        - values: Series of values to extract features from.
        - fill_method: Method to fill missing values ('mean', 'median', or 'zero').
        - fill: Boolean indicating whether to fill missing values.

        Returns:
        - A dictionary of extracted features.
        """
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

        diff_last_first = filled_values.iloc[-1] - filled_values.iloc[0]

        features = {
            'mean': filled_values.mean(),
            'median': filled_values.median(),
            'std': filled_values.std(),
            'min': filled_values.min(),
            'max': filled_values.max(),
            'diff_last_first': diff_last_first,
            'missing_count': missing_count,
            'missing_ratio': missing_ratio
        }
        return features

    def extract_features_from_dataframe(self, fill=True, fill_method='mean'):
        """
        Extract features from the entire DataFrame grouped by the group column.

        Parameters:
        - fill: Boolean indicating whether to fill missing values.
        - fill_method: Method to fill missing values ('mean', 'median', or 'zero').

        Returns:
        - A DataFrame of extracted features.
        """
        grouped = self.df.groupby(self.group_col)
        feature_dict = {}

        for name, group in grouped:
            features = {'ID': name}
            for value_col in self.value_cols:
                values = group.sort_values(by=self.time_col)[value_col]
                extracted_features = self.extract_basic_features(values, fill_method=fill_method, fill=fill)
                for feature_name, feature_value in extracted_features.items():
                    features[f"{value_col}_{feature_name}"] = feature_value

            outcome_value = group[self.outcome_col].max()
            features[self.outcome_col] = outcome_value

            first_time = group[self.time_col].min()
            last_time = group[self.time_col].max()
            duration = (pd.to_datetime(last_time) - pd.to_datetime(first_time)).days
            features['duration'] = duration

            feature_dict[name] = features

        return pd.DataFrame.from_dict(feature_dict, orient='index')

    def prepare_data(self, fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False):
        """
        Prepare the data for model training and evaluation.

        Parameters:
        - fill: Boolean indicating whether to fill missing values.
        - fill_method: Method to fill missing values ('mean', 'median', or 'zero').
        - test_size: Proportion of the data to use for testing.
        - balance_data: Boolean indicating whether to apply SMOTE for class balancing.
        - cross_val: Boolean indicating whether to perform cross-validation.

        Returns:
        - Depending on cross_val, returns either training and test sets or data and cross-validation strategy.
        """
        features_df = self.extract_features_from_dataframe(fill=fill, fill_method=fill_method)

        print("Features DataFrame (First 5 lines):")
        print(features_df.head(5))

        imputer = SimpleImputer(strategy=fill_method)
        features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

        selected_columns = [col for col in features_df.columns if
                            any(col.endswith(feature_type) for feature_type in self.selected_features)]

        if self.include_duration:
            selected_columns += ['duration']

        X = features_df[selected_columns].copy()
        y = features_df[self.outcome_col].copy()

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
        """
        Evaluate the performance of a classification model.

        Parameters:
        - model: The trained model.
        - X_test: The test features.
        - y_test: The true labels for the test set.
        - y_prob: The predicted probabilities from the model.
        """
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
        print("Confusion Matrix:")
        print(cm)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 16}, vmin=0, vmax=max(cm.max(), 1))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        plt.close()

    def evaluate_lasso_model(self, model, X_test, y_test, y_pred):
        """
        Evaluate the performance of a Lasso regression model.

        Parameters:
        - model: The trained Lasso model.
        - X_test: The test features.
        - y_test: The true labels for the test set.
        - y_pred: The predicted values from the model.
        """
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()

    def describe_data(self, plot_type, value_col=None, feature1=None, feature2=None):
        """
        Describe the data using various plots.

        Parameters:
        - plot_type: Type of plot ('boxplot', 'violinplot', 'histogram', 'correlation_matrix').
        - value_col: Feature to visualize for single feature plots.
        - feature1, feature2: Features to visualize for correlation matrix.
        """
        if plot_type in ['boxplot', 'violinplot', 'histogram']:
            if value_col is None:
                raise ValueError("For plot types 'boxplot', 'violinplot', and 'histogram', value_col must be provided.")

            # Call the appropriate plotting method based on the plot type
            if plot_type == 'boxplot':
                self.plot_boxplot(value_col=value_col)
            elif plot_type == 'violinplot':
                self.plot_violinplot(value_col=value_col)
            elif plot_type == 'histogram':
                self.plot_histogram(value_col=value_col)
        elif plot_type == 'correlation_matrix':
            if feature1 is None or feature2 is None:
                raise ValueError("For 'correlation_matrix' plot_type, feature1 and feature2 must be provided.")
            self.plot_correlation_matrix(feature1=feature1, feature2=feature2)
        else:
            raise ValueError(f"Invalid plot type: {plot_type}")

    def plot_boxplot(self, value_col):
        # Create a boxplot for the specified column
        plt.figure(figsize=(10, 6))
        sns.boxplot(y=self.df[value_col])
        plt.title(f'Boxplot for {value_col}')
        plt.ylabel(value_col)
        plt.show()

    def plot_violinplot(self, value_col):
        # Create a violin plot for the specified column
        plt.figure(figsize=(10, 6))
        sns.violinplot(y=self.df[value_col])
        plt.title(f'Violin Plot for {value_col}')
        plt.ylabel(value_col)
        plt.show()

    def plot_histogram(self, value_col):
        # Create a histogram for the specified column
        plt.figure(figsize=(10, 6))
        plt.hist(self.df[value_col].dropna(), bins=20, alpha=0.7)
        plt.xlabel(value_col)
        plt.ylabel('Frequency')
        plt.title(f'Histogram for {value_col}')
        plt.show()

    def plot_correlation_matrix(self, feature1, feature2):
        # Create a correlation matrix for the specified features
        if feature1 and feature2:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df[[feature1, feature2]].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5)
            plt.title('Correlation Matrix')
            plt.show()
        else:
            print("Both feature1 and feature2 must be provided for correlation_matrix.")

    def plot_feature_importance(self, model, feature_names):
        # Plot feature importance for the given model
        if hasattr(model, 'feature_importances_'):  # For XGBoost
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):  # For Logistic Regression and Lasso
            importance = np.abs(model.coef_[0])
        else:
            raise ValueError("Model does not have feature importance attribute.")

        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance, y=feature_names)
        plt.title('Feature Importance')
        plt.show()

    def plot_shap_values(self, model, X):
        # Plot SHAP values for tree-based models
        if not hasattr(model, 'feature_importances_'):
            raise ValueError("SHAP values are only available for tree-based models like XGBoost.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, X, feature_names=X.columns)

    def run(self, model_type='logit', fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False,
            n_splits=5, plot_importance=False):
        # Main method to run the model training and evaluation
        if cross_val:
            X, y, skf = self.prepare_data(fill=fill, fill_method=fill_method, balance_data=balance_data, cross_val=True)
            if model_type == 'logit':
                model = LogisticRegression(max_iter=1000)
                accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
                recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
                f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
                auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

                print(f"Cross-Validated Accuracy: {accuracy_scores.mean()} +/- {accuracy_scores.std()}")
                print(f"Cross-Validated Precision: {precision_scores.mean()} +/- {precision_scores.std()}")
                print(f"Cross-Validated Recall: {recall_scores.mean()} +/- {recall_scores.std()}")
                print(f"Cross-Validated F1 Score: {f1_scores.mean()} +/- {f1_scores.std()}")
                print(f"Cross-Validated AUC: {auc_scores.mean()} +/- {auc_scores.std()}")
            elif model_type == 'cox':
                cox_model = CoxPHFitter()
                X['outcome'] = y
                cv_results = k_fold_cross_validation(cox_model, X, duration_col='duration', event_col='outcome',
                                                     k=n_splits)
                print("Cox Model Cross-Validation Results:")
                print(cv_results)
                print("\nDetails of each fold:")
                print("Cox Model Cross-Validation Log-Likelihood Results:")
                for i, log_likelihood in enumerate(cv_results):
                    print(f"Fold {i + 1} Log-Likelihood: {log_likelihood}")
                print(f"Mean Log-Likelihood: {np.mean(cv_results)}")
                concordance_indices = []
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
                    cox_model.fit(X_train, duration_col='duration', event_col='outcome')
                    concordance_index = cox_model.score(X_test)
                    concordance_indices.append(concordance_index)

                print(
                    f"Cross-Validated Concordance Index: {np.mean(concordance_indices)} +/- {np.std(concordance_indices)}")
            elif model_type == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
                precision_scores = cross_val_score(model, X, y, cv=skf, scoring='precision')
                recall_scores = cross_val_score(model, X, y, cv=skf, scoring='recall')
                f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
                auc_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')

                print(f"Cross-Validated Accuracy: {accuracy_scores.mean()} +/- {accuracy_scores.std()}")
                print(f"Cross-Validated Precision: {precision_scores.mean()} +/- {precision_scores.std()}")
                print(f"Cross-Validated Recall: {recall_scores.mean()} +/- {recall_scores.std()}")
                print(f"Cross-Validated F1 Score: {f1_scores.mean()} +/- {f1_scores.std()}")
                print(f"Cross-Validated AUC: {auc_scores.mean()} +/- {auc_scores.std()}")

                # Plot feature importance for XGBoost model
                if plot_importance:
                    self.plot_feature_importance(model, X.columns)

                # Plot SHAP values for XGBoost model
                if plot_importance:
                    self.plot_shap_values(model, X)
            elif model_type == 'lasso':
                model = LassoCV(cv=n_splits)
                mse_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error')
                r2_scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')

                print(f"Cross-Validated Mean Squared Error: {-mse_scores.mean()} +/- {mse_scores.std()}")
                print(f"Cross-Validated R2 Score: {r2_scores.mean()} +/- {r2_scores.std()}")
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
                if plot_importance:
                    self.plot_feature_importance(model, X_train.columns)
            elif model_type == 'cox':
                cox_model = CoxPHFitter()
                X_train['outcome'] = y_train
                cox_model.fit(X_train, duration_col='duration', event_col='outcome')
                cox_pred = cox_model.predict_partial_hazard(X_test)
                y_prob = cox_pred.values.flatten()
                y_pred = (y_prob > y_prob.mean()).astype(int)
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
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, annot_kws={"size": 16}, vmin=0, vmax=max(cm.max(), 1))
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                plt.show()
                plt.close()
            elif model_type == 'xgboost':
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                self.evaluate_model(model, X_test, y_test, y_prob)

                # Plot feature importance for XGBoost model
                if plot_importance:
                    self.plot_feature_importance(model, X_train.columns)

                # Plot SHAP values for XGBoost model
                if plot_importance:
                    self.plot_shap_values(model, X_train)
            elif model_type == 'lasso':
                model = LassoCV(cv=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.evaluate_lasso_model(model, X_test, y_test, y_pred)
                if plot_importance:
                    self.plot_feature_importance(model, X_train.columns)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
