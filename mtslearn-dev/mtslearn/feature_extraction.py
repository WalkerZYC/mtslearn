import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import k_fold_cross_validation
import warnings
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LassoCV
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE

# Define a class for feature extraction and model evaluation
class FeatureExtractorAndModelEvaluator:
    def __init__(self, df, group_col, time_col, outcome_col, value_cols, selected_features=None):
        # Initialize the data frame and column information
        self.df = df
        self.group_col = group_col
        self.time_col = time_col
        self.outcome_col = outcome_col
        self.value_cols = value_cols
        self.selected_features = selected_features if selected_features else value_cols

    def extract_basic_features(self, values, fill_method='mean', fill=True):
        """Extract basic statistical features from a series of values."""
        # Calculate the number and ratio of missing values
        missing_count = values.isna().sum()
        total_count = len(values)
        missing_ratio = missing_count / total_count

        # Fill missing values if required
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

        # Calculate the difference between the last and the first measurements
        diff_last_first = filled_values.iloc[-1] - filled_values.iloc[0]

        # Extract various statistical features
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
        # Group the data frame by the group (ID) column and extract features for each group
        grouped = self.df.groupby(self.group_col)
        feature_dict = {}

        for name, group in grouped:
            features = {'ID': name}
            for value_col in self.value_cols:
                values = group.sort_values(by=self.time_col)[value_col]
                extracted_features = self.extract_basic_features(values, fill_method=fill_method, fill=fill)
                for feature_name, feature_value in extracted_features.items():
                    features[f"{value_col}_{feature_name}"] = feature_value

            # Extract the outcome for the group
            # In the current model, if one patient has outcome=1 at any time points, then the outcome for him/her is 1
            outcome_value = group[self.outcome_col].max()  # Assuming outcome is 1 if any entry is 1
            features[self.outcome_col] = outcome_value

            # Calculate duration as the difference between the first and last measurement times
            first_time = group[self.time_col].min()
            last_time = group[self.time_col].max()
            duration = (pd.to_datetime(last_time) - pd.to_datetime(first_time)).days
            features['duration'] = duration

            feature_dict[name] = features

        return pd.DataFrame.from_dict(feature_dict, orient='index')

    def prepare_data(self, fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False):
        # Extract features and fill missing values
        features_df = self.extract_features_from_dataframe(fill=fill, fill_method=fill_method)

        # Print the first few rows of the features data frame
        print("Features DataFrame (First 5 lines):")
        print(features_df.head(5))

        # Fill NaN values in the data frame
        imputer = SimpleImputer(strategy=fill_method)
        features_df = pd.DataFrame(imputer.fit_transform(features_df), columns=features_df.columns)

        # Select the required feature columns
        selected_columns = [col for col in features_df.columns if
                            any(col.endswith(feature_type) for feature_type in self.selected_features)]

        # Ensure 'duration' is included in the selected columns
        selected_columns += ['duration']

        X = features_df[selected_columns].copy()  # Create a copy to avoid SettingWithCopyWarning
        y = features_df[self.outcome_col].copy()  # Create a copy to avoid SettingWithCopyWarning

        if cross_val:
            # If cross-validation is required, return the features, target variable, and StratifiedKFold object
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            return X, y, skf
        else:
            # Otherwise, split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

            if balance_data:
                # If data balancing is required, use SMOTE to oversample the minority class
                smote = SMOTE(random_state=42)
                X_train, y_train = smote.fit_resample(X_train, y_train)

            return X_train, X_test, y_train, y_test

    def evaluate_model(self, model, X_test, y_test, y_prob):
        # Predict the target variable for the test set
        y_pred = model.predict(X_test)

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Check if there are two classes in y_test
        if len(set(y_test)) > 1:
            auc = roc_auc_score(y_test, y_prob)
            print(f"AUC: {auc}")

            # Plot the ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f"Model (AUC = {auc:.2f})")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            plt.show()
            plt.close()  # Close the figure to release resources
        else:
            print("Only one class present in y_test. ROC AUC score is not defined.")

        # Plot the confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()
        plt.close()  # Close the figure to release resources

    def evaluate_lasso_model(self, model, X_test, y_test, y_pred):
        # Calculate evaluation metrics for the Lasso model
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        # Plot the actual vs predicted values
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred, alpha=0.3)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        plt.show()

    def run(self, model_type='logit', fill=True, fill_method='mean', test_size=0.2, balance_data=True, cross_val=False,
            n_splits=5):
        if cross_val:
            # Prepare data for cross-validation
            X, y, skf = self.prepare_data(fill=fill, fill_method=fill_method, balance_data=balance_data, cross_val=True)
            if model_type == 'logit':
                # Logistic Regression model with cross-validation
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
                # Cox Proportional Hazards model with cross-validation
                cox_model = CoxPHFitter()
                X['outcome'] = y
                cv_results = k_fold_cross_validation(cox_model, X, duration_col='duration', event_col='outcome',
                                                     k=n_splits)
                print("Cox Model Cross-Validation Results:")
                print(cv_results)
                print("\nDetails of each fold:")
                # Output log-likelihood results for each fold
                print("Cox Model Cross-Validation Log-Likelihood Results:")
                for i, log_likelihood in enumerate(cv_results):
                    print(f"Fold {i + 1} Log-Likelihood: {log_likelihood}")
                print(f"Mean Log-Likelihood: {np.mean(cv_results)}")
                # Calculate and output concordance index
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
                # XGBoost model with cross-validation
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
            elif model_type == 'lasso':
                # Lasso Regression model with cross-validation
                model = LassoCV(cv=n_splits)
                mse_scores = cross_val_score(model, X, y, cv=n_splits, scoring='neg_mean_squared_error')
                r2_scores = cross_val_score(model, X, y, cv=n_splits, scoring='r2')

                print(f"Cross-Validated Mean Squared Error: {-mse_scores.mean()} +/- {mse_scores.std()}")
                print(f"Cross-Validated R2 Score: {r2_scores.mean()} +/- {r2_scores.std()}")
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            # Prepare data for training and testing
            X_train, X_test, y_train, y_test = self.prepare_data(fill=fill, fill_method=fill_method,
                                                                 test_size=test_size, balance_data=balance_data)

            if model_type == 'logit':
                # Logistic Regression model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                self.evaluate_model(model, X_test, y_test, y_prob)
            elif model_type == 'cox':
                # Cox Proportional Hazards model
                cox_model = CoxPHFitter()
                X_train['outcome'] = y_train
                cox_model.fit(X_train, duration_col='duration', event_col='outcome')

                # Use partial hazard function for prediction
                cox_pred = cox_model.predict_partial_hazard(X_test)
                y_prob = cox_pred.values.flatten()  # Convert DataFrame to 1D array

                # Manually calculate evaluation metrics
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
                    plt.close()  # Close the figure to release resources
                else:
                    print("Only one class present in y_test. ROC AUC score is not defined.")

                cm = confusion_matrix(y_test, y_pred)
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.xlabel("Predicted")
                plt.ylabel("Actual")
                plt.title("Confusion Matrix")
                plt.show()
                plt.close()  # Close the figure to release resources
            elif model_type == 'xgboost':
                # XGBoost model
                model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
                model.fit(X_train, y_train)
                y_prob = model.predict_proba(X_test)[:, 1]
                self.evaluate_model(model, X_test, y_test, y_prob)
            elif model_type == 'lasso':
                # Lasso Regression model
                model = LassoCV(cv=5)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                self.evaluate_lasso_model(model, X_test, y_test, y_pred)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
