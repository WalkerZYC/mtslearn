import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


# LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


# 数据处理类
class LSTMDataProcessor:
    def __init__(self, df, id_col, time_col, value_cols, outcome_col, fill_missing=False, fill_method='ffill'):
        self.df = df
        self.id_col = id_col
        self.time_col = time_col
        self.value_cols = value_cols
        self.outcome_col = outcome_col

        # 处理缺失值
        if fill_missing:
            for col in value_cols + [outcome_col]:
                if col in self.df.columns:
                    if fill_method == 'ffill':
                        self.df[col].ffill(inplace=True)
                    elif fill_method == 'bfill':
                        self.df[col].bfill(inplace=True)
                    elif fill_method == 'mean':
                        self.df[col].fillna(self.df[col].mean(), inplace=True)
                    elif fill_method == 'median':
                        self.df[col].fillna(self.df[col].median(), inplace=True)
                    elif fill_method == 'zero':
                        self.df[col].fillna(0, inplace=True)
                    else:
                        raise ValueError("Unsupported fill method. Use 'ffill', 'bfill', 'mean', 'median', or 'zero'.")

        # 检查是否有缺失值
        if self.df[value_cols + [outcome_col]].isnull().any().any():
            raise ValueError("Data contains NaN values after filling. Please check the input data.")

    def preprocess_data(self, sequence_length=10, test_size=0.2):
        # 排序数据并标准化
        self.df.sort_values(by=[self.id_col, self.time_col], inplace=True)
        scaler = StandardScaler()
        self.df[self.value_cols] = scaler.fit_transform(self.df[self.value_cols])

        # 构造输入序列和标签
        data, labels = [], []
        ids = self.df[self.id_col].unique()

        for unique_id in ids:
            sub_df = self.df[self.df[self.id_col] == unique_id]
            values = sub_df[self.value_cols].values
            outcomes = sub_df[self.outcome_col].values

            for i in range(len(sub_df) - sequence_length):
                data.append(values[i:i + sequence_length])
                labels.append(outcomes[i + sequence_length])

        data = np.array(data)
        labels = np.array(labels)

        # 数据划分为训练集和测试集
        split_idx = int(len(data) * (1 - test_size))
        train_data, test_data = data[:split_idx], data[split_idx:]
        train_labels, test_labels = labels[:split_idx], labels[split_idx:]

        # 转换为Tensor数据
        train_dataset = TensorDataset(torch.tensor(train_data, dtype=torch.float32),
                                      torch.tensor(train_labels, dtype=torch.float32))
        test_dataset = TensorDataset(torch.tensor(test_data, dtype=torch.float32),
                                     torch.tensor(test_labels, dtype=torch.float32))

        return DataLoader(train_dataset, batch_size=32, shuffle=True), DataLoader(test_dataset, batch_size=32)

    def train_and_evaluate(self, hidden_size=64, num_layers=1, num_epochs=20, learning_rate=0.001, sequence_length=10,
                           test_size=0.2):
        train_loader, test_loader = self.preprocess_data(sequence_length, test_size)

        input_size = len(self.value_cols)
        output_size = 1  # Assuming binary classification or regression task
        model = LSTMModel(input_size, hidden_size, output_size, num_layers)

        criterion = nn.BCEWithLogitsLoss()  # Binary classification
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Training the model
        model.train()
        for epoch in range(num_epochs):
            for inputs, labels in train_loader:
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

        # Evaluating the model
        model.eval()
        predictions, actuals = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                preds = torch.sigmoid(outputs).squeeze().round()  # Use 0.5 as threshold
                predictions.append(preds.numpy())
                actuals.append(labels.numpy())

        predictions = np.concatenate(predictions)
        actuals = np.concatenate(actuals)

        accuracy = accuracy_score(actuals, predictions)
        print(f'Accuracy: {accuracy:.4f}')
        return accuracy


# Example usage with a dataset
file_path = '/Users/zhaoyuechao/Documents/Research/Projects/mtslearn/tests/test_data/375_patients_example.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Initialize the processor with options for filling missing data
processor = LSTMDataProcessor(df, id_col='PATIENT_ID', time_col='RE_DATE',
                              value_cols=['eGFR', 'creatinine'], outcome_col='outcome',
                              fill_missing=True, fill_method='mean')  # Change fill_method as needed

# Call the pipeline to train LSTM and evaluate the model
accuracy = processor.train_and_evaluate(hidden_size=64, num_layers=2, num_epochs=20, learning_rate=0.001,
                                        sequence_length=10)
print(f'Model Accuracy: {accuracy:.4f}')
