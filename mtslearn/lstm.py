import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer


def prepare_data_for_lstm(df, value_cols, feature_cols, outcome_col, sequence_length=1, fill=True, fill_method='mean',
                          test_size=0.2):
    if fill:
        imputer = SimpleImputer(strategy=fill_method)
        df[feature_cols] = imputer.fit_transform(df[feature_cols])

    X = df[feature_cols].values
    y = df[outcome_col].values

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Reshape for LSTM: (num_samples, sequence_length, num_features)
    if sequence_length > 1:
        X_tensor = X_tensor.reshape(-1, sequence_length, X_tensor.shape[1])
    else:
        X_tensor = X_tensor.unsqueeze(1)

    # Split data
    num_samples = len(X_tensor)
    split = int(num_samples * (1 - test_size))

    X_train, X_test = X_tensor[:split], X_tensor[split:]
    y_train, y_test = y_tensor[:split], y_tensor[split:]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64)


import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out
import torch.optim as optim
from sklearn.metrics import accuracy_score

def train_and_evaluate_lstm(train_loader, test_loader, input_size, hidden_size=64, output_size=1, num_layers=1, num_epochs=20, learning_rate=0.001):
    model = LSTMModel(input_size, hidden_size, output_size, num_layers)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

    # Evaluation
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            y_pred.extend(torch.sigmoid(outputs).squeeze().cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    y_pred = (np.array(y_pred) > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy}')

    return model


# example usage
from lstm_module import prepare_data_for_lstm, LSTMModel, train_and_evaluate_lstm

# Load your data into a DataFrame
df = pd.read_csv('your_data.csv')

# Prepare data
value_cols = ['value1', 'value2']  # Replace with your actual columns
feature_cols = ['feature1', 'feature2']  # Replace with your actual features
outcome_col = 'outcome'
train_loader, test_loader = prepare_data_for_lstm(df, value_cols, feature_cols, outcome_col, sequence_length=1)

# Train and evaluate the model
input_size = len(feature_cols)
model = train_and_evaluate_lstm(train_loader, test_loader, input_size)
