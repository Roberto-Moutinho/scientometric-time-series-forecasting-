import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm

# =========================
# Reproducibility
# =========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# =========================
# Config
# =========================
DATA_PATH = "data/scientometric_series.csv"
TRAIN_END_YEAR = 2018
VAL_END_YEAR = 2021
WINDOW_SIZE = 5
EPOCHS = 300
LR = 0.01
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# Metrics
# =========================
def compute_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return rmse, mae, mape

# =========================
# LSTM Model
# =========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

# =========================
# Sequence Builder
# =========================
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(data[i+window])
    return np.array(X), np.array(y)

# =========================
# Load Data
# =========================
df = pd.read_csv(DATA_PATH)

areas = df["area"].unique()
results = []

for area in areas:
    print(f"\nProcessing area: {area}")
    sub = df[df["area"] == area].sort_values("year")

    train = sub[sub["year"] <= TRAIN_END_YEAR]
    val = sub[(sub["year"] > TRAIN_END_YEAR) & (sub["year"] <= VAL_END_YEAR)]
    test = sub[sub["year"] > VAL_END_YEAR]

    y_train = train["publications"].values
    y_val = val["publications"].values
    y_test = test["publications"].values

    # =========================
    # Polynomial Regression
    # =========================
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(train["year"].values.reshape(-1,1))
    X_test_poly = poly.transform(test["year"].values.reshape(-1,1))

    reg = LinearRegression()
    reg.fit(X_train_poly, y_train)
    y_pred_poly = reg.predict(X_test_poly)

    poly_metrics = compute_metrics(y_test, y_pred_poly)

    # =========================
    # ARIMA
    # =========================
    arima_model = ARIMA(y_train, order=(2,1,2))
    arima_fit = arima_model.fit()
    arima_forecast = arima_fit.forecast(steps=len(y_test))
    arima_metrics = compute_metrics(y_test, arima_forecast)

    # =========================
    # LSTM
    # =========================
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(y_train.reshape(-1,1))

    X_seq, y_seq = create_sequences(scaled_train, WINDOW_SIZE)

    X_tensor = torch.tensor(X_seq, dtype=torch.float32).to(DEVICE)
    y_tensor = torch.tensor(y_seq, dtype=torch.float32).to(DEVICE)

    model = LSTMModel().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Forecast recursively
    model.eval()
    input_seq = scaled_train[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)
    input_seq = torch.tensor(input_seq, dtype=torch.float32).to(DEVICE)

    preds = []
    for _ in range(len(y_test)):
        with torch.no_grad():
            pred = model(input_seq)
        preds.append(pred.item())
        new_input = torch.cat(
            (input_seq[:,1:,:], pred.unsqueeze(0)), dim=1
        )
        input_seq = new_input

    lstm_forecast = scaler.inverse_transform(
        np.array(preds).reshape(-1,1)
    ).flatten()

    lstm_metrics = compute_metrics(y_test, lstm_forecast)

    results.append({
        "area": area,
        "model": "Polynomial",
        "RMSE": poly_metrics[0],
        "MAE": poly_metrics[1],
        "MAPE": poly_metrics[2]
    })

    results.append({
        "area": area,
        "model": "ARIMA",
        "RMSE": arima_metrics[0],
        "MAE": arima_metrics[1],
        "MAPE": arima_metrics[2]
    })

    results.append({
        "area": area,
        "model": "LSTM",
        "RMSE": lstm_metrics[0],
        "MAE": lstm_metrics[1],
        "MAPE": lstm_metrics[2]
    })


