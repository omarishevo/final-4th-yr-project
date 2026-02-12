"""
Kenya Agricultural LSTM Forecast Dashboard (PyTorch Version)
1960–2020 Data → 3-Year Forecast (2021–2023)
Omari Galana Shevo – MUST
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math
import torch
import torch.nn as nn

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Kenya Agricultural LSTM Forecast (PyTorch)",
                   page_icon="🌾",
                   layout="wide")

st.title("🌾 Kenya Agricultural Production Forecast (PyTorch LSTM)")
st.markdown("Deep Learning Forecast using 1960–2020 FAOSTAT Data")

# ──────────────────────────────────────────────
# LOAD DATA
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("Kenyas_Agricultural_Production.xlsx")
    df = df[df["Element"] == "Production"]
    df = df[df["Year"].between(1960, 2020)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])
    return df

try:
    df = load_data()
except:
    st.error("Excel file not found. Place it in same directory.")
    st.stop()

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
crop_list = sorted(df["Item"].unique())
crop_selected = st.sidebar.selectbox("Select Crop", crop_list)
look_back = st.sidebar.slider("Look-back Window (years)", 3, 10, 5)
epochs = st.sidebar.slider("Training Epochs", 20, 150, 80)
lr = st.sidebar.slider("Learning Rate", 0.001, 0.01, 0.005)

# ──────────────────────────────────────────────
# PREPARE SERIES
# ──────────────────────────────────────────────
series_df = df[df["Item"] == crop_selected].sort_values("Year")
values = series_df["Value"].values.reshape(-1, 1)

# ──────────────────────────────────────────────
# SCALING & METRICS
# ──────────────────────────────────────────────
def min_max_scale(array):
    min_val = array.min()
    max_val = array.max()
    scaled = (array - min_val) / (max_val - min_val)
    return scaled, min_val, max_val

def inverse_scale(scaled, min_val, max_val):
    return scaled * (max_val - min_val) + min_val

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# ──────────────────────────────────────────────
# CREATE SEQUENCES
# ──────────────────────────────────────────────
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

# ──────────────────────────────────────────────
# PYTORCH LSTM MODEL
# ──────────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last timestep
        out = self.dropout(out)
        out = self.fc(out)
        return out

# ──────────────────────────────────────────────
# TRAIN & FORECAST FUNCTION
# ──────────────────────────────────────────────
@st.cache_resource
def train_lstm_pytorch(series, look_back, epochs, lr):
    scaled, min_val, max_val = min_max_scale(series)
    X, y = create_sequences(scaled, look_back)
    
    X_train, X_test = X[:int(0.8*len(X))], X[int(0.8*len(X)):]
    y_train, y_test = y[:int(0.8*len(y))], y[int(0.8*len(y)):]
    
    # Convert to PyTorch tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)
    
    model = LSTMModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, y_train_t)
        loss.backward()
        optimizer.step()
    
    # Predictions
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test_t).numpy().flatten()
    
    y_test_actual = inverse_scale(y_test, min_val, max_val)
    test_pred_actual = inverse_scale(test_pred, min_val, max_val)
    
    # Metrics
    return_rmse = rmse(y_test_actual, test_pred_actual)
    return_mae = mae(y_test_actual, test_pred_actual)
    return_mape = mape(y_test_actual, test_pred_actual)
    
    # 3-year forecast
    last_seq = scaled[-look_back:]
    future = []
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            seq_t = torch.tensor(last_seq.reshape(1, look_back, 1), dtype=torch.float32)
            next_pred = model(seq_t).numpy()[0][0]
            future.append(next_pred)
            last_seq = np.append(last_seq[1:], next_pred)
    
    future_vals = inverse_scale(np.array(future), min_val, max_val)
    return return_rmse, return_mae, return_mape, future_vals.flatten(), model

# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
if len(values) > look_back + 5:
    rmse_val, mae_val, mape_val, future_vals, model = train_lstm_pytorch(values, look_back, epochs, lr)
    
    st.subheader("📊 Model Validation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse_val:,.0f}")
    col2.metric("MAE", f"{mae_val:,.0f}")
    col3.metric("MAPE (%)", f"{mape_val:.2f}")
    
    future_years = [2021, 2022, 2023]
    forecast_df = pd.DataFrame({"Year": future_years,"Value": future_vals,"Type":"Forecast"})
    history_df = series_df[["Year","Value"]].copy()
    history_df["Type"] = "Actual"
    combined = pd.concat([history_df, forecast_df])
    
    st.subheader("📈 Actual vs LSTM Forecast")
    chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X("Year:Q", axis=alt.Axis(format="d")),
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        color=alt.Color("Type:N", scale=alt.Scale(domain=["Actual","Forecast"], range=["#2d8a45","#e74c3c"])),
        strokeDash=alt.condition(alt.datum.Type == "Forecast", alt.value([6,4]), alt.value([0])),
        tooltip=["Year","Value","Type"]
    ).properties(height=450).interactive()
    st.altair_chart(chart, use_container_width=True)
    st.info("Forecast horizon: 3 years (2021–2023). Model trained on 1960–2020 historical production data using a two-layer PyTorch LSTM network.")
else:
    st.warning("Not enough data points for selected look-back window.")
