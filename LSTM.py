"""
Kenya Agricultural LSTM Forecast Dashboard
1960–2020 Data → 3-Year Forecast (2021–2023)
Omari Galana Shevo – MUST
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Kenya Agricultural LSTM Forecast",
                   page_icon="🌾",
                   layout="wide")

st.title("🌾 Kenya Agricultural Production Forecast (LSTM)")
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

# ──────────────────────────────────────────────
# PREPARE SERIES
# ──────────────────────────────────────────────
series_df = df[df["Item"] == crop_selected].sort_values("Year")
years = series_df["Year"].values
values = series_df["Value"].values.reshape(-1, 1)

# ──────────────────────────────────────────────
# MANUAL SCALING & METRICS
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
# LSTM FUNCTION
# ──────────────────────────────────────────────
@st.cache_resource
def train_lstm(series, look_back, epochs):

    scaled, min_val, max_val = min_max_scale(series)

    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i+look_back, 0])
        y.append(scaled[i+look_back, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(look_back,1)),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(patience=15, restore_best_weights=True)

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=16,
              verbose=0,
              callbacks=[early_stop])

    # Predictions
    test_pred = model.predict(X_test).flatten()
    y_test_actual = y_test

    test_pred_actual = inverse_scale(test_pred, min_val, max_val)
    y_test_actual_scaled = inverse_scale(y_test_actual, min_val, max_val)

    # Metrics
    return_rmse = rmse(y_test_actual_scaled, test_pred_actual)
    return_mae = mae(y_test_actual_scaled, test_pred_actual)
    return_mape = mape(y_test_actual_scaled, test_pred_actual)

    # 3-year future forecast
    last_seq = scaled[-look_back:]
    future = []

    for _ in range(3):
        input_seq = last_seq.reshape(1, look_back, 1)
        next_pred = model.predict(input_seq)[0][0]
        future.append(next_pred)
        last_seq = np.append(last_seq[1:], next_pred)

    future_vals = inverse_scale(np.array(future), min_val, max_val)

    return return_rmse, return_mae, return_mape, future_vals.flatten(), model

# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
if len(values) > look_back + 5:

    rmse_val, mae_val, mape_val, future_vals, model = train_lstm(values, look_back, epochs)

    st.subheader("📊 Model Validation Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse_val:,.0f}")
    col2.metric("MAE", f"{mae_val:,.0f}")
    col3.metric("MAPE (%)", f"{mape_val:.2f}")

    # ──────────────────────────────────────────────
    # FORECAST DATAFRAME
    # ──────────────────────────────────────────────
    future_years = [2021, 2022, 2023]
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Value": future_vals,
        "Type": "Forecast"
    })

    history_df = series_df[["Year","Value"]].copy()
    history_df["Type"] = "Actual"
    combined = pd.concat([history_df, forecast_df])

    # ──────────────────────────────────────────────
    # VISUALIZATION
    # ──────────────────────────────────────────────
    st.subheader("📈 Actual vs LSTM Forecast")

    chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X("Year:Q", axis=alt.Axis(format="d")),
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        color=alt.Color("Type:N",
                        scale=alt.Scale(domain=["Actual","Forecast"],
                                        range=["#2d8a45","#e74c3c"])),
        strokeDash=alt.condition(
            alt.datum.Type == "Forecast",
            alt.value([6,4]),
            alt.value([0])
        ),
        tooltip=["Year","Value","Type"]
    ).properties(height=450).interactive()

    st.altair_chart(chart, use_container_width=True)
    st.info("""
    Forecast horizon: 3 years (2021–2023).
    Model trained on 1960–2020 historical production data using a two-layer LSTM network.
    """)
else:
    st.warning("Not enough data points for selected look-back window.")
