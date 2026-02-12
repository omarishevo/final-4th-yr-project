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

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
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
# LSTM FUNCTION
# ──────────────────────────────────────────────
@st.cache_resource
def train_lstm(series, look_back, epochs):

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)

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
    test_pred = model.predict(X_test)

    test_pred = scaler.inverse_transform(test_pred)
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    rmse = math.sqrt(mean_squared_error(y_test_actual, test_pred))
    mae = mean_absolute_error(y_test_actual, test_pred)
    mape = np.mean(np.abs((y_test_actual - test_pred) / y_test_actual)) * 100

    # 3-year future forecast
    last_seq = scaled[-look_back:]
    future = []

    for _ in range(3):
        input_seq = last_seq.reshape(1, look_back, 1)
        next_pred = model.predict(input_seq)
        future.append(next_pred[0][0])
        last_seq = np.append(last_seq[1:], next_pred, axis=0)

    future = scaler.inverse_transform(np.array(future).reshape(-1,1))

    return rmse, mae, mape, future.flatten(), model

# ──────────────────────────────────────────────
# TRAIN MODEL
# ──────────────────────────────────────────────
if len(values) > look_back + 5:

    rmse, mae, mape, future_vals, model = train_lstm(values, look_back, epochs)

    st.subheader("📊 Model Validation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse:,.0f}")
    col2.metric("MAE", f"{mae:,.0f}")
    col3.metric("MAPE (%)", f"{mape:.2f}")

    # ──────────────────────────────────────────────
    # BUILD FORECAST DATAFRAME
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
