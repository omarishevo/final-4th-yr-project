"""
Kenya Agricultural Forecast Dashboard (Pure NumPy Version)
1960–2020 Data → 3-Year Forecast (2021–2023)
Omari Galana Shevo – MUST
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Kenya Agricultural Forecast",
                   page_icon="🌾",
                   layout="wide")

st.title("🌾 Kenya Agricultural Production Forecast (NumPy)")
st.markdown("Lightweight Forecast using historical FAOSTAT data")

# ──────────────────────────────────────────────
# DATA UPLOAD
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader("Upload Excel dataset (FAOSTAT)", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df = df[df["Element"] == "Production"]
    df = df[df["Year"].between(1960, 2020)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    # ──────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────
    crop_list = sorted(df["Item"].unique())
    crop_selected = st.sidebar.selectbox("Select Crop", crop_list)
    look_back = st.sidebar.slider("Look-back Window (years)", 3, 10, 5)

    # ──────────────────────────────────────────────
    # PREPARE SERIES
    # ──────────────────────────────────────────────
    series_df = df[df["Item"] == crop_selected].sort_values("Year")
    values = series_df["Value"].values
    years = series_df["Year"].values

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
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # ──────────────────────────────────────────────
    # SEQUENCE DATA
    # ──────────────────────────────────────────────
    def create_sequences(series, look_back):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)

    # ──────────────────────────────────────────────
    # ROLLING LINEAR FORECAST
    # ──────────────────────────────────────────────
    def train_forecast(series, look_back, forecast_horizon=3):
        scaled, min_val, max_val = min_max_scale(series)
        X, y = create_sequences(scaled, look_back)
        
        # Train simple linear regression on each sequence
        coeffs = []
        for i in range(len(X)):
            # Add bias term
            Xi = np.vstack([X[i], np.ones(look_back)]).T
            yi = y[i]
            w = np.linalg.lstsq(Xi, np.full(look_back, yi), rcond=None)[0]
            coeffs.append(w)
        
        # Use last look_back sequence to forecast
        last_seq = scaled[-look_back:]
        future_scaled = []
        for _ in range(forecast_horizon):
            Xi = np.vstack([last_seq, np.ones(look_back)]).T
            avg_w = np.mean(coeffs, axis=0)
            next_pred = Xi @ avg_w
            next_val = next_pred.mean()
            future_scaled.append(next_val)
            last_seq = np.append(last_seq[1:], next_val)
        
        future_vals = inverse_scale(np.array(future_scaled), min_val, max_val)
        
        # Metrics
        y_true_scaled = y[-look_back:]
        y_pred_scaled = np.mean(np.array(coeffs)[:,0]) * last_seq + np.mean(np.array(coeffs)[:,1])
        y_true = inverse_scale(y_true_scaled, min_val, max_val)
        y_pred = inverse_scale(np.array(y_pred_scaled), min_val, max_val)
        
        return rmse(y_true, y_pred), mae(y_true, y_pred), mape(y_true, y_pred), future_vals

    # ──────────────────────────────────────────────
    # TRAIN MODEL & FORECAST
    # ──────────────────────────────────────────────
    if len(values) > look_back + 3:
        rmse_val, mae_val, mape_val, future_vals = train_forecast(values, look_back)
        
        st.subheader("📊 Forecast Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{rmse_val:,.0f}")
        col2.metric("MAE", f"{mae_val:,.0f}")
        col3.metric("MAPE (%)", f"{mape_val:.2f}")
        
        future_years = [2021, 2022, 2023]
        forecast_df = pd.DataFrame({"Year": future_years,"Value": future_vals,"Type":"Forecast"})
        history_df = series_df[["Year","Value"]].copy()
        history_df["Type"] = "Actual"
        combined = pd.concat([history_df, forecast_df])
        
        st.subheader("📈 Actual vs Forecast")
        chart = alt.Chart(combined).mark_line(point=True).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["Actual","Forecast"], range=["#2d8a45","#e74c3c"])),
            strokeDash=alt.condition(alt.datum.Type=="Forecast", alt.value([6,4]), alt.value([0])),
            tooltip=["Year","Value","Type"]
        ).properties(height=450).interactive()
        st.altair_chart(chart, use_container_width=True)
        st.info("Forecast horizon: 3 years (2021–2023). Prediction done using lightweight rolling-window linear regression (NumPy only).")
    else:
        st.warning("Not enough data points for selected look-back window.")
else:
    st.info("Upload your FAOSTAT Excel dataset to begin forecasting.")
