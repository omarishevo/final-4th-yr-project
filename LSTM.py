"""
Kenya Agricultural Forecast Dashboard
1960–2020 Data → Predict 2021–2025
Omari Galana Shevo – MUST
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(page_title="Kenya Agricultural Forecast",
                   page_icon="🌾",
                   layout="wide")

st.title("🌾 Kenya Agricultural Production Forecast (NumPy)")
st.markdown("""
Forecast agricultural production in Kenya using FAOSTAT data (1960–2020).
Model: Rolling-window Linear Regression implemented in NumPy.
Forecast Horizon: 2021–2025 (5 Years)
""")

# ──────────────────────────────────────────────
# DATA UPLOAD
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload FAOSTAT CSV (must include 'Year', 'Item', 'Element', 'Value')",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Filter production data
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

    forecast_horizon = 5  # FIXED → 2021–2025

    # ──────────────────────────────────────────────
    # PREPARE SERIES
    # ──────────────────────────────────────────────
    series_df = df[df["Item"] == crop_selected].sort_values("Year")
    values = series_df["Value"].values
    years = series_df["Year"].values

    st.subheader("📋 Historical Data Summary")
    st.write(series_df.describe())

    hist_chart = alt.Chart(series_df).mark_line(point=True).encode(
        x="Year:Q",
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        tooltip=["Year", "Value"]
    ).properties(height=300)

    st.altair_chart(hist_chart, use_container_width=True)

    # ──────────────────────────────────────────────
    # HELPER FUNCTIONS
    # ──────────────────────────────────────────────
    def rmse(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred) ** 2))

    def mae(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))

    def mape(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def create_sequences(series, look_back):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)

    # ──────────────────────────────────────────────
    # MODEL TRAINING + FORECAST
    # ──────────────────────────────────────────────
    if len(values) > look_back + 5:

        # Create supervised dataset
        X, y = create_sequences(values, look_back)

        # Train/Test split (last 5 actual years for metrics)
        X_train, X_test = X[:-5], X[-5:]
        y_train, y_test = y[:-5], y[-5:]

        # Add bias term
        X_train_bias = np.c_[X_train, np.ones(len(X_train))]
        X_test_bias = np.c_[X_test, np.ones(len(X_test))]

        # Linear regression solution
        w = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]

        # Predictions for test set
        y_pred_test = X_test_bias @ w

        # Metrics (NO dimension mismatch now)
        rmse_val = rmse(y_test, y_pred_test)
        mae_val = mae(y_test, y_pred_test)
        mape_val = mape(y_test, y_pred_test)

        # ──────────────────────────────────────────
        # FUTURE FORECAST 2021–2025
        # ──────────────────────────────────────────
        last_sequence = values[-look_back:].copy()
        future_predictions = []

        progress = st.progress(0)

        for i in range(forecast_horizon):
            seq_with_bias = np.append(last_sequence, 1)
            next_pred = seq_with_bias @ w
            future_predictions.append(next_pred)

            last_sequence = np.append(last_sequence[1:], next_pred)
            progress.progress((i + 1) / forecast_horizon)
            time.sleep(0.05)

        progress.empty()

        future_years = list(range(2021, 2026))

        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Value": future_predictions,
            "Type": "Forecast"
        })

        history_df = series_df[["Year", "Value"]].copy()
        history_df["Type"] = "Actual"

        combined = pd.concat([history_df, forecast_df])

        # ──────────────────────────────────────────
        # DISPLAY METRICS
        # ──────────────────────────────────────────
        st.subheader("📊 Model Performance (Last 5 Known Years)")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse_val:,.0f}")
        c2.metric("MAE", f"{mae_val:,.0f}")
        c3.metric("MAPE (%)", f"{mape_val:.2f}")

        # ──────────────────────────────────────────
        # FORECAST TABLE
        # ──────────────────────────────────────────
        st.subheader("📈 Forecast (2021–2025)")
        st.dataframe(forecast_df)

        # ──────────────────────────────────────────
        # VISUALIZATION
        # ──────────────────────────────────────────
        chart = alt.Chart(combined).mark_line(point=True).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
            color="Type:N",
            tooltip=["Year", "Value", "Type"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

        # ──────────────────────────────────────────
        # DOWNLOAD
        # ──────────────────────────────────────────
        csv = combined.to_csv(index=False)
        st.download_button("📥 Download Forecast CSV",
                           csv,
                           "kenya_agriculture_forecast_2021_2025.csv",
                           "text/csv")

    else:
        st.warning("Not enough data for selected look-back window.")

else:
    st.info("Upload FAOSTAT CSV file to begin.")
