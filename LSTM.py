"""
Kenya Agricultural Forecast Dashboard
1960–2020 Data → Forecast Horizon
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
st.set_page_config(
    page_title="Kenya Agricultural Forecast",
    page_icon="🌾",
    layout="wide"
)

st.title("🌾 Kenya Agricultural Production Forecast (NumPy)")
st.markdown("""
This dashboard forecasts agricultural production in Kenya using historical FAOSTAT data (1960–2020).  
The forecast uses a rolling-window linear regression approach implemented purely with NumPy.
""")

# ──────────────────────────────────────────────
# DATA UPLOAD
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload FAOSTAT CSV (must contain Year, Item, Element, Value)",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Basic validation
    required_cols = {"Year", "Item", "Element", "Value"}
    if not required_cols.issubset(df.columns):
        st.error("CSV missing required columns.")
        st.stop()

    # Filter production
    df = df[df["Element"] == "Production"]
    df = df[df["Year"].between(1960, 2020)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    crop_list = sorted(df["Item"].unique())
    crop_selected = st.sidebar.selectbox("Select Crop", crop_list)

    series_df = df[df["Item"] == crop_selected].sort_values("Year")

    values = series_df["Value"].values
    years = series_df["Year"].values

    if len(values) < 3:
        st.warning("Not enough data points for forecasting.")
        st.stop()

    # ──────────────────────────────────────────────
    # AUTO ADJUST SLIDERS
    # ──────────────────────────────────────────────
    max_look_back = len(values) - 1
    look_back = st.sidebar.slider(
        "Look-back Window (years)",
        1,
        max_look_back,
        min(5, max_look_back)
    )

    max_forecast_horizon = len(values) - look_back
    forecast_horizon = st.sidebar.slider(
        "Forecast Horizon (years)",
        1,
        max_forecast_horizon,
        min(3, max_forecast_horizon)
    )

    # ──────────────────────────────────────────────
    # HISTORICAL SUMMARY
    # ──────────────────────────────────────────────
    st.subheader("📋 Historical Summary")
    st.write(series_df.describe())
    st.write(f"Data Range: {years.min()} – {years.max()}")
    st.write(f"Total Observations: {len(values)}")

    hist_chart = alt.Chart(series_df).mark_line(point=True).encode(
        x="Year:Q",
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        tooltip=["Year", "Value"]
    ).properties(height=300)

    st.altair_chart(hist_chart, use_container_width=True)

    # ──────────────────────────────────────────────
    # HELPER FUNCTIONS
    # ──────────────────────────────────────────────
    def min_max_scale(array):
        min_val = array.min()
        max_val = array.max()
        scaled = (array - min_val) / (max_val - min_val)
        return scaled, min_val, max_val

    def inverse_scale(scaled, min_val, max_val):
        return scaled * (max_val - min_val) + min_val

    def create_sequences(series, look_back):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back])
        return np.array(X), np.array(y)

    # ──────────────────────────────────────────────
    # CORRECTED FORECAST FUNCTION
    # ──────────────────────────────────────────────
    def train_forecast(series, look_back, forecast_horizon):

        scaled, min_val, max_val = min_max_scale(series)
        X, y = create_sequences(scaled, look_back)

        if len(X) == 0:
            return np.nan, np.nan, np.nan, np.array([])

        coeffs = []
        y_preds_train = []

        # Train and compute predictions
        for i in range(len(X)):
            Xi = np.vstack([X[i], np.ones(look_back)]).T
            yi = y[i]

            w = np.linalg.lstsq(Xi, np.full(look_back, yi), rcond=None)[0]
            coeffs.append(w)

            pred = (Xi @ w).mean()
            y_preds_train.append(pred)

        y_preds_train = np.array(y_preds_train)

        # Compute metrics correctly
        y_true = inverse_scale(y, min_val, max_val)
        y_pred = inverse_scale(y_preds_train, min_val, max_val)

        rmse_val = np.sqrt(np.mean((y_true - y_pred) ** 2))
        mae_val = np.mean(np.abs(y_true - y_pred))
        mape_val = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # ───────────── Future Forecast ─────────────
        progress_text = "Forecasting..."
        my_bar = st.progress(0, text=progress_text)

        last_seq = scaled[-look_back:]
        future_scaled = []

        avg_w = np.mean(coeffs, axis=0)

        for i in range(forecast_horizon):
            time.sleep(0.05)

            Xi = np.vstack([last_seq, np.ones(look_back)]).T
            next_pred = (Xi @ avg_w).mean()

            future_scaled.append(next_pred)
            last_seq = np.append(last_seq[1:], next_pred)

            my_bar.progress((i + 1) / forecast_horizon, text=progress_text)

        my_bar.empty()

        future_vals = inverse_scale(np.array(future_scaled), min_val, max_val)

        return rmse_val, mae_val, mape_val, future_vals

    # ──────────────────────────────────────────────
    # RUN FORECAST
    # ──────────────────────────────────────────────
    rmse_val, mae_val, mape_val, future_vals = train_forecast(
        values,
        look_back,
        forecast_horizon
    )

    # Metrics Display
    st.subheader("📊 Forecast Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{rmse_val:,.0f}")
    col2.metric("MAE", f"{mae_val:,.0f}")
    col3.metric("MAPE (%)", f"{mape_val:.2f}")

    # ──────────────────────────────────────────────
    # Forecast Data
    # ──────────────────────────────────────────────
    future_years = list(range(years.max() + 1,
                              years.max() + 1 + forecast_horizon))

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Value": future_vals,
        "Type": "Forecast"
    })

    history_df = series_df[["Year", "Value"]].copy()
    history_df["Type"] = "Actual"

    combined = pd.concat([history_df, forecast_df])

    # Growth table
    st.subheader("📈 Forecast Growth")

    growth_percent = 100 * (future_vals / values[-1] - 1)

    growth_df = pd.DataFrame({
        "Year": future_years,
        "Forecast": future_vals,
        "Change (%)": growth_percent
    })

    st.dataframe(growth_df)

    # ──────────────────────────────────────────────
    # CHART WITH UNCERTAINTY BAND
    # ──────────────────────────────────────────────
    st.subheader("📈 Actual vs Forecast")

    forecast_df["Lower"] = forecast_df["Value"] * 0.9
    forecast_df["Upper"] = forecast_df["Value"] * 1.1

    band = alt.Chart(forecast_df).mark_area(opacity=0.2).encode(
        x="Year:Q",
        y="Lower:Q",
        y2="Upper:Q"
    )

    line = alt.Chart(combined).mark_line(point=True).encode(
        x="Year:Q",
        y="Value:Q",
        color="Type:N",
        tooltip=["Year", "Value", "Type"]
    )

    st.altair_chart(line + band, use_container_width=True)

    # Download
    csv = combined.to_csv(index=False)
    st.download_button(
        "📥 Download Forecast Data",
        csv,
        "forecast.csv",
        "text/csv"
    )

else:
    st.info("Upload your FAOSTAT CSV dataset to begin.")
