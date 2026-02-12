"""
Kenya Agricultural Forecast Dashboard
1960–2020 Historical Backtest + Forecast (2021–2025)
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
Historical Backtesting (1960–2020) + Future Forecast (2021–2025)  
Model: Rolling Linear Regression implemented using NumPy
""")

# ──────────────────────────────────────────────
# DATA UPLOAD
# ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload FAOSTAT CSV (must contain 'Year', 'Item', 'Element', 'Value')",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # Filter production
    df = df[df["Element"] == "Production"]
    df = df[df["Year"].between(1960, 2020)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    # ──────────────────────────────────────────────
    # SIDEBAR
    # ──────────────────────────────────────────────
    st.sidebar.header("Forecast Settings")

    crop_list = sorted(df["Item"].unique())
    crop_selected = st.sidebar.selectbox("Select Crop", crop_list)

    look_back = st.sidebar.slider("Look-back Window (years)", 3, 10, 5)

    forecast_horizon = st.sidebar.slider(
        "Forecast Years (2021–2025)",
        min_value=1,
        max_value=5,
        value=5
    )

    # ──────────────────────────────────────────────
    # PREPARE DATA
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
        # Avoid division by zero
        y_true = np.where(y_true == 0, 1e-8, y_true)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def create_sequences(series, look_back):
        X, y = [], []
        for i in range(len(series) - look_back):
            X.append(series[i:i + look_back])
            y.append(series[i + look_back])
        return np.array(X), np.array(y)

    # ──────────────────────────────────────────────
    # MODEL BACKTEST (1960–2020)
    # ──────────────────────────────────────────────
    if len(values) > look_back:

        X, y = create_sequences(values, look_back)

        predictions = []
        actuals = []

        for i in range(len(X)):
            X_train = X[:i + 1]
            y_train = y[:i + 1]

            X_train_bias = np.c_[X_train, np.ones(len(X_train))]

            w = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]

            X_test = np.append(X[i], 1)
            y_pred = X_test @ w

            predictions.append(y_pred)
            actuals.append(y[i])

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse_val = rmse(actuals, predictions)
        mae_val = mae(actuals, predictions)
        mape_val = mape(actuals, predictions)

        # ──────────────────────────────────────────
        # DISPLAY BACKTEST METRICS
        # ──────────────────────────────────────────
        st.subheader("📊 Model Performance (1960–2020 Backtest)")
        c1, c2, c3 = st.columns(3)
        c1.metric("RMSE", f"{rmse_val:,.0f}")
        c2.metric("MAE", f"{mae_val:,.0f}")
        c3.metric("MAPE (%)", f"{mape_val:.2f}")

        # Backtest chart
        backtest_years = years[look_back:]
        backtest_df = pd.DataFrame({
            "Year": backtest_years,
            "Actual": actuals,
            "Predicted": predictions
        })

        backtest_chart = alt.Chart(backtest_df).transform_fold(
            ["Actual", "Predicted"],
            as_=["Type", "Value"]
        ).mark_line().encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("Value:Q", axis=alt.Axis(format="~s")),
            color="Type:N",
            tooltip=["Year", "Type", "Value"]
        ).interactive()

        st.altair_chart(backtest_chart, use_container_width=True)

        # ──────────────────────────────────────────
        # FUTURE FORECAST (2021–2025)
        # ──────────────────────────────────────────
        X_full_bias = np.c_[X, np.ones(len(X))]
        w_final = np.linalg.lstsq(X_full_bias, y, rcond=None)[0]

        last_sequence = values[-look_back:].copy()
        future_predictions = []

        progress = st.progress(0)

        for i in range(forecast_horizon):
            seq_bias = np.append(last_sequence, 1)
            next_pred = seq_bias @ w_final
            future_predictions.append(next_pred)

            last_sequence = np.append(last_sequence[1:], next_pred)
            progress.progress((i + 1) / forecast_horizon)
            time.sleep(0.05)

        progress.empty()

        future_years = list(range(2021, 2021 + forecast_horizon))

        forecast_df = pd.DataFrame({
            "Year": future_years,
            "Value": future_predictions,
            "Type": "Forecast"
        })

        history_df = series_df[["Year", "Value"]].copy()
        history_df["Type"] = "Actual"

        combined = pd.concat([history_df, forecast_df])

        st.subheader("📈 Forecast (2021–2025)")
        st.dataframe(forecast_df)

        forecast_chart = alt.Chart(combined).mark_line(point=True).encode(
            x=alt.X("Year:Q", axis=alt.Axis(format="d")),
            y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
            color="Type:N",
            tooltip=["Year", "Value", "Type"]
        ).interactive()

        st.altair_chart(forecast_chart, use_container_width=True)

        csv = combined.to_csv(index=False)
        st.download_button(
            "📥 Download Forecast CSV",
            csv,
            "kenya_agriculture_forecast.csv",
            "text/csv"
        )

    else:
        st.warning("Not enough data for selected look-back window.")

else:
    st.info("Upload FAOSTAT CSV file to begin.")
