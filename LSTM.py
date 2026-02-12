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
st.set_page_config(
    page_title="Kenya Agricultural Forecast",
    page_icon="🌾",
    layout="wide"
)

# ──────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F5F7F2; }
    .stMetric { background-color: #E8F5E9; padding: 10px; border-radius: 10px; }
    .stSidebar .sidebar-content { background-color: #F0F4F1; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
st.title("🌾 Kenya Agricultural Production Forecast")
st.markdown("""
Forecast agricultural production in Kenya using FAOSTAT data (1960–2020).  
**Model:** Linear Regression implemented in NumPy.
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
    df = df[df["Element"] == "Production"]
    df = df[df["Year"].between(1960, 2020)]
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna(subset=["Value"])

    # ──────────────────────────────────────────────
    # SIDEBAR SETTINGS
    # ──────────────────────────────────────────────
    st.sidebar.header("Forecast Settings")
    crop_list = sorted(df["Item"].unique())
    crop_selected = st.sidebar.selectbox("Select Crop", crop_list)
    look_back = st.sidebar.slider("Look-back Window (years)", 3, 10, 5)
    forecast_horizon = st.sidebar.slider("Forecast Years (2021–2025)", 1, 5, 5)
    st.sidebar.markdown("""
- **Look-back Window:** Number of past years used to predict next year  
- **Forecast Years:** How many years ahead to predict (max 5)
""")

    # ──────────────────────────────────────────────
    # PREPARE SERIES
    # ──────────────────────────────────────────────
    series_df = df[df["Item"] == crop_selected].sort_values("Year")
    values = series_df["Value"].values

    # ──────────────────────────────────────────────
    # HELPER FUNCTIONS
    # ──────────────────────────────────────────────
    def rmse(y_true, y_pred): return np.sqrt(np.mean((y_true - y_pred) ** 2))
    def mae(y_true, y_pred): return np.mean(np.abs(y_true - y_pred))
    def mape(y_true, y_pred): return np.mean(np.abs((y_true - y_pred)/y_true)) * 100
    def create_sequences(series, look_back):
        X, y = [], []
        for i in range(len(series)-look_back):
            X.append(series[i:i+look_back])
            y.append(series[i+look_back])
        return np.array(X), np.array(y)

    # ──────────────────────────────────────────────
    # MODEL + FORECAST
    # ──────────────────────────────────────────────
    if len(values) > look_back + 5:

        X, y = create_sequences(values, look_back)
        X_train, X_test = X[:-5], X[-5:]
        y_train, y_test = y[:-5], y[-5:]
        X_train_bias, X_test_bias = np.c_[X_train, np.ones(len(X_train))], np.c_[X_test, np.ones(len(X_test))]
        w = np.linalg.lstsq(X_train_bias, y_train, rcond=None)[0]
        y_pred_test = X_test_bias @ w

        # Metrics
        rmse_val, mae_val, mape_val = rmse(y_test, y_pred_test), mae(y_test, y_pred_test), mape(y_test, y_pred_test)

        # Forecast
        last_sequence = values[-look_back:].copy()
        future_predictions = []
        progress = st.progress(0)
        for i in range(forecast_horizon):
            seq_with_bias = np.append(last_sequence, 1)
            next_pred = seq_with_bias @ w
            future_predictions.append(next_pred)
            last_sequence = np.append(last_sequence[1:], next_pred)
            progress.progress((i+1)/forecast_horizon)
            time.sleep(0.05)
        progress.empty()

        future_years = list(range(2021, 2021 + forecast_horizon))
        forecast_df = pd.DataFrame({"Year": future_years, "Value": future_predictions, "Type":"Forecast"})
        history_df = series_df[["Year","Value"]].copy()
        history_df["Type"]="Actual"
        combined = pd.concat([history_df, forecast_df])

        # ──────────────────────────────────────────────
        # COLLAPSIBLE SECTIONS
        # ──────────────────────────────────────────────
        with st.expander("📋 Historical Data Summary", expanded=True):
            st.dataframe(series_df.describe().transpose())
            hist_chart = alt.Chart(series_df).mark_line(point=True, color="#2E7D32").encode(
                x="Year:Q", y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
                tooltip=["Year","Value"]
            ).properties(height=300)
            st.altair_chart(hist_chart, use_container_width=True)

        with st.expander("📊 Model Performance"):
            c1, c2, c3 = st.columns(3)
            c1.metric("RMSE", f"{rmse_val:,.0f}")
            c2.metric("MAE", f"{mae_val:,.0f}")
            c3.metric("MAPE (%)", f"{mape_val:.2f}")

        with st.expander("📈 Forecast Results", expanded=True):
            st.dataframe(forecast_df)
            chart = alt.Chart(combined).mark_line(point=True).encode(
                x=alt.X("Year:Q", axis=alt.Axis(format="d")),
                y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
                color=alt.Color("Type:N", scale=alt.Scale(domain=["Actual","Forecast"], range=["#2E7D32","#FF8F00"])),
                tooltip=["Year","Value","Type"]
            ).interactive()
            st.altair_chart(chart, use_container_width=True)

        with st.expander("📥 Download Forecast Data"):
            st.download_button(
                "Download CSV",
                combined.to_csv(index=False),
                "kenya_agriculture_forecast.csv",
                "text/csv"
            )

    else:
        st.warning("Not enough data for selected look-back window.")

else:
    st.info("Upload FAOSTAT CSV file to begin.")
