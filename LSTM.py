"""
Kenya Agricultural Forecast Dashboard (LSTM)
1960–2020 Data → Predict 2021–2025
Omari Galana Shevo – MUST
"""

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import time
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Agricultural Forecast (LSTM)",
    page_icon="🌾",
    layout="wide"
)

# ──────────────────────────────────────────────
# GLOBAL STYLES
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #F5F7F2; }
    .stMetric { background-color: #E8F5E9; padding:10px; border-radius:10px; }
    .stSidebar .sidebar-content { background-color: #F0F4F1; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# TITLE
# ──────────────────────────────────────────────
st.title("🌾 Kenya Agricultural Production Forecast (LSTM)")
st.markdown("""
Forecast agricultural production in Kenya using FAOSTAT data (1960–2020)  
**Model:** LSTM implemented with TensorFlow/Keras
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
    forecast_horizon = st.sidebar.slider(
        "Forecast Years (2021–2025)",
        1, 5, 5
    )
    st.sidebar.markdown("""
- **Look-back Window:** Past years used for prediction  
- **Forecast Years:** How many years ahead to forecast
""")

    # ──────────────────────────────────────────────
    # PREPARE SERIES
    # ──────────────────────────────────────────────
    series_df = df[df["Item"] == crop_selected].sort_values("Year")
    values = series_df["Value"].values.reshape(-1,1)
    years = series_df["Year"].values

    st.subheader("📋 Historical Data Summary")
    st.dataframe(series_df.describe().transpose())

    hist_chart = alt.Chart(series_df).mark_line(point=True, color="#2E7D32").encode(
        x="Year:Q",
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        tooltip=["Year","Value"]
    ).properties(height=300)

    st.altair_chart(hist_chart, use_container_width=True)

    # ──────────────────────────────────────────────
    # SCALE DATA
    # ──────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(values)

    def create_sequences(data, look_back):
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:i+look_back])
            y.append(data[i+look_back])
        return np.array(X), np.array(y)

    X, y_seq = create_sequences(scaled_values, look_back)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Train/Test split (last 5 for testing)
    X_train, X_test = X[:-5], X[-5:]
    y_train, y_test = y_seq[:-5], y_seq[-5:]

    # ──────────────────────────────────────────────
    # BUILD LSTM MODEL
    # ──────────────────────────────────────────────
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=200, verbose=0)

    # ──────────────────────────────────────────────
    # TEST PERFORMANCE
    # ──────────────────────────────────────────────
    y_pred_test = model.predict(X_test, verbose=0)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
    y_pred_inv = scaler.inverse_transform(y_pred_test)

    rmse_val = np.sqrt(np.mean((y_test_inv - y_pred_inv)**2))
    mae_val = np.mean(np.abs(y_test_inv - y_pred_inv))
    mape_val = np.mean(np.abs((y_test_inv - y_pred_inv)/y_test_inv))*100

    st.subheader("📊 Model Performance")
    c1,c2,c3 = st.columns(3)
    c1.metric("RMSE", f"{rmse_val:,.0f}")
    c2.metric("MAE", f"{mae_val:,.0f}")
    c3.metric("MAPE (%)", f"{mape_val:.2f}")

    # ──────────────────────────────────────────────
    # FUTURE FORECAST
    # ──────────────────────────────────────────────
    last_sequence = scaled_values[-look_back:].reshape(1,look_back,1)
    future_predictions = []

    progress = st.progress(0)
    for i in range(forecast_horizon):
        next_pred = model.predict(last_sequence, verbose=0)
        future_predictions.append(next_pred[0,0])
        last_sequence = np.append(last_sequence[:,1:,:], [[next_pred]], axis=1)
        progress.progress((i+1)/forecast_horizon)
        time.sleep(0.05)
    progress.empty()

    future_predictions_inv = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
    future_years = list(range(2021, 2021+forecast_horizon))

    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Value": future_predictions_inv.flatten(),
        "Type":"Forecast"
    })

    history_df = series_df[["Year","Value"]].copy()
    history_df["Type"]="Actual"
    combined = pd.concat([history_df, forecast_df])

    st.subheader("📈 Forecast Results")
    st.dataframe(forecast_df)

    chart = alt.Chart(combined).mark_line(point=True).encode(
        x=alt.X("Year:Q", axis=alt.Axis(format="d")),
        y=alt.Y("Value:Q", title="Production (tonnes)", axis=alt.Axis(format="~s")),
        color=alt.Color(
            "Type:N",
            scale=alt.Scale(domain=["Actual","Forecast"], range=["#2E7D32","#FF8F00"])
        ),
        tooltip=["Year","Value","Type"]
    ).interactive()

    st.altair_chart(chart, use_container_width=True)

    st.download_button(
        "📥 Download Forecast CSV",
        combined.to_csv(index=False),
        "kenya_agriculture_forecast_lstm.csv",
        "text/csv"
    )

else:
    st.info("Upload FAOSTAT CSV file to begin.")
