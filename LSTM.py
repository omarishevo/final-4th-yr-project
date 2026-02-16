"""
============================================================
 Predicting Agricultural Seasonal Trends in Kenya (1960-2020)
 Using Deep Learning (LSTM) — Streamlit Application
 Author : Omari Galana Shevo | Meru University of Science & Technology
============================================================

HOW TO RUN
----------
1. Install dependencies:
   pip install streamlit tensorflow scikit-learn pandas matplotlib seaborn plotly

2. Place the dataset CSV in the same folder as this script, OR upload via the
   sidebar in the app.

3. Launch:
   streamlit run kenya_agri_lstm_app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import io
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Kenya Agricultural LSTM Forecasting",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #f8faf5; }

    /* Header banner */
    .banner {
        background: linear-gradient(135deg, #1a5c2a, #2e8b57, #3cb371);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }
    .banner h1 { margin: 0; font-size: 2.0rem; font-weight: 700; }
    .banner p  { margin: 6px 0 0 0; font-size: 1.0rem; opacity: 0.9; }

    /* Metric cards */
    .metric-card {
        background: white;
        border-left: 5px solid #2e8b57;
        border-radius: 8px;
        padding: 16px 20px;
        margin: 4px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .metric-card h3 { margin: 0 0 4px 0; color: #555; font-size: 0.85rem; text-transform: uppercase; }
    .metric-card p  { margin: 0; font-size: 1.6rem; font-weight: 700; color: #1a5c2a; }

    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #1a5c2a;
        border-bottom: 2px solid #2e8b57;
        padding-bottom: 6px;
        margin-bottom: 16px;
    }

    /* Info box */
    .info-box {
        background: #e8f5e9;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 10px 0;
        font-size: 0.92rem;
        color: #2e5d3a;
    }

    /* Sidebar styles */
    .css-1d391kg { background-color: #f0f7f0; }
    
    /* Training progress */
    .training-log {
        background: #1e1e1e;
        color: #00ff88;
        font-family: monospace;
        font-size: 0.8rem;
        padding: 12px;
        border-radius: 8px;
        height: 200px;
        overflow-y: auto;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_MAP = {
    "Maize (corn)": "Maize",
    "Tea leaves": "Tea",
    "Coffee, green": "Coffee",
    "Wheat": "Wheat",
    "Oranges": "Oranges",
}
CROP_COLORS = {
    "Maize":   "#f4b942",
    "Tea":     "#2e8b57",
    "Coffee":  "#6b3a2a",
    "Wheat":   "#c9a227",
    "Oranges": "#ff7f0e",
}


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading & Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_and_preprocess(uploaded_file):
    """Load the FAO Kenya CSV and return a clean wide-format DataFrame."""
    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)
    else:
        return None, None

    # Filter: Production element & key crops
    crops_of_interest = list(CROP_MAP.keys())
    df_prod = df_raw[
        (df_raw["Element"] == "Production") &
        (df_raw["Item"].isin(crops_of_interest))
    ][["Item", "Year", "Value"]].copy()

    df_prod["Item"] = df_prod["Item"].map(CROP_MAP)
    df_prod = df_prod.sort_values("Year")

    # Pivot to wide format
    df_wide = df_prod.pivot_table(index="Year", columns="Item", values="Value", aggfunc="first")
    df_wide = df_wide.reset_index()
    df_wide.columns.name = None

    # Keep only 1960-2020
    df_wide = df_wide[(df_wide["Year"] >= 1960) & (df_wide["Year"] <= 2020)]
    df_wide = df_wide.reset_index(drop=True)

    # Forward/backward fill small gaps
    crop_cols = [c for c in df_wide.columns if c != "Year"]
    df_wide[crop_cols] = df_wide[crop_cols].interpolate(method="linear").fillna(method="bfill").fillna(method="ffill")

    return df_wide, df_prod


def create_sequences(data, seq_len):
    """Create (X, y) sequences for LSTM from a 1-D numpy array."""
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)


def build_lstm_model(seq_len, lstm_units, dropout_rate, num_lstm_layers, bidirectional, learning_rate):
    """Build and compile the LSTM model based on user-specified hyperparameters."""
    model = Sequential(name="Kenya_Agri_LSTM")

    for i in range(num_lstm_layers):
        return_seq = (i < num_lstm_layers - 1)
        if bidirectional:
            layer = Bidirectional(
                LSTM(lstm_units, return_sequences=return_seq, activation="tanh"),
                input_shape=(seq_len, 1) if i == 0 else None,
            )
        else:
            layer = LSTM(
                lstm_units,
                return_sequences=return_seq,
                activation="tanh",
                input_shape=(seq_len, 1) if i == 0 else None,
            )
        model.add(layer)
        model.add(Dropout(dropout_rate))

    model.add(Dense(32, activation="relu"))
    model.add(Dense(1, activation="linear"))

    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def compute_metrics(y_true, y_pred):
    """Return RMSE, MAE, MAPE as a dictionary."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # Avoid division by zero
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE (%)": mape}


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_trend(df, crop):
    color = CROP_COLORS.get(crop, "#2e8b57")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df[crop], mode="lines+markers",
        line=dict(color=color, width=2.5),
        marker=dict(size=5),
        name=crop,
        hovertemplate="Year: %{x}<br>Production: %{y:,.0f} tonnes<extra></extra>",
    ))
    fig.update_layout(
        title=f"{crop} — Historical Production Trend (1960–2020)",
        xaxis_title="Year", yaxis_title="Production (tonnes)",
        template="plotly_white", height=380,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_all_crops_normalised(df):
    crop_cols = [c for c in df.columns if c != "Year"]
    fig = go.Figure()
    for crop in crop_cols:
        s = df[crop]
        s_norm = (s - s.min()) / (s.max() - s.min() + 1e-9)
        fig.add_trace(go.Scatter(
            x=df["Year"], y=s_norm, mode="lines", name=crop,
            line=dict(color=CROP_COLORS.get(crop, "#888"), width=2),
        ))
    fig.update_layout(
        title="Normalised Production Comparison — All Crops",
        xaxis_title="Year", yaxis_title="Normalised Production (0–1)",
        template="plotly_white", height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_correlation_heatmap(df):
    crop_cols = [c for c in df.columns if c != "Year"]
    corr = df[crop_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, annot=True, fmt=".2f", cmap="YlGn",
        ax=ax, linewidths=0.5, square=True,
    )
    ax.set_title("Pearson Correlation Between Crop Productions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def plot_forecast(years_train, y_train_inv, years_test, y_test_inv, y_pred_inv,
                  future_years, future_preds, crop):
    color = CROP_COLORS.get(crop, "#2e8b57")
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=years_train, y=y_train_inv.ravel(),
        mode="lines", name="Training Data",
        line=dict(color="#aaaaaa", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=years_test, y=y_test_inv.ravel(),
        mode="lines+markers", name="Actual (Test)",
        line=dict(color=color, width=2.5),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=years_test, y=y_pred_inv.ravel(),
        mode="lines+markers", name="Predicted (Test)",
        line=dict(color="#ff4444", width=2.5, dash="dash"),
        marker=dict(size=6, symbol="x"),
    ))
    fig.add_trace(go.Scatter(
        x=future_years, y=future_preds,
        mode="lines+markers", name="Future Forecast",
        line=dict(color="#0077cc", width=2.5, dash="dot"),
        marker=dict(size=7, symbol="diamond"),
    ))

    # Shaded future zone
    fig.add_vrect(
        x0=future_years[0], x1=future_years[-1],
        fillcolor="lightblue", opacity=0.10, layer="below", line_width=0,
    )

    fig.update_layout(
        title=f"{crop} — LSTM Forecast vs Actual",
        xaxis_title="Year", yaxis_title="Production (tonnes)",
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def plot_loss_curve(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=history.history["loss"], mode="lines", name="Train Loss",
        line=dict(color="#2e8b57", width=2),
    ))
    if "val_loss" in history.history:
        fig.add_trace(go.Scatter(
            y=history.history["val_loss"], mode="lines", name="Val Loss",
            line=dict(color="#ff4444", width=2, dash="dash"),
        ))
    fig.update_layout(
        title="Training & Validation Loss", xaxis_title="Epoch",
        yaxis_title="MSE Loss", template="plotly_white", height=320,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def plot_residuals(y_true, y_pred, crop):
    residuals = y_true.ravel() - y_pred.ravel()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Residuals over Time", "Residuals Distribution"))
    fig.add_trace(go.Scatter(
        y=residuals, mode="lines+markers",
        line=dict(color="#2e8b57"), name="Residuals",
    ), row=1, col=1)
    fig.add_trace(go.Histogram(
        x=residuals, nbinsx=12,
        marker_color="#2e8b57", opacity=0.75, name="Frequency",
    ), row=1, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
    fig.update_layout(
        title=f"{crop} — Residual Analysis",
        template="plotly_white", height=320, showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/Flag_of_Kenya.svg/320px-Flag_of_Kenya.svg.png",
             use_column_width=True)
    st.markdown("## 🌾 Kenya Agri LSTM")
    st.markdown("**Meru University of Science & Technology**")
    st.markdown("*BSc Data Science — 4th Year Project*")
    st.markdown("---")

    # Dataset
    st.markdown("### 📁 Dataset")
    uploaded_file = st.file_uploader(
        "Upload `Kenyas_Agricultural_Production.csv`",
        type=["csv"],
        help="FAO Kenya crops dataset (1961–2021)",
    )

    st.markdown("---")
    st.markdown("### 🌿 Crop Selection")
    crop_choice = st.selectbox(
        "Select crop to model",
        ["Maize", "Tea", "Coffee", "Wheat", "Oranges"],
        index=0,
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Hyperparameters")

    seq_len = st.slider("Sequence Length (look-back years)", 3, 15, 5,
                        help="Number of past years used to predict one future year.")
    lstm_units = st.select_slider("LSTM Units per Layer", [32, 64, 128, 256], value=64)
    num_lstm_layers = st.radio("Number of LSTM Layers", [1, 2, 3], index=1, horizontal=True)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
    bidirectional = st.checkbox("Bidirectional LSTM", value=False,
                                help="Wraps each LSTM layer in a Bidirectional wrapper.")
    epochs = st.slider("Epochs", 50, 500, 200, 50)
    batch_size = st.select_slider("Batch Size", [4, 8, 16, 32], value=8)
    learning_rate = st.select_slider("Learning Rate", [1e-4, 5e-4, 1e-3, 5e-3, 1e-2], value=1e-3)
    train_split = st.slider("Train / Test Split (%)", 60, 90, 80, 5,
                            help="Percentage of data used for training.")
    forecast_years = st.slider("Future Forecast Horizon (years)", 1, 15, 5)

    st.markdown("---")
    run_btn = st.button("🚀 Train LSTM Model", use_container_width=True, type="primary")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="banner">
  <h1>🌾 Predicting Agricultural Seasonal Trends in Kenya</h1>
  <p>Long Short-Term Memory (LSTM) Deep Learning Forecasting System &nbsp;|&nbsp; 1960 – 2020 Dataset</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────
if uploaded_file is None:
    st.markdown("""
    <div class="info-box">
    ℹ️ <b>Getting Started:</b> Please upload the <code>Kenyas_Agricultural_Production.csv</code> file
    using the sidebar panel on the left. The dataset should contain FAO Kenya crop production records.
    </div>
    """, unsafe_allow_html=True)

    # Show a demo architecture diagram
    st.markdown('<p class="section-title">📐 LSTM Model Architecture</p>', unsafe_allow_html=True)
    arch_md = """
    ```
    ┌─────────────────────────────────────────────────────┐
    │                   Input Layer                        │
    │         Shape: (batch, seq_len, 1)                  │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │           LSTM Layer 1  (64 units)                   │
    │     return_sequences=True  |  activation=tanh        │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │           Dropout Layer  (rate=0.2)                  │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │           LSTM Layer 2  (64 units)                   │
    │     return_sequences=False |  activation=tanh        │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │           Dropout Layer  (rate=0.2)                  │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │          Dense Layer  (32 units, ReLU)               │
    └─────────────────┬───────────────────────────────────┘
                      │
    ┌─────────────────▼───────────────────────────────────┐
    │          Output Dense  (1 unit, Linear)              │
    │          → Next year production forecast             │
    └─────────────────────────────────────────────────────┘
    ```
    """
    st.markdown(arch_md)
    st.stop()


df_wide, df_long = load_and_preprocess(uploaded_file)

if df_wide is None:
    st.error("Failed to load dataset. Please ensure you uploaded the correct FAO Kenya CSV file.")
    st.stop()

crop_cols = [c for c in df_wide.columns if c != "Year"]

# Verify chosen crop exists
if crop_choice not in df_wide.columns:
    st.error(f"Crop '{crop_choice}' not found in dataset. Available: {crop_cols}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Tab Layout
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data Exploration",
    "🤖 Model Training & Forecast",
    "📈 Results & Metrics",
    "📋 About the Study",
])

# =========================================================================
# TAB 1 — Data Exploration
# =========================================================================
with tab1:
    st.markdown('<p class="section-title">📊 Data Exploration & Visualisation</p>', unsafe_allow_html=True)

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    for col, crop in zip([col1, col2, col3, col4], crop_cols[:4]):
        latest = df_wide[crop].iloc[-1]
        peak   = df_wide[crop].max()
        col.markdown(f"""
        <div class="metric-card">
            <h3>{crop}</h3>
            <p>{latest:,.0f} t</p>
            <small style="color:#888">Peak: {peak:,.0f} t</small>
        </div>""", unsafe_allow_html=True)
    if len(crop_cols) > 4:
        crop = crop_cols[4]
        latest = df_wide[crop].iloc[-1]
        peak   = df_wide[crop].max()
        st.markdown(f"""
        <div class="metric-card" style="width:24%">
            <h3>{crop}</h3>
            <p>{latest:,.0f} t</p>
            <small style="color:#888">Peak: {peak:,.0f} t</small>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Individual trend chart
    st.subheader(f"📉 {crop_choice} Historical Production Trend")
    st.plotly_chart(plot_trend(df_wide, crop_choice), use_container_width=True)

    # All-crops normalised
    st.subheader("📊 All Crops — Normalised Comparison")
    st.plotly_chart(plot_all_crops_normalised(df_wide), use_container_width=True)

    # Correlation heatmap
    st.subheader("🔗 Crop Production Correlation")
    col_heat, col_info = st.columns([3, 2])
    with col_heat:
        fig_corr = plot_correlation_heatmap(df_wide)
        st.pyplot(fig_corr, use_container_width=True)
    with col_info:
        st.markdown("""
        <div class="info-box">
        <b>Interpreting the Heatmap</b><br>
        Values close to <b>+1.0</b> indicate strong positive correlation
        (crops grow/decline together), while values near <b>0</b> indicate
        independence. This helps identify shared climate sensitivities
        among Kenya's major crops.
        </div>""", unsafe_allow_html=True)

    # Data table
    with st.expander("📋 View Raw Dataset (Production in Tonnes)"):
        styled_df = df_wide.set_index("Year")
        st.dataframe(styled_df.style.format("{:,.1f}").background_gradient(cmap="YlGn"), height=350)

    # Descriptive statistics
    with st.expander("📐 Descriptive Statistics"):
        st.dataframe(df_wide[crop_cols].describe().applymap(lambda x: f"{x:,.2f}"), use_container_width=True)


# =========================================================================
# TAB 2 — Model Training & Forecast
# =========================================================================
with tab2:
    st.markdown('<p class="section-title">🤖 LSTM Model Training & Forecasting</p>', unsafe_allow_html=True)

    # Model config summary
    arch_col, param_col = st.columns(2)
    with arch_col:
        st.markdown(f"""
        <div class="info-box">
        <b>Model Architecture:</b><br>
        • {'Bidirectional ' if bidirectional else ''}LSTM × {num_lstm_layers} layers ({lstm_units} units each)<br>
        • Dropout: {dropout_rate} &nbsp;|&nbsp; Dense (32, ReLU) → Dense (1, Linear)<br>
        • Optimiser: Adam (lr={learning_rate}) &nbsp;|&nbsp; Loss: MSE
        </div>""", unsafe_allow_html=True)
    with param_col:
        st.markdown(f"""
        <div class="info-box">
        <b>Training Configuration:</b><br>
        • Crop: <b>{crop_choice}</b> &nbsp;|&nbsp; Sequence length: {seq_len} years<br>
        • Train/Test split: {train_split}% / {100-train_split}%<br>
        • Epochs: {epochs} &nbsp;|&nbsp; Batch size: {batch_size}
        </div>""", unsafe_allow_html=True)

    if not run_btn:
        st.info("👈 Configure the model in the sidebar, then click **🚀 Train LSTM Model** to begin.")
        st.stop()

    # ── Data preparation ──────────────────────────────────────────────────
    series = df_wide[crop_choice].values.reshape(-1, 1).astype(float)
    years  = df_wide["Year"].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series)

    split_idx = int(len(scaled) * train_split / 100)

    train_data = scaled[:split_idx]
    test_data  = scaled[split_idx - seq_len:]          # keep seq_len overlap for continuity

    X_train, y_train = create_sequences(train_data, seq_len)
    X_test,  y_test  = create_sequences(test_data,  seq_len)

    # LSTM expects (samples, timesteps, features)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    # Corresponding years
    years_train = years[seq_len : split_idx]
    years_test  = years[split_idx : split_idx + len(y_test)]

    # ── Build model ───────────────────────────────────────────────────────
    model = build_lstm_model(seq_len, lstm_units, dropout_rate,
                             num_lstm_layers, bidirectional, learning_rate)

    model_summary_buf = io.StringIO()
    model.summary(print_fn=lambda x: model_summary_buf.write(x + "\n"))
    model_summary_str = model_summary_buf.getvalue()

    # ── Train ─────────────────────────────────────────────────────────────
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=30, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=15, min_lr=1e-6, verbose=0),
    ]

    progress_bar  = st.progress(0, text="Training LSTM…")
    status_text   = st.empty()

    class StreamlitCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            pct = int((epoch + 1) / epochs * 100)
            progress_bar.progress(pct, text=f"Epoch {epoch+1}/{epochs} — "
                                             f"loss: {logs['loss']:.4f}  "
                                             f"val_loss: {logs.get('val_loss', 0):.4f}")

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=callbacks + [StreamlitCallback()],
        verbose=0,
        shuffle=False,
    )

    progress_bar.progress(100, text="✅ Training complete!")
    actual_epochs = len(history.history["loss"])
    status_text.success(f"Training finished in {actual_epochs} epochs (early stopping applied if < {epochs})")

    # ── Predictions ───────────────────────────────────────────────────────
    y_pred_scaled = model.predict(X_test, verbose=0)
    y_pred_inv    = scaler.inverse_transform(y_pred_scaled)
    y_test_inv    = scaler.inverse_transform(y_test.reshape(-1, 1))
    y_train_inv   = scaler.inverse_transform(y_train.reshape(-1, 1))

    # ── Future Forecast ───────────────────────────────────────────────────
    last_seq   = scaled[-seq_len:].reshape(1, seq_len, 1).copy()
    future_pred_list = []
    for _ in range(forecast_years):
        nxt = model.predict(last_seq, verbose=0)[0, 0]
        future_pred_list.append(nxt)
        last_seq = np.append(last_seq[:, 1:, :], [[[nxt]]], axis=1)

    future_preds = scaler.inverse_transform(
        np.array(future_pred_list).reshape(-1, 1)
    ).ravel()
    future_years = list(range(int(years[-1]) + 1, int(years[-1]) + 1 + forecast_years))

    # ── Store in session state ────────────────────────────────────────────
    st.session_state["model"]         = model
    st.session_state["history"]       = history
    st.session_state["y_test_inv"]    = y_test_inv
    st.session_state["y_pred_inv"]    = y_pred_inv
    st.session_state["y_train_inv"]   = y_train_inv
    st.session_state["years_train"]   = years_train
    st.session_state["years_test"]    = years_test
    st.session_state["future_years"]  = future_years
    st.session_state["future_preds"]  = future_preds
    st.session_state["crop_choice"]   = crop_choice
    st.session_state["summary_str"]   = model_summary_str
    st.session_state["trained"]       = True

    # ── Forecast Chart ────────────────────────────────────────────────────
    st.subheader("📈 Forecast vs Actual")
    st.plotly_chart(
        plot_forecast(years_train, y_train_inv, years_test, y_test_inv,
                      y_pred_inv, future_years, future_preds, crop_choice),
        use_container_width=True,
    )

    # Future forecast table
    st.subheader(f"🔮 Future Forecast ({future_years[0]}–{future_years[-1]})")
    forecast_df = pd.DataFrame({
        "Year": future_years,
        f"{crop_choice} Production Forecast (tonnes)": future_preds.round(1),
    })
    col_t, col_c = st.columns([1, 2])
    with col_t:
        st.dataframe(forecast_df.set_index("Year").style.format("{:,.1f}"), use_container_width=True)
    with col_c:
        fig_fut = go.Figure(go.Bar(
            x=future_years,
            y=future_preds,
            marker_color=CROP_COLORS.get(crop_choice, "#2e8b57"),
            text=[f"{v:,.0f} t" for v in future_preds],
            textposition="outside",
        ))
        fig_fut.update_layout(
            title=f"{crop_choice} — Forecasted Production",
            xaxis_title="Year", yaxis_title="Tonnes",
            template="plotly_white", height=320,
            margin=dict(l=20, r=20, t=50, b=20),
        )
        st.plotly_chart(fig_fut, use_container_width=True)


# =========================================================================
# TAB 3 — Results & Metrics
# =========================================================================
with tab3:
    st.markdown('<p class="section-title">📈 Results, Metrics & Model Analysis</p>', unsafe_allow_html=True)

    if not st.session_state.get("trained"):
        st.info("⚠️ Train the model first (Tab 2) to see results here.")
        st.stop()

    y_test_inv  = st.session_state["y_test_inv"]
    y_pred_inv  = st.session_state["y_pred_inv"]
    history     = st.session_state["history"]
    crop        = st.session_state["crop_choice"]
    model       = st.session_state["model"]
    summary_str = st.session_state["summary_str"]

    metrics = compute_metrics(y_test_inv.ravel(), y_pred_inv.ravel())

    # ── Metrics cards ──────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.markdown(f"""<div class="metric-card">
        <h3>RMSE (tonnes)</h3><p>{metrics['RMSE']:,.0f}</p>
        <small style="color:#888">Root Mean Squared Error</small></div>""",
        unsafe_allow_html=True)
    m2.markdown(f"""<div class="metric-card">
        <h3>MAE (tonnes)</h3><p>{metrics['MAE']:,.0f}</p>
        <small style="color:#888">Mean Absolute Error</small></div>""",
        unsafe_allow_html=True)
    m3.markdown(f"""<div class="metric-card">
        <h3>MAPE (%)</h3><p>{metrics['MAPE (%)']:.2f}%</p>
        <small style="color:#888">Mean Absolute % Error</small></div>""",
        unsafe_allow_html=True)

    st.markdown("---")

    # ── Training loss curve ─────────────────────────────────────────────
    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("📉 Training Loss Curve")
        st.plotly_chart(plot_loss_curve(history), use_container_width=True)

    with col_r:
        st.subheader("🔍 Residual Analysis")
        st.plotly_chart(plot_residuals(y_test_inv, y_pred_inv, crop), use_container_width=True)

    # ── Predicted vs Actual scatter ─────────────────────────────────────
    st.subheader("🎯 Predicted vs Actual Scatter Plot")
    fig_scatter = go.Figure()
    fig_scatter.add_trace(go.Scatter(
        x=y_test_inv.ravel(), y=y_pred_inv.ravel(),
        mode="markers", marker=dict(color=CROP_COLORS.get(crop, "#2e8b57"), size=9, opacity=0.7),
        name="Predictions",
    ))
    mn = min(y_test_inv.min(), y_pred_inv.min()) * 0.9
    mx = max(y_test_inv.max(), y_pred_inv.max()) * 1.1
    fig_scatter.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines",
                                      line=dict(color="red", dash="dash"), name="Perfect Fit"))
    fig_scatter.update_layout(
        xaxis_title="Actual (tonnes)", yaxis_title="Predicted (tonnes)",
        template="plotly_white", height=380,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ── Model Summary ──────────────────────────────────────────────────
    with st.expander("🧠 Keras Model Summary"):
        st.code(summary_str, language="text")

    # ── Metrics interpretation ─────────────────────────────────────────
    with st.expander("📖 Metrics Interpretation Guide"):
        mape_val = metrics["MAPE (%)"]
        if mape_val < 10:
            rating = "✅ Excellent — MAPE < 10%: Highly accurate forecast."
        elif mape_val < 20:
            rating = "🟡 Good — MAPE 10–20%: Acceptable forecasting accuracy."
        elif mape_val < 50:
            rating = "🟠 Fair — MAPE 20–50%: Moderate accuracy; consider tuning."
        else:
            rating = "🔴 Poor — MAPE > 50%: Consider more data or architecture changes."

        st.markdown(f"""
        **MAPE = {mape_val:.2f}% → {rating}**

        | Metric | Value | Interpretation |
        |--------|-------|----------------|
        | RMSE   | {metrics['RMSE']:,.0f} tonnes | Average magnitude of errors in original units |
        | MAE    | {metrics['MAE']:,.0f} tonnes | Mean deviation of predictions from actual values |
        | MAPE   | {mape_val:.2f}% | Percentage deviation — independent of production scale |

        **Key:** Lower values for all metrics indicate better model performance.
        A MAPE below 20% is generally considered acceptable for agricultural forecasting.
        """)


# =========================================================================
# TAB 4 — About the Study
# =========================================================================
with tab4:
    st.markdown('<p class="section-title">📋 About the Study</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Predicting Agricultural Seasonal Trends in Kenya (1960–2021) Using Deep Learning (LSTM)

        **Author:** Omari Galana Shevo
        **Registration:** CT204/108375/21
        **Supervisor:** Dr. Kibaara
        **Institution:** Meru University of Science and Technology
        **Department:** Computer Science / Data Science

        ---

        #### 🎯 Study Objectives

        1. Preprocess and analyse Kenya's agricultural production data (1961–2021)
        2. Identify long-term and seasonal production trends for major crops
        3. Design and train an LSTM-based forecasting model
        4. Evaluate model performance using RMSE, MAE, and MAPE
        5. Generate forecasts and visualisations of future agricultural production trends

        ---

        #### 🌱 Crops Studied
        | Crop | Significance |
        |------|-------------|
        | Maize | Staple food crop; dominant in production volume |
        | Tea | Major export earner; consistent long-term growth |
        | Coffee | Key export commodity; sensitive to climate |
        | Wheat | Important cereal; partly import-dependent |
        | Oranges | Horticultural export with growing demand |

        ---

        #### 🧠 Why LSTM?

        Long Short-Term Memory (LSTM) neural networks are a specialised form of
        Recurrent Neural Network (RNN) designed to overcome the *vanishing gradient*
        problem. Key advantages for this study:

        - Captures **long-term temporal dependencies** (decades of production history)
        - Models **non-linear patterns** that conventional methods miss
        - Handles **sequential data** natively — ideal for annual time-series
        - Learns **crop-specific seasonal signatures** from historical fluctuations

        ---

        #### 📐 LSTM Architecture (Proposed)
        ```
        Input → [LSTM(64) → Dropout(0.2)] × 2 → Dense(32, ReLU) → Dense(1, Linear)
        Optimiser: Adam | Loss: MSE | Metrics: RMSE, MAE, MAPE
        ```

        ---

        #### 📚 References
        - FAO (2023). *FAOSTAT Database*. Food and Agriculture Organization.
        - Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*.
        - Kenya National Bureau of Statistics (KNBS). *Economic Survey Reports*.
        - Chollet, F. (2021). *Deep Learning with Python*. Manning Publications.
        """)

    with col2:
        st.markdown("""
        <div class="info-box">
        <b>📊 Dataset Summary</b><br><br>
        • Source: FAO FAOSTAT<br>
        • Coverage: 1960–2020 (60 years)<br>
        • Crops: Maize, Tea, Coffee, Wheat, Oranges<br>
        • Element: Production (tonnes)<br>
        • Country: Kenya
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:16px">
        <b>⚙️ Technology Stack</b><br><br>
        • Python 3.10+<br>
        • TensorFlow / Keras<br>
        • Streamlit<br>
        • Pandas, NumPy<br>
        • Scikit-learn<br>
        • Plotly, Matplotlib, Seaborn
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-box" style="margin-top:16px">
        <b>📏 Evaluation Metrics</b><br><br>
        • <b>RMSE</b>: Root Mean Squared Error<br>
        • <b>MAE</b>: Mean Absolute Error<br>
        • <b>MAPE</b>: Mean Absolute % Error<br><br>
        Target: MAPE &lt; 20% for reliable forecasting
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#888; font-size:0.85rem; padding:12px 0;">
  🌾 Kenya Agricultural LSTM Forecasting System &nbsp;|&nbsp;
  Meru University of Science and Technology &nbsp;|&nbsp;
  BSc Data Science — 4th Year Project &nbsp;|&nbsp;
  Built with Streamlit + TensorFlow
</div>
""", unsafe_allow_html=True)
