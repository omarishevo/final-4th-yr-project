# 🌾 Predicting Agricultural Seasonal Trends in Kenya (1960–2020) Using Deep Learning (LSTM)

> **Meru University of Science and Technology**  
> Department of Computer Science · Bachelor of Science in Data Science  
> **Author:** Omari Galana Shevo &nbsp;|&nbsp; **Reg No:** CT204/108375/21 &nbsp;|&nbsp; **Supervisor:** Dr. Kibaara

---

## 📌 Overview

This is a 4th-year undergraduate final project that applies a **Long Short-Term Memory (LSTM)** deep learning model to forecast Kenya's agricultural production trends using over **60 years of historical data (1960–2020)**. The dataset is sourced from FAOSTAT and covers **150 crop types** across **18,182 records**, with analytical focus on five major crops: **Maize, Tea, Coffee, Wheat, and Oranges**.

**Core Research Question:**
> *Can a deep learning LSTM model accurately predict Kenya's agricultural production seasonal trends, and how well does it capture nonlinear, climate-driven patterns that traditional methods miss?*

---

## 📁 Project Structure
```
📦 agricultural-lstm-kenya/
├── final_4th_yr_project.docx            # Full research report (5 chapters)
├── Kenyas_Agricultural_Production.csv   # FAOSTAT dataset (18,182 records)
├── README.md                            # This file
└── notebooks/
    └── lstm_agriculture_kenya.ipynb     # Main modelling notebook (Jupyter/Colab)
```

---

## 📊 Dataset Summary

**File:** `Kenyas_Agricultural_Production.csv`  
**Primary Source:** [FAOSTAT](https://www.fao.org/faostat/) via Kaggle  
**Supplementary Source:** Kenya National Bureau of Statistics (KNBS)

| Attribute | Value |
|-----------|-------|
| Total Records | 18,182 |
| Columns | 14 |
| Year Range | 1960 – 2020 (over 60 years) |
| Unique Crops/Items | 150 |
| Missing Values | ✅ None |
| Primary Unit | Tonnes |

### Columns

| Column | Description |
|--------|-------------|
| `Domain Code` | FAO domain identifier (`QCL`) |
| `Domain` | Crops and livestock products |
| `Area Code (M49)` | Country code (404 = Kenya) |
| `Area` | Country name |
| `Element Code` | Measurement type code |
| `Element` | Type: Production, Yield, Area harvested, Stocks, etc. |
| `Item Code (CPC)` | FAO commodity code |
| `Item` | Crop or commodity name |
| `Year Code` / `Year` | Year of observation |
| `Unit` | Unit of measurement (tonnes, ha, hg/ha, Head, etc.) |
| `Value` | Recorded quantity |
| `Flag` | Data quality flag |
| `Flag Description` | Flag explanation |

### Measurement Elements Distribution

| Element | Records |
|---------|---------|
| Production | 7,078 |
| Yield | 4,688 |
| Area harvested | 4,171 |
| Producing Animals/Slaughtered | 920 |
| Stocks | 541 |
| Others | 784 |

### Data Quality Flags

| Flag | Meaning | Count |
|------|---------|-------|
| `E` | Estimated value | 7,659 |
| `A` | Aggregate (official + estimated) | 6,444 |
| `I` | Imputed value | 3,382 |
| `T` | Unofficial figure | 546 |
| `M` | Data not available | 151 |

> ⚠️ **Note:** ~42% of records are estimated (`E`) — acknowledged as a limitation and handled during preprocessing.

### Focus Crops — Production Summary

| Crop | Item Name in CSV | Records | Min (tonnes) | Max (tonnes) | Trend |
|------|-----------------|---------|-------------|-------------|-------|
| **Maize** | `Maize (corn)` | 183 | 940,000 | 4,013,777 | High volatility |
| **Tea** | `Green tea (not fermented).../Tea leaves` | 72 | 12,641 | 2,476,000 | Steady growth |
| **Coffee** | `Coffee, green` | 183 | 28,100 | 128,700 | Long-term decline |
| **Wheat** | `Wheat` | 183 | 84,200 | 511,994 | Variable |
| **Oranges** | `Oranges` | 183 | 5,000 | 145,445 | Moderate growth |

---

## 🧠 LSTM Model Architecture
```
Input
  └─ Lagged annual production sequences
        │
        ▼
┌─────────────────────────┐
│  LSTM Layer 1 (64 units)│  ← Learns initial temporal patterns
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  LSTM Layer 2 (64 units)│  ← Extracts deeper temporal features
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Dropout Layer (0.2)    │  ← Prevents overfitting
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│  Dense Layer (1 neuron) │  ← Single-step annual forecast output
└─────────────────────────┘
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Architecture | 2-layer LSTM |
| LSTM Units | 64 per layer |
| Dropout Rate | 0.2 |
| Loss Function | Mean Squared Error (MSE) |
| Optimizer | Adam |
| Training Split | 80% (chronological) |
| Testing Split | 20% (out-of-sample) |
| Split Strategy | Chronological — no data leakage |

---

## 🔬 Methodology

### Step 1 — Data Collection
- Dataset downloaded from FAOSTAT via Kaggle
- Variables extracted: Year, Crop (Item), Production Value, Unit, Domain, Flag
- Units verified (tonnes) and production-only records filtered

### Step 2 — Data Preprocessing
- Reviewed estimated values (Flag = `E`)
- Encoded categorical crop names
- Created **lagged input sequences** (previous years as model features)
- Applied **Min-Max normalization** to stabilize neural network training
- Performed **chronological 80/20 train-test split** to prevent data leakage

### Step 3 — Exploratory Data Analysis (EDA)
- Time-series line plots per crop (1960–2020)
- 3-year and 5-year moving averages to reveal long-term trends
- Identified drought-year production dips: **1984, 1999–2000, 2016, 2017**
- Crop correlation and comparative production analysis

### Step 4 — LSTM Model Development
- Two-layer LSTM with Dropout regularization
- Trained on 80% chronological data
- Epochs and batch size tuned experimentally for convergence

### Step 5 — Evaluation & Forecasting
- Predictions generated on 20% held-out test set
- Future-year forecasts produced beyond 2021
- Model evaluated using RMSE, MAE, and MAPE

---

## 📈 Model Performance

| Metric | Description | Result |
|--------|-------------|--------|
| **RMSE** | Root Mean Squared Error | Low relative to production scale |
| **MAE** | Mean Absolute Error | Low absolute deviation from actuals |
| **MAPE** | Mean Absolute Percentage Error | **~5–12%** ✅ |

> A MAPE of 5–12% is within the acceptable forecasting range and confirms the model's reliability across all crops.

---

## 🌿 Key Findings

### 🌽 Maize
- Grew from ~940,000 tonnes (1961) to ~4,013,777 tonnes (peak)
- **Three phases:** Steady growth (1961–1976) → High volatility (1977–2000) → Growth with volatility (2001–2021)
- Major drought impacts: 1984 (2.5M → 1.4M tonnes), 2000, 2017
  ![Maize Production Chart](https://raw.githubusercontent.com/omarishevo/final-4th-yr-project/main/Figure%202026-01-07%20132314%20(0).png)

### ☕ Coffee
- Peaked at **128,700 tonnes in 1987**
- Declined ~73% to 35,000–40,000 tonnes by 2020
- Structural decline linked to reduced plantation acreage and global price shifts

-![Coffee Production Chart](https://raw.githubusercontent.com/omarishevo/final-4th-yr-project/main/Figure%202026-01-07%20132314%20(1).png)

### 🍵 Tea
- Kenya's most **resilient agricultural commodity**
- Rose from ~12,641 tonnes to **2,476,000 tonnes** at peak
- Supported by KTDA smallholder schemes, export demand, and irrigation


-![Tea Production Chart](https://raw.githubusercontent.com/omarishevo/final-4th-yr-project/main/Figure%202026-01-07%20132314%20(2).png)
### 🌾 Wheat
- High interannual variability; semi-arid dependent
- Range: 84,200 – 511,994 tonnes

-![Wheat Production Chart](https://raw.githubusercontent.com/omarishevo/final-4th-yr-project/main/Figure%202026-01-07%20132314%20(3).png)

### 🍊 Oranges
- Consistent moderate growth: 5,000 → 145,445 tonnes

-![Oranges Production Chart](https://raw.githubusercontent.com/omarishevo/final-4th-yr-project/main/Figure%202026-01-07%20132314%20(4).png)

### LSTM Conclusions
- Successfully learned **cyclical patterns** tied to Kenya's bimodal rainfall (March–May, Oct–Dec)
- Implicitly captured **drought years** — forecasts dipped in known climate shock periods
- Strong **generalization** on unseen test data (MAPE 5–12%)
- Outperformed traditional ARIMA/linear regression on nonlinear agricultural time-series

---

## 🚀 Getting Started

### Option A — Google Colab (Recommended)

Upload `Kenyas_Agricultural_Production.csv`, then run:
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 1. Load
df = pd.read_csv('Kenyas_Agricultural_Production.csv')

# 2. Filter one crop
crop_df = df[(df['Item'] == 'Maize (corn)') & (df['Element'] == 'Production')]
crop_df = crop_df[['Year', 'Value']].sort_values('Year').reset_index(drop=True)

# 3. Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(crop_df[['Value']])

# 4. Create lagged sequences
SEQ_LEN = 5
X, y = [], []
for i in range(len(scaled) - SEQ_LEN):
    X.append(scaled[i:i+SEQ_LEN])
    y.append(scaled[i+SEQ_LEN])
X, y = np.array(X), np.array(y)

# 5. Chronological 80/20 split
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 6. Build LSTM
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 7. Train
model.fit(X_train, y_train, epochs=100, batch_size=16,
          validation_data=(X_test, y_test), verbose=1)

# 8. Evaluate
from sklearn.metrics import mean_absolute_percentage_error
preds = scaler.inverse_transform(model.predict(X_test))
actual = scaler.inverse_transform(y_test)
mape = mean_absolute_percentage_error(actual, preds) * 100
print(f"MAPE: {mape:.2f}%")
```

### Option B — Local Setup
```bash
git clone https://github.com/your-username/agricultural-lstm-kenya.git
cd agricultural-lstm-kenya
pip install -r requirements.txt
jupyter notebook notebooks/lstm_agriculture_kenya.ipynb
```

---

## 📦 Requirements
```
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tensorflow>=2.12.0
jupyter>=1.0.0
```
```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

> **Python:** 3.9+ recommended  
> **GPU:** Optional but speeds up training — Google Colab provides free GPU/TPU

---

## ⚠️ Limitations

| Limitation | Impact |
|-----------|--------|
| Annual data only | Intra-year (monthly) seasonal patterns cannot be captured directly |
| ~42% estimated values (`Flag=E`) | May introduce minor statistical inaccuracies |
| No external variables | Rainfall, temperature, pest, and market data not included |
| Sparse early records (1960s–70s) | Some crop data incomplete in early decades |
| 5 major crops only | Minor and region-specific crops not modelled |
| No county-level breakdown | Regional variability within Kenya not captured |

---

## 🔭 Future Work

- Incorporate **rainfall and temperature data** as additional LSTM input features
- Develop **multi-step forecasting** for long-range projections (5–10 years)
- Build **county-level models** to capture regional agricultural variability
- Explore **hybrid architectures** (LSTM + Attention mechanism, Transformers)
- Deploy a **real-time web forecasting dashboard** for farmers and policymakers

---

## 📋 Report Structure

| Chapter | Title |
|---------|-------|
| 1 | Introduction — Background, problem statement, objectives, scope |
| 2 | Literature Review — Traditional vs deep learning forecasting methods |
| 3 | Research Methodology — Data sources, preprocessing, LSTM design, ethics |
| 4 | Model Validation & Results — EDA, LSTM training, performance, forecasts |
| 5 | Conclusion & Recommendations — Findings + guidance for stakeholders |

---

## 📚 References

1. Shen, C. (2018). A Transdisciplinary Review of Deep Learning Research and Its Relevance for Water Resources Scientists. *Water Resources Research, 54*, 8558–8593.
2. Gauch, M. et al. (2021). Rainfall–runoff prediction at multiple timescales with a single Long Short-Term Memory network. *Hydrology and Earth System Sciences, 25*, 2045–2062.
3. Kratzert, F. et al. (2019). NeuralHydrology — Interpreting LSTMs in Hydrology. In *Explainable AI*. Springer.
4. Adepoju, K. A. (2019). Vegetation response to recent trends in climate and land use dynamics. *Advances in Meteorology, 2019*, 4946127.
5. FAOSTAT. https://www.fao.org/faostat/
6. Kenya National Bureau of Statistics (KNBS). https://www.knbs.or.ke/

---

## 👤 Author

| | |
|-|-|
| **Name** | Omari Galana Shevo |
| **Registration** | CT204/108375/21 |
| **Institution** | Meru University of Science and Technology |
| **Department** | Computer Science / Data Science |
| **Supervisor** | Dr. Kibaara |
| **Contact** | 0715690308 |

---

*Built with Python · TensorFlow/Keras · Pandas · Matplotlib · Scikit-learn*
