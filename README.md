# Transparent Water Quality Alert System

A modular machine learning and telemetry forecasting platform that classifies current water quality, predicts 6-hour and 24-hour parameter trends, evaluates future quality degradation, and generates operational alerts.

---

> ### ⚠️ CRITICAL NOTICE & DISCLAIMER
> **ALL DATA IN THIS SYSTEM IS SYNTHETIC / SIMULATED.**  
> This project operates solely on mathematically modeled synthetic telemetry for educational algorithm prototyping. It **does not represent real-world physical water samples** and **must not be used to certify drinking water safety, potability, or compliance with regulatory public health standards**.

---

## 🌟 Key Features

1. **Synthetic Telemetry Simulation (`src/data_simulator.py`):**
   - Models realistic diurnal (day-night) cycles for water temperature, dissolved oxygen, and pH.
   - Simulates sensor noise and controllable pollution anomalies (storm runoff, acid spikes, hypoxia).
2. **Current Quality Classification (`src/current_classifier.py`):**
   - Categorizes telemetry readings into **Good**, **Needs Treatment**, or **Poor**.
   - Includes graceful rule-based fallback when model weights are not yet trained.
3. **Multi-Horizon Forecasting (`src/forecasting.py`):**
   - Forecasts numerical parameter values at **+6-hour** and **+24-hour** horizons.
   - Evaluates forecasted vectors to project future quality categories and warning rationale.
   - **Persistent Storage:** Stored in the `predictions` table with station ID, creation timestamp, target timestamp, horizon, predicted category, and parameter vectors.
4. **Early-Warning Alert Intelligence (`src/alert_engine.py`):**
   - Detects instantaneous threshold breaches (`CURRENT`) and anticipatory predictive degradation (`PREDICTIVE`).
   - Assigns severity levels (`INFO`, `WARNING`, `CRITICAL`) with operator action recommendations.
   - **Event Deduplication:** Suppresses duplicate alert insertions across rapid clicks or dashboard reruns using stable composite keys (`station:alert_type:severity:event_key:event_window`).
5. **Dual-Mode Persistence Layer (`src/database.py`):**
   - Integrates with **Supabase PostgreSQL** in the cloud (configured when credentials exist).
   - Automatically defaults to zero-configuration **local SQLite** (`data/local_water_quality.db`) for offline execution.
   - Features automatic, non-destructive schema migration for `predictions` and `alerts` tables.
6. **Interactive Streamlit Console (`app.py`):**
   - Station selector, time range filter, interactive Plotly charts, KPI metrics, stored forecast history, and operational alert center with source filtering.

---

## 📁 Project Directory Structure

```text
water_quality_system/
├── app.py                     # Interactive Streamlit dashboard
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview & documentation
├── PROJECT_PLAN.md            # Detailed architecture, roadmap & team allocation
├── REUSE_REVIEW.md            # Original repository analysis & GPL-3.0 review
├── .gitignore                 # Version control exclusions
├── .env.example               # Environment variable templates
├── config/
│   └── thresholds.yaml        # Standard reference thresholds & weights
├── data/
│   └── sample/                # Sample simulated CSV datasets & documentation
├── src/
│   ├── __init__.py
│   ├── data_simulator.py      # Telemetry generator with diurnal cycles & anomalies
│   ├── preprocessing.py       # Data cleaning, cyclical features & scalers
│   ├── labeling.py            # Rule-based ground truth categorizer
│   ├── current_classifier.py  # Machine learning classifier for Good/Needs Treatment/Poor
│   ├── forecasting.py         # Multi-output regressors for +6h & +24h horizons
│   ├── alert_engine.py        # Real-time breach & predictive early-warning alert engine
│   └── database.py            # Supabase PostgreSQL client with SQLite offline fallback
├── scripts/
│   ├── generate_data.py       # CLI script to generate bulk synthetic data
│   └── train_models.py        # CLI script to train and serialize all models
├── models/
│   └── README.md              # Model artifacts documentation
└── tests/
    ├── test_simulator.py      # Tests for synthetic data generator
    ├── test_classifier.py     # Tests for quality classification logic
    ├── test_forecasting.py    # Tests for 6h and 24h parameter forecasting
    ├── test_alerts.py         # Tests for alert generation rules
    └── test_persistence.py    # Tests for forecast/alert storage, migrations & deduplication
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12 recommended.

### 2. Set Up Virtual Environment
```bash
# Navigate to the project subfolder
cd water_quality_system

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Generate Simulated Telemetry Data
```bash
python scripts/generate_data.py --days 30 --interval 60 --save-to-db
```

### 4. Train Models
```bash
python scripts/train_models.py
```
This script trains and serializes:
- `models/current_classifier.joblib`
- `models/forecaster_6h.joblib`
- `models/forecaster_24h.joblib`

### 5. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 6. Run Unit Tests
```bash
pytest tests/ -v
```

---

## 👥 Five-Member Team Division of Work

For the complete task descriptions, timelines, and milestone roadmap, refer to [PROJECT_PLAN.md](file:///c:/Users/tjeev/Downloads/CC_project/transparent-water-quality-alert-system/water_quality_system/PROJECT_PLAN.md).

| Member | Focus Area | Primary Deliverables |
| :--- | :--- | :--- |
| **Member 1** | Data Simulation & Telemetry | `src/data_simulator.py`, `scripts/generate_data.py`, `tests/test_simulator.py` |
| **Member 2** | Feature Engineering & Thresholds | `src/preprocessing.py`, `src/labeling.py`, `config/thresholds.yaml` |
| **Member 3** | ML Classification & Forecasting | `src/current_classifier.py`, `src/forecasting.py`, `scripts/train_models.py` |
| **Member 4** | Alert Engine & Cloud Database | `src/alert_engine.py`, `src/database.py`, `tests/test_alerts.py`, Supabase DDL |
| **Member 5** | Streamlit UI & Deployment | `app.py`, interactive charts, demo workflows, containerization |

---

## 📜 Original Repository Review & Licensing

This project is an isolated clean-room redesign created inside `water_quality_system/` to avoid legacy dependency conflicts (GDAL / GeoPandas / Google Earth Engine) and respect **GPL-3.0** copyleft requirements.  
See [REUSE_REVIEW.md](file:///c:/Users/tjeev/Downloads/CC_project/transparent-water-quality-alert-system/water_quality_system/REUSE_REVIEW.md) for full legal and technical evaluation.
