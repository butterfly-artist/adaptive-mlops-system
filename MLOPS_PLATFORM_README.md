# 🚀 Car Sales Prediction MLOps Platform

This platform is a production-ready MLOps solution for car sales prediction, featuring automated data validation, model training with MLflow, drift detection, and autonomous retraining.

## 🏗️ Architecture

### 1. Data Validation (`src/data_validation/`)
- **Strict Schema Enforcement**: Ensures all incoming data matches expected types and ranges.
- **Statistical Checks**: Detects outliers and missing value patterns.
- **Business Logic**: Validates relationships (e.g., resale value < price).

### 2. Training Pipeline (`src/training/`)
- **Deterministic Preprocessing**: Custom sklearn transformers for date extraction and feature scaling.
- **MLflow Tracking**: Every run logs parameters, metrics (RMSE, MAE, R2), and model artifacts.
- **Baseline Comparison**: Mandatory evaluation against baseline models (Mean/Median) before promotion.

### 3. Monitoring & Observability (`src/monitoring/`)
- **Drift Detection**: Uses **PSI** and **KS Tests** to detect feature and prediction drift.
- **System Health**: Centralized `reports/system_health.json` for senior engineer observability.
- **Audit Trail**: Detailed `logs/retraining_history.json` for every lifecycle decision.

### 4. Autonomous Retraining (`src/training/retraining_pipeline.py`)
- **Rule-Based Decision Engine**: Retrains ONLY if:
    - Feature drift > 20%
    - Prediction drift is statistically significant (p < 0.05)
    - Sufficient new data is available (>= 50 samples)
    - Cooldown period has passed (24 hours)
- **Safe Deployment**: Candidate models are compared against current production models. Promotion only happens if performance improves.

## 📊 Observability Reports

- **`reports/drift_summary.json`**: High-level drift status.
- **`reports/feature_drift.json`**: Detailed PSI and KS scores for every feature.
- **`reports/prediction_drift.json`**: Statistical analysis of model output stability.
- **`reports/system_health.json`**: The "Senior Engineer View" of system stability and model performance over time.

## 🛠️ How to Run

### Manual Retraining Check
```bash
python src/training/retraining_pipeline.py
```

### Run Tests
```bash
pytest tests/
```

### View MLflow UI
```bash
mlflow ui --backend-store-uri file:///d:/projects/datamodeMLOPS/mlops-auto-retrain/mlruns
```

## 🤖 Automation
The system is integrated with **GitHub Actions** (`.github/workflows/retrain.yml`) to run weekly retraining checks and model updates automatically.

---
**Status**: Fully Operational | **Reliability**: High | **Observability**: Comprehensive
