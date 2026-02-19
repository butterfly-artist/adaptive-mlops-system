# 🌌 Zero-Touch MLOps Autonomy Platform

A high-performance, autonomous MLOps system that adapts to **any** tabular dataset without code changes. Featuring dynamic schema inference, AutoML preprocessing, and a self-building glassmorphic UI.

## 🚀 Quick Start (Zero-Touch)

### 1. Setup
```bash
pip install -r requirements.txt
$env:PYTHONPATH="src"
```

### 2. Launch the Autonomous API
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Access the UI at `http://127.0.0.1:8000`*

### 3. Trigger Autonomous Retraining
Train on any dataset (Car Sales, College details, or your own CSV):
```powershell
python train_model_v1_mlflow.py data/raw/car_sales.csv production_models/v1.0
```
*The system will automatically detect the target column, numerical features, and categorical classes.*

---

## 🛠️ Key Features

### 🧠 Dataset Autonomy
- **Schema Auto-Inference**: `DataValidator` detects columns, types, and ranges dynamically.
- **AutoML Pipeline**: Preprocessing adapts to feature types on-the-fly (Median Imputation, Scaling, One-Hot Encoding).
- **Domain-Agnostic**: Works for Cars, Colleges, Finance, etc., out of the box.

### 🔒 Enterprise Security
- **API Key Required**: All requests must include `X-API-KEY: MLOPS_PLATFORM_KEY_2026`.
- **Validation Gate**: Predictive inputs are strictly validated against inferred ranges to prevent garbage-in-garbage-out.

### 🎨 Self-Building Web UI
- **Metadata-Driven**: The frontend queries `/schema` to build forms dynamically.
- **Premium Design**: Dark mode, glassmorphism, and interactive Intelligence Reports.

### 📈 Full Lifecycle Observability
- **Drift Detection**: Automatic PSI/KS-Test reports in `reports/`.
- **MLflow Integration**: Every run is logged at `localhost:5000`.
- **Health Dashboard**: Real-time status at `/observability/health`.

---

## 📂 Project Structure

- `api/`: FastAPI backend and Dynamic Frontend.
- `src/data_validation/`: Dynamic schema inference engine.
- `src/preprocessing/`: AutoML preprocessor module.
- `src/monitoring/`: Drift detection and health monitoring.
- `production_models/`: Latest verified model artifacts.

---

## 🧪 Testing

Run the full suite:
```bash
pytest tests/
```

Test a specific prediction:
```powershell
python debug_model.py
```

---
**Status**: Fully Autonomous | **Security**: API Key Enforced | **Aesthetics**: Premium Glassmorphism
