---
description: How to run the Zero-Touch MLOps Platform
---

# Execution Workflow: Zero-Touch MLOps

Follow these steps to run the autonomous MLOps platform.

## 1. Environment Setup
Ensure dependencies are installed and the `PYTHONPATH` is set.
```bash
pip install -r requirements.txt
$env:PYTHONPATH="src"
```

## 2. Start the API Server
Launch the FastAPI backend. It will automatically load the latest production model.
// turbo
```powershell
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```

## 3. Train a New Model (Zero-Touch)
You can train a model on any dataset (Cars, Colleges, etc.). The script will autonomously infer the schema and target.
// turbo
```powershell
python train_model_v1_mlflow.py data/raw/car_sales.csv production_models/v1.0
```

## 4. Verify with the Web UI
1. Open the browser to `http://127.0.0.1:8000`.
2. Navigate to the **PREDICT** section.
3. Observe that the form has dynamically matched your dataset's columns.
4. Input data and click **Generate Intelligence Report**.

## 5. Switch Datasets
To test autonomy, swap the raw data:
1. Replace `data/raw/car_sales.csv` with a different tabular file (e.g., `colleges.csv`).
2. Run the training script (Step 3).
3. The API and UI will automatically update to the new dataset upon refresh.
