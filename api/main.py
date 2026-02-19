"""
Car Sales Prediction API
FastAPI implementation with integrated data validation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Header, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_validation import validate_data
from deployment.model_loader import load_production_model
from deployment.prediction_logger import log_prediction

# Initialize FastAPI
app = FastAPI(
    title="Car Sales Prediction API",
    description="MLOps Auto-Retrain Project - Production API",
    version="1.0.0"
)

# Global variables
model = None
API_KEY = "MLOPS_PLATFORM_KEY_2026"

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Mount static files
app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
async def read_index():
    """Serve the web interface"""
    return FileResponse("api/static/index.html")

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    global model
    try:
        model = load_production_model()
        print(f"[OK] Production model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")

# ============================================================
# SCHEMAS
# ============================================================

# Strict schemas removed in favor of dynamic Dict[str, Any]

class PredictionResponse(BaseModel):
    """Output schema for prediction"""
    prediction: float
    model_version: str
    status: str
    timestamp: str
    processing_time_ms: float

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Car Sales Prediction API is running",
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    if model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy"}

@app.get("/schema")
async def get_schema():
    """Returns the current expected schema inferred from the training data"""
    from data_validation.validator import DataValidator
    validator = DataValidator()
    # Read the first row of training data to infer
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'car_sales.csv'
    if data_path.exists():
        df = pd.read_csv(data_path, nrows=5)
        is_valid, err, warn = validator.validate(df)
        # Convert everything to standard Python types for JSON serializability
        ranges_serializable = {}
        for col, r in validator.inferred_ranges.items():
            ranges_serializable[col] = {
                "min": float(r['min']) if pd.notnull(r['min']) else None,
                "max": float(r['max']) if pd.notnull(r['max']) else None
            }
            
        return {
            "required_columns": validator.inferred_required_columns,
            "target_column": validator.inferred_target,
            "dtypes": {k: str(v) for k, v in validator.inferred_dtypes.items()},
            "ranges": ranges_serializable
        }
    return {"error": "Training data not found to infer schema"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(
    data: Dict[str, Any], 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """
    Predict sales for a car.
    Includes integrated data validation.
    """
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # 1. Convert input to DataFrame and align with schema
    from data_validation.validator import DataValidator
    validator = DataValidator()
    # Read training data to get full schema for alignment
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'car_sales.csv'
    if data_path.exists():
        ref_df = pd.read_csv(data_path, nrows=1)
        validator.validate(ref_df) # This populates the inferred fields
        expected_cols = validator.inferred_required_columns
    else:
        expected_cols = list(data.keys())

    input_dict = data
    for col in expected_cols:
        if col not in input_dict:
            input_dict[col] = np.nan
        # Clean existing values
        val = input_dict[col]
        if val is None or val == "" or val == "None" or val == "NaN":
            input_dict[col] = np.nan
            
    # Ensure correct order for the model
    df = pd.DataFrame([input_dict])
    if data_path.exists():
        # Only keep columns the model expects (dropping target if present)
        cols_to_use = [c for c in expected_cols if c != validator.inferred_target]
        df = df[cols_to_use]
    
    # 2. Validate Data
    # We pass is_prediction=True to bypass checks for target column and dataset size
    is_valid, validated_df, errors, warnings = validate_data(df, is_prediction=True)
    
    if not is_valid:
        raise HTTPException(
            status_code=400, 
            detail={
                "message": "Data validation failed",
                "errors": errors
            }
        )
    
    # 3. Generate Prediction
    try:
        # The pipeline handles preprocessing (imputation, encoding, scaling)
        prediction = model.predict(validated_df)[0]
        
        # Ensure prediction is not negative (business logic)
        prediction = max(0, float(prediction))
        
        processing_time = (time.time() - start_time) * 1000
        
        # 4. Log Prediction (Background Task)
        background_tasks.add_task(
            log_prediction, 
            input_data=input_dict, 
            prediction=prediction, 
            model_version="v1.0"
        )
        
        return PredictionResponse(
            prediction=round(prediction, 4),
            model_version="v1.0",
            status="success",
            timestamp=pd.Timestamp.now().isoformat(),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        import traceback
        error_msg = f"Prediction error: {str(e)}\n{traceback.format_exc()}"
        with open('logs/api_debug.log', 'a') as f:
            f.write("\n" + "="*50 + "\n")
            f.write(error_msg + "\n")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/observability/health")
async def get_system_health():
    """Get the latest system health report"""
    report_path = Path(__file__).parent.parent / "reports" / "system_health.json"
    if not report_path.exists():
        return {"status": "error", "message": "Health report not found"}
    import json
    with open(report_path, 'r') as f:
        return json.load(f)

@app.get("/observability/drift")
async def get_drift_summary():
    """Get the latest drift summary"""
    report_path = Path(__file__).parent.parent / "reports" / "drift_summary.json"
    if not report_path.exists():
        return {"status": "error", "message": "Drift report not found"}
    import json
    with open(report_path, 'r') as f:
        return json.load(f)

@app.get("/observability/history")
async def get_retraining_history():
    """Get the retraining history"""
    history_path = Path(__file__).parent.parent / "logs" / "retraining_history.json"
    if not history_path.exists():
        return []
    import json
    with open(history_path, 'r') as f:
        return json.load(f)

@app.post("/data/upload")
async def upload_data(
    file_data: List[Dict[str, Any]], 
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Batch upload to the prediction log."""
    for item in file_data:
        background_tasks.add_task(log_prediction, input_data=item, prediction=None, model_version="manual_upload")
    return {"status": "success", "message": f"Queued {len(file_data)} records"}

@app.post("/predict_batch")
async def predict_batch(
    data: List[Dict[str, Any]],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_api_key)
):
    """Batch prediction endpoint"""
    results = []
    for item in data:
        try:
            res = await predict(item, background_tasks, api_key)
            results.append(res)
        except HTTPException as e:
            results.append({"status": "error", "detail": e.detail})
    return results
