"""
Car Sales Prediction API
FastAPI implementation with integrated data validation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
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

# Global model variable
model = None

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
        print("[OK] Production model loaded successfully")
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        # In production, you might want to exit or retry
        pass

# ============================================================
# SCHEMAS
# ============================================================

class CarData(BaseModel):
    """Input schema for a single car prediction"""
    Manufacturer: str = Field(..., example="Acura")
    Model: str = Field(..., example="Integra")
    Vehicle_type: str = Field(..., example="Passenger")
    Price_in_thousands: float = Field(..., example=21.5)
    Engine_size: float = Field(..., example=1.8)
    Horsepower: int = Field(..., example=140)
    Wheelbase: float = Field(..., example=101.2)
    Width: float = Field(..., example=67.3)
    Length: float = Field(..., example=172.4)
    Curb_weight: float = Field(..., example=2.639)
    Fuel_capacity: float = Field(..., example=13.2)
    Fuel_efficiency: float = Field(..., example=28.0)
    Latest_Launch: str = Field(..., example="2/2/2012")
    Power_perf_factor: float = Field(..., example=58.28)
    # Optional fields that might be missing in raw data
    # Pydantic v2 uses alias for fields starting with __
    year_resale_value: Optional[float] = Field(None, alias="__year_resale_value", example=16.36)

    model_config = {
        "populate_by_name": True
    }

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

@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CarData, background_tasks: BackgroundTasks):
    """
    Predict sales for a car.
    Includes integrated data validation.
    """
    start_time = time.time()
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    
    # 1. Convert input to DataFrame
    # Use model_dump(by_alias=True) to get the original column names (like __year_resale_value)
    input_dict = data.model_dump(by_alias=True)
    
    # Clean optional fields: convert None or "None" strings to NaN
    for key, value in input_dict.items():
        if value is None or value == "None" or value == "NaN":
            input_dict[key] = np.nan
            
    df = pd.DataFrame([input_dict])
    
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
async def upload_csv(background_tasks: BackgroundTasks, file: List[CarData]):
    """
    Experimental: Batch upload cars to the prediction log.
    In a real app, this would handle a CSV file. 
    For now, we accept a list of CarData objects (validated by Pydantic).
    """
    for item in file:
        input_dict = item.model_dump(by_alias=True)
        # We don't predict here, just log them as new 'potential' samples
        background_tasks.add_task(log_prediction, input_data=input_dict, prediction=None, model_version="manual_upload")
    return {"status": "success", "message": f"Queued {len(file)} records for logging"}

@app.post("/predict_batch")
async def predict_batch(data: List[CarData]):
    """Batch prediction endpoint"""
    # Simplified batch implementation
    results = []
    for item in data:
        try:
            res = await predict(item, background_tasks)
            results.append(res)
        except HTTPException as e:
            results.append({"status": "error", "detail": e.detail})
    return results
