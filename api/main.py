"""
Car Sales Prediction API
FastAPI implementation with integrated data validation
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
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
    df = pd.DataFrame([input_dict])
    
    # 2. Validate Data
    # Note: validate_data expects a target column or handles missing ones
    # We pass it to our validator which checks schema, ranges, and types
    is_valid, validated_df, errors, warnings = validate_data(df)
    
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
