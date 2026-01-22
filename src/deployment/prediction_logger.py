"""
Prediction Logger Utility
Persists every prediction to logs/predictions.csv for drift detection and auditability.
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import csv

def log_prediction(input_data: dict, prediction: float, model_version: str):
    """
    Log input features, prediction, timestamp, and model version to a CSV file.
    """
    log_dir = Path(__file__).parent.parent.parent / "logs"
    log_file = log_dir / "predictions.csv"
    
    # Ensure logs directory exists
    log_dir.mkdir(exist_ok=True)
    
    # Prepare the log entry
    log_entry = input_data.copy()
    log_entry['prediction'] = prediction
    log_entry['model_version'] = model_version
    log_entry['timestamp'] = datetime.now().isoformat()
    
    # Check if file exists to write header
    file_exists = log_file.exists()
    
    try:
        # Use pandas for consistent CSV handling
        df_entry = pd.DataFrame([log_entry])
        
        # Append to CSV
        df_entry.to_csv(
            log_file, 
            mode='a', 
            header=not file_exists, 
            index=False
        )
    except Exception as e:
        # In a real system, we might log this error to a separate system
        print(f"[ERROR] Failed to log prediction: {e}")
