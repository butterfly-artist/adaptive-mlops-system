"""
Failure Simulation Script
Proves the MLOps system can fail safely under various stress conditions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import subprocess
import sys
import shutil
import os

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from training.retraining_pipeline import run_retraining_pipeline

def setup_clean_state():
    """Reset logs and history for a clean test."""
    log_dir = Path("logs")
    if log_dir.exists():
        for f in log_dir.glob("*"):
            if f.name != ".gitignore":
                f.unlink()
    
    # Ensure reports dir exists
    Path("reports").mkdir(exist_ok=True)
    
    # Ensure production model exists (v1.0)
    prod_dir = Path("production_models/v1.0")
    if not prod_dir.exists():
        print("Training initial model v1.0...")
        subprocess.run([sys.executable, "train_model_v1_mlflow.py"], capture_output=True)

def run_scenario(name, setup_func):
    print(f"\n" + "!"*80)
    print(f"SCENARIO: {name}")
    print("!"*80)
    
    setup_func()
    
    try:
        triggered = run_retraining_pipeline()
        print(f"\nPipeline finished. Retraining triggered: {triggered}")
    except Exception as e:
        print(f"\nPipeline caught exception: {e}")
        
    # Check history
    history_path = Path("logs/retraining_history.json")
    if history_path.exists():
        with open(history_path, 'r') as f:
            history = json.load(f)
            last_decision = history[-1]["decision"]
            print(f"Last Decision: {last_decision.get('retraining_status', 'N/A')}")
            print(f"Reason: {last_decision.get('reason', 'N/A')}")
            if "promotion_status" in last_decision:
                print(f"Promotion: {last_decision['promotion_status']}")
            if "error" in last_decision:
                print(f"Error Logged: {last_decision['error'][:200]}...")

# --- Setup Functions ---

def setup_empty_data():
    print("Setting up: Empty new data...")
    log_path = Path("logs/predictions.csv")
    if log_path.exists(): log_path.unlink()

def setup_missing_features():
    print("Setting up: Missing required features in logs...")
    df = pd.DataFrame({'Manufacturer': ['Toyota'], 'prediction': [20.0]}) # Missing almost everything
    df.to_csv("logs/predictions.csv", index=False)

def setup_bad_data_validation():
    print("Setting up: Data that fails validation (Price = 99999)...")
    df = pd.read_csv("data/raw/car_sales.csv").head(60).copy()
    df['Price_in_thousands'] = 99999.0 # Out of range
    df['prediction'] = 20.0
    df['model_version'] = 'v1.0'
    df['timestamp'] = pd.Timestamp.now().isoformat()
    df.to_csv("logs/predictions.csv", index=False)
    # Clear history to bypass cooldown
    history_path = Path("logs/retraining_history.json")
    if history_path.exists(): history_path.unlink()

def setup_worse_model():
    print("Setting up: Data that produces a worse model...")
    df = pd.read_csv("data/raw/car_sales.csv").head(100).copy()
    # Add random noise to features to make them useless
    for col in df.select_dtypes(include=[np.number]).columns:
        if col != 'Sales_in_thousands':
            df[col] = np.random.normal(0, 1000, len(df))
    
    df['prediction'] = 20.0
    df['model_version'] = 'v1.0'
    df['timestamp'] = pd.Timestamp.now().isoformat()
    df.to_csv("logs/predictions.csv", index=False)
    # Clear history to bypass cooldown
    history_path = Path("logs/retraining_history.json")
    if history_path.exists(): history_path.unlink()

if __name__ == "__main__":
    setup_clean_state()
    
    run_scenario("Empty New Data", setup_empty_data)
    run_scenario("Missing Features", setup_missing_features)
    run_scenario("Validation Failure (Bad Data)", setup_bad_data_validation)
    run_scenario("Performance Degradation (Worse Model)", setup_worse_model)
