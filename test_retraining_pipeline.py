"""
Test Automated Retraining Pipeline
Simulates drift and verifies the pipeline triggers retraining.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from training.retraining_pipeline import run_retraining_pipeline

def simulate_drifted_logs():
    print("Simulating drifted logs...")
    # Load original data
    df = pd.read_csv("data/raw/car_sales.csv")
    
    # Create a drifted version (subtle but significant for KS test)
    drifted_df = df.copy()
    drifted_df['Price_in_thousands'] = drifted_df['Price_in_thousands'] * 1.2
    drifted_df['Horsepower'] = drifted_df['Horsepower'] * 1.1
    drifted_df['Engine_size'] = drifted_df['Engine_size'] * 1.1
    drifted_df['Wheelbase'] = drifted_df['Wheelbase'] + 5
    drifted_df['Width'] = drifted_df['Width'] + 2
    
    # Add metadata columns required by the logger
    # Make predictions very different to trigger prediction drift
    drifted_df['prediction'] = drifted_df['Sales_in_thousands'] * 5.0 
    drifted_df['model_version'] = 'v1.0'
    drifted_df['timestamp'] = pd.Timestamp.now().isoformat()
    
    # Save to logs/predictions.csv
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    drifted_df.to_csv(log_dir / "predictions.csv", index=False)
    
    # Clear history to bypass cooldown for testing
    history_path = Path("logs/retraining_history.json")
    if history_path.exists():
        history_path.unlink()
        
    print(f"[OK] Drifted logs created and history cleared.")

def test_pipeline():
    print("\n" + "="*80)
    print("TESTING AUTOMATED RETRAINING PIPELINE")
    print("="*80)
    
    # 1. Simulate drift
    simulate_drifted_logs()
    
    # 2. Run pipeline
    print("\nRunning pipeline...")
    triggered = run_retraining_pipeline()
    
    if triggered:
        print("\n[OK] SUCCESS: Retraining was triggered as expected!")
    else:
        print("\n[FAIL] FAILURE: Retraining was NOT triggered.")
        
    # 3. Check history
    history_path = Path("logs/retraining_history.json")
    if history_path.exists():
        print(f"\n[OK] Retraining history found at {history_path}")
    else:
        print("\n[FAIL] Retraining history NOT found.")

if __name__ == "__main__":
    test_pipeline()
