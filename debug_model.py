
import pandas as pd
import joblib
from pathlib import Path
import sys
import os

# Add src to path
project_root = Path("d:/projects/datamodeMLOPS/mlops-auto-retrain")
sys.path.insert(0, str(project_root / "src"))

from deployment.model_loader import load_production_model

def debug_prediction():
    try:
        model = load_production_model()
        version = "v1.0" # Hardcoded for debug
        print(f"Model version: {version}")
        
        # Load one row from raw data
        raw_data_path = project_root / "data" / "raw" / "car_sales.csv"
        df = pd.read_csv(raw_data_path, nrows=1)
        target = "Sales_in_thousands"
        X = df.drop(columns=[target])
        
        print(f"X columns ({len(X.columns)}): {X.columns.tolist()}")
        
        # Try prediction
        pred = model.predict(X)
        print(f"Prediction successful: {pred}")
        
        # Now try with missing column
        X_missing = X.drop(columns=['__year_resale_value'])
        print(f"X_missing columns ({len(X_missing.columns)}): {X_missing.columns.tolist()}")
        try:
            model.predict(X_missing)
        except Exception as e:
            print(f"Caught expected error with missing col: {e}")
            
        # Now try with NaN filled column
        X_nan = X.copy()
        X_nan['__year_resale_value'] = float('nan')
        pred_nan = model.predict(X_nan)
        print(f"Prediction with NaN successful: {pred_nan}")

    except Exception as e:
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_prediction()
