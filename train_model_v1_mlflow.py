
"""
Zero-Touch Model Training Script
Automatically handles any dataset by using AutoPreprocessor and dynamic DataValidator.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import sys
import os

# Add src to path
project_root = Path(__file__).parent.resolve()
sys.path.insert(0, str(project_root / "src"))

from data_validation.validator import DataValidator
from preprocessing.auto_preprocessor import AutoPreprocessor

def train_autonomous_model(data_path, output_dir):
    """Refactored training script for Zero-Touch MLOps"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Zero_Touch_MLOps")
    
    # 1. Load and Validate
    validator = DataValidator()
    df = pd.read_csv(data_path)
    
    # We must run a basic validation to populate inferred fields
    # But wait, validate needs inferred_target to work properly if we want it completely autonomous
    # So we call _infer_schema_from_df directly first
    validator._infer_schema_from_df(df)
    
    target = validator.inferred_target
    if not target:
        print("Could not identify target column. Aborting.")
        return
        
    # Drop rows where target is NaN before processing
    original_len = len(df)
    df = df.dropna(subset=[target])
    if len(df) < original_len:
        print(f"Dropped {original_len - len(df)} rows with missing target.")

    is_valid, errors, warn = validator.validate(df)
        
    print(f"Target identified: {target}")
    
    # 2. Preprocess
    preprocessor = AutoPreprocessor()
    preprocessor.fit(df, target)
    
    X = df.drop(columns=[target])
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Train
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor.preprocessor),
        ('model', Ridge())
    ])
    
    with mlflow.start_run(run_name=f"Retrain_{Path(data_path).name}"):
        print(f"Fitting model on {len(X_train)} samples...")
        model_pipeline.fit(X_train, y_train)
        
        y_pred = model_pipeline.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_param("dataset", Path(data_path).name)
        mlflow.log_param("num_features", len(X.columns))
        
        # Save artifacts
        # We save as 'lasso_pipeline.pkl' to maintain compatibility with the existing loader
        # even if the model is Ridge. Loader just looks for *_pipeline.pkl.
        model_name = "lasso_pipeline.pkl" 
        joblib.dump(model_pipeline, output_path / model_name)
        
        mlflow.sklearn.log_model(model_pipeline, "model")
        print(f"✓ Training complete. RMSE: {rmse:.4f}, R2: {r2:.4f}")
        print(f"✓ Model saved to {output_path / model_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("data_path", nargs="?", default="data/raw/car_sales.csv")
    parser.add_argument("output_dir", nargs="?", default="production_models/v1.0")
    args = parser.parse_args()
    
    train_autonomous_model(args.data_path, args.output_dir)
