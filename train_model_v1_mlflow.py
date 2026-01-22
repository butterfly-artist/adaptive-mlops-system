"""
Model v1 Training with MLflow Logging

This script provides FULL MLOps traceability:
- Dataset version (DVC hash)
- Model type and parameters
- All metrics (RMSE, MAE, R², MAPE)
- Preprocessing configuration
- Model artifacts

WITHOUT this logging: It's just ML
WITH THIS LOGGING: IT'S MLOPS [OK]
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import json
import mlflow
import mlflow.sklearn
from datetime import datetime
from pathlib import Path
import sys
import os
import subprocess

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.insert(0, os.path.join(project_root, 'src'))

from data_validation import validate_data, TARGET_COLUMN
from training.full_pipeline import create_full_pipeline
from preprocessing.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATE_FEATURES,
    NUMERICAL_IMPUTATION_STRATEGY,
    CATEGORICAL_IMPUTATION_STRATEGY,
    SCALING_STRATEGY,
    CATEGORICAL_ENCODING
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION = "v1.0"
MODEL_NAME = "lasso"
EXPERIMENT_NAME = "car_sales_prediction"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15

MODEL_PARAMS = {
    'alpha': 1.0,
    'random_state': RANDOM_STATE
}

OUTPUT_DIR = Path(project_root) / 'production_models' / MODEL_VERSION
MLFLOW_TRACKING_URI = Path(project_root) / 'mlruns'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_dvc_hash(file_path):
    """Get DVC hash for data versioning"""
    dvc_file = f"{file_path}.dvc"
    
    if not Path(dvc_file).exists():
        return "no_dvc_tracking"
    
    try:
        with open(dvc_file, 'r') as f:
            import yaml
            dvc_data = yaml.safe_load(f)
            # Get MD5 hash from DVC metadata
            if 'outs' in dvc_data and len(dvc_data['outs']) > 0:
                return dvc_data['outs'][0].get('md5', 'unknown')
    except Exception as e:
        print(f"[WARNING] Could not read DVC hash: {e}")
        return "dvc_read_error"
    
    return "unknown"


def calculate_metrics(y_true, y_pred):
    """Calculate comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else np.nan
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


# ============================================================
# MLFLOW LOGGING FUNCTIONS
# ============================================================

def log_preprocessing_config():
    """Log all preprocessing configuration"""
    mlflow.log_param("num_features_count", len(NUMERICAL_FEATURES))
    mlflow.log_param("cat_features_count", len(CATEGORICAL_FEATURES))
    mlflow.log_param("date_features_count", len(DATE_FEATURES))
    
    mlflow.log_param("num_imputation", NUMERICAL_IMPUTATION_STRATEGY)
    mlflow.log_param("cat_imputation", CATEGORICAL_IMPUTATION_STRATEGY)
    mlflow.log_param("scaling_strategy", SCALING_STRATEGY)
    mlflow.log_param("cat_encoding", CATEGORICAL_ENCODING)
    
    # Log feature lists as JSON
    mlflow.log_dict({
        "numerical_features": NUMERICAL_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "date_features": DATE_FEATURES
    }, "preprocessing_features.json")


def log_dataset_info(df, data_hash, X_train, X_val, X_test):
    """Log dataset information and versioning"""
    # Dataset version
    mlflow.log_param("dataset_dvc_hash", data_hash)
    mlflow.log_param("dataset_total_rows", len(df))
    mlflow.log_param("dataset_features", df.shape[1])
    
    # Split information
    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("val_samples", len(X_val))
    mlflow.log_param("test_samples", len(X_test))
    mlflow.log_param("test_size", TEST_SIZE)
    mlflow.log_param("val_size", VAL_SIZE)


def log_model_info():
    """Log model configuration"""
    mlflow.log_param("model_type", "Lasso")
    mlflow.log_param("model_version", MODEL_VERSION)
    mlflow.log_param("model_name", MODEL_NAME)
    mlflow.log_param("random_state", RANDOM_STATE)
    
    # Log all model hyperparameters
    for param_name, param_value in MODEL_PARAMS.items():
        mlflow.log_param(f"model_{param_name}", param_value)


def log_metrics_all_sets(train_metrics, val_metrics, test_metrics):
    """Log metrics for all data splits"""
    # Train metrics
    for metric_name, metric_value in train_metrics.items():
        mlflow.log_metric(f"train_{metric_name}", metric_value)
    
    # Validation metrics
    for metric_name, metric_value in val_metrics.items():
        mlflow.log_metric(f"val_{metric_name}", metric_value)
    
    # Test metrics (most important)
    for metric_name, metric_value in test_metrics.items():
        mlflow.log_metric(f"test_{metric_name}", metric_value)
    
    # Log generalization gap
    train_test_gap = test_metrics['rmse'] - train_metrics['rmse']
    mlflow.log_metric("generalization_gap_rmse", train_test_gap)


def log_artifacts_to_mlflow(pipeline, predictions_df, metrics_all):
    """Log all artifacts to MLflow"""
    # Create temp directory for artifacts
    temp_dir = Path("temp_mlflow_artifacts")
    temp_dir.mkdir(exist_ok=True)
    
    try:
        # Save and log model
        mlflow.sklearn.log_model(pipeline, "model")
        
        # Log predictions
        pred_path = temp_dir / "predictions.csv"
        predictions_df.to_csv(pred_path, index=False)
        mlflow.log_artifact(pred_path, "predictions")
        
        # Log metrics JSON
        metrics_path = temp_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics_all, f, indent=2)
        mlflow.log_artifact(metrics_path)
        
        print("[OK] Artifacts logged to MLflow")
        
    finally:
        # Cleanup temp directory
        import shutil
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


# ============================================================
# MAIN TRAINING WITH MLFLOW
# ============================================================

def train_model_with_mlflow(custom_data_path=None, custom_output_dir=None):
    """Train model with full MLflow logging"""
    
    print("="*80)
    print(f"TRAINING MODEL {MODEL_VERSION} WITH MLFLOW")
    print("="*80)
    print("Logging: Dataset version, Model config, Metrics, Artifacts")
    print("This is MLOps, not just ML!")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(f"file:///{MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"{MODEL_NAME}_{MODEL_VERSION}") as run:
        
        print(f"\n[OK] MLflow Run ID: {run.info.run_id}")
        print(f"[OK] Experiment: {EXPERIMENT_NAME}")
        
        # --------------------------------------------------------
        # STEP 1: Load and Validate Data
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 1: LOAD AND VALIDATE DATA")
        print("="*80)
        
        data_path = Path(custom_data_path) if custom_data_path else Path(project_root) / 'data' / 'raw' / 'car_sales.csv'
        
        # Get DVC hash for data versioning
        data_hash = get_dvc_hash(str(data_path))
        print(f"Dataset DVC hash: {data_hash}")
        
        is_valid, df, errors, warnings = validate_data(str(data_path))
        
        if not is_valid:
            print("\n[ERROR] Data validation failed!")
            for error in errors:
                print(f"  - {error}")
            mlflow.log_param("validation_status", "failed")
            sys.exit(1)
        
        mlflow.log_param("validation_status", "passed")
        print(f"[OK] Data validated: {df.shape}")
        
        # --------------------------------------------------------
        # STEP 2: Split Data
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 2: SPLIT DATA")
        print("="*80)
        
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        
        val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=RANDOM_STATE
        )
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # --------------------------------------------------------
        # STEP 3: Log Configuration to MLflow
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 3: LOG CONFIGURATION TO MLFLOW")
        print("="*80)
        
        # Log dataset info
        log_dataset_info(df, data_hash, X_train, X_val, X_test)
        print("[OK] Dataset info logged")
        
        # Log model info
        log_model_info()
        print("[OK] Model config logged")
        
        # Log preprocessing config
        log_preprocessing_config()
        print("[OK] Preprocessing config logged")
        
        # Log additional tags
        mlflow.set_tag("version", MODEL_VERSION)
        mlflow.set_tag("model_type", "Lasso")
        mlflow.set_tag("created_by", "train_model_v1_mlflow.py")
        mlflow.set_tag("purpose", "first_production_model")
        
        # --------------------------------------------------------
        # STEP 4: Create and Fit Pipeline
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 4: CREATE AND FIT PIPELINE")
        print("="*80)
        
        model = Lasso(**MODEL_PARAMS)
        pipeline = create_full_pipeline(model, model_name=MODEL_NAME)
        
        print("Fitting pipeline...")
        pipeline.fit(X_train, y_train)
        print("[OK] Pipeline fitted")
        
        # --------------------------------------------------------
        # STEP 5: Generate Predictions
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 5: GENERATE PREDICTIONS")
        print("="*80)
        
        y_train_pred = pipeline.predict(X_train)
        y_val_pred = pipeline.predict(X_val)
        y_test_pred = pipeline.predict(X_test)
        print("[OK] Predictions generated")
        
        # --------------------------------------------------------
        # STEP 6: Calculate and Log Metrics
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 6: CALCULATE AND LOG METRICS")
        print("="*80)
        
        train_metrics = calculate_metrics(y_train, y_train_pred)
        val_metrics = calculate_metrics(y_val, y_val_pred)
        test_metrics = calculate_metrics(y_test, y_test_pred)
        
        print(f"\nTest RMSE: {test_metrics['rmse']:.4f}")
        print(f"Test MAE:  {test_metrics['mae']:.4f}")
        print(f"Test R²:   {test_metrics['r2']:.6f}")
        
        # Log to MLflow
        log_metrics_all_sets(train_metrics, val_metrics, test_metrics)
        print("[OK] Metrics logged to MLflow")
        
        # --------------------------------------------------------
        # STEP 7: Compare with Baseline
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 7: COMPARE WITH BASELINE")
        print("="*80)
        
        baseline_path = Path(project_root) / 'baseline_lower_bounds.json'
        if baseline_path.exists():
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_rmse = baseline_data['baseline_lower_bounds']['min_rmse_to_beat']
            improvement = ((baseline_rmse - test_metrics['rmse']) / baseline_rmse * 100)
            
            mlflow.log_metric("baseline_rmse", baseline_rmse)
            mlflow.log_metric("improvement_vs_baseline_pct", improvement)
            
            print(f"Baseline RMSE: {baseline_rmse:.4f}")
            print(f"Model RMSE:    {test_metrics['rmse']:.4f}")
            print(f"Improvement:   {improvement:.1f}%")
        
        # --------------------------------------------------------
        # STEP 8: Log Artifacts
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 8: LOG ARTIFACTS TO MLFLOW")
        print("="*80)
        
        # Create predictions dataframe
        predictions_df = pd.DataFrame({
            'actual': y_test.values,
            'predicted': y_test_pred,
            'residual': y_test.values - y_test_pred,
            'abs_error': np.abs(y_test.values - y_test_pred)
        })
        
        metrics_all = {
            'model_version': MODEL_VERSION,
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        log_artifacts_to_mlflow(pipeline, predictions_df, metrics_all)
        
        # --------------------------------------------------------
        # STEP 9: Save Locally (for backup)
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("STEP 9: SAVE LOCAL BACKUP")
        print("="*80)
        
        final_output_dir = Path(custom_output_dir) if custom_output_dir else OUTPUT_DIR
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline
        pipeline_path = final_output_dir / f'{MODEL_NAME}_pipeline.pkl'
        joblib.dump(pipeline, pipeline_path)
        print(f"[OK] Pipeline saved: {pipeline_path}")
        
        # Save metrics locally for the deployment gate
        metrics_path = final_output_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics_all, f, indent=2)
        
        # Save MLflow run info
        mlflow_info = {
            'run_id': run.info.run_id,
            'experiment_id': run.info.experiment_id,
            'experiment_name': EXPERIMENT_NAME,
            'run_name': f"{MODEL_NAME}_{MODEL_VERSION}",
            'tracking_uri': str(MLFLOW_TRACKING_URI)
        }
        
        with open(final_output_dir / 'mlflow_run_info.json', 'w') as f:
            json.dump(mlflow_info, f, indent=2)
        print(f"[OK] MLflow info saved")
        
        # --------------------------------------------------------
        # SUMMARY
        # --------------------------------------------------------
        print("\n" + "="*80)
        print("MLFLOW LOGGING COMPLETE")
        print("="*80)
        
        print(f"\n[OK] Experiment: {EXPERIMENT_NAME}")
        print(f"[OK] Run ID: {run.info.run_id}")
        print(f"[OK] Dataset hash: {data_hash}")
        print(f"[OK] Model: Lasso (alpha={MODEL_PARAMS['alpha']})")
        print(f"[OK] Test RMSE: {test_metrics['rmse']:.4f}")
        
        print(f"\nLogged to MLflow:")
        print(f"  - Dataset version (DVC hash)")
        print(f"  - Model type and parameters")
        print(f"  - All metrics (train/val/test)")
        print(f"  - Preprocessing configuration")
        print(f"  - Model pipeline artifact")
        print(f"  - Predictions")
        
        print(f"\nView in MLflow UI:")
        print(f"  mlflow ui --backend-store-uri file:///{MLFLOW_TRACKING_URI}")
        
        return pipeline, metrics_all, run.info.run_id


if __name__ == "__main__":
    print("\n" + "*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 15 + "MODEL v1 TRAINING WITH MLFLOW LOGGING" + " " * 25 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    print("\nWithout MLflow: It's just ML")
    print("With MLflow: It's MLOps [OK]\n")
    
    custom_data = sys.argv[1] if len(sys.argv) > 1 else None
    custom_out = sys.argv[2] if len(sys.argv) > 2 else None
    pipeline, metrics, run_id = train_model_with_mlflow(custom_data, custom_out)
    
    print("\n" + "*" * 80)
    print("TRAINING COMPLETE - FULLY LOGGED")
    print("*" * 80)
    print(f"\nMLflow Run ID: {run_id}")
    print("\nYou now have:")
    print("  [OK] Traceability (dataset version tracked)")
    print("  [OK] Auditability (all params and metrics logged)")
    print("  [OK] Rollback ability (can load any previous run)")
    print("\nThis is MLOps! [OK]\n")
