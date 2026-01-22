"""
Train Model v1 - First Production Model

Goal: Working, logged, reproducible model
Rules:
- NO hyperparameter tuning
- NO trying multiple models
- NO optimization
- Focus: Reproducibility and proper logging
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = current_dir
sys.path.insert(0, os.path.join(project_root, 'src'))

from data_validation import validate_data, TARGET_COLUMN
from training.full_pipeline import create_full_pipeline


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_VERSION = "v1.0"
MODEL_NAME = "lasso"
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.15  # 15% of train for validation

# Model hyperparameters (simple, not tuned)
MODEL_PARAMS = {
    'alpha': 1.0,
    'random_state': RANDOM_STATE
}

# Output paths
OUTPUT_DIR = Path(project_root) / 'production_models' / MODEL_VERSION
MODEL_PATH = OUTPUT_DIR / f'{MODEL_NAME}_pipeline.pkl'
METRICS_PATH = OUTPUT_DIR / 'metrics.json'
PREDICTIONS_PATH = OUTPUT_DIR / 'predictions.csv'
METADATA_PATH = OUTPUT_DIR / 'metadata.json'


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_metrics(y_true, y_pred, set_name=''):
    """Calculate comprehensive metrics"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - handle potential division by zero
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        # Manual calculation if needed
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if (y_true != 0).all() else np.nan
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape),
        'n_samples': len(y_true)
    }
    
    if set_name:
        print(f"\n{set_name} Metrics:")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.6f}")
        print(f"  MAPE: {mape:.2f}%")
        print(f"  Samples: {len(y_true)}")
    
    return metrics


def create_metadata():
    """Create metadata for model tracking"""
    return {
        'model_version': MODEL_VERSION,
        'model_name': MODEL_NAME,
        'model_type': 'Lasso',
        'model_params': MODEL_PARAMS,
        'created_at': datetime.now().isoformat(),
        'random_state': RANDOM_STATE,
        'test_size': TEST_SIZE,
        'val_size': VAL_SIZE,
        'target_column': TARGET_COLUMN,
        'description': 'First production model - no tuning, baseline performance'
    }


def save_predictions(y_true, y_pred, set_name, output_path):
    """Save predictions for analysis"""
    pred_df = pd.DataFrame({
        'actual': y_true.values,
        'predicted': y_pred,
        'residual': y_true.values - y_pred,
        'abs_error': np.abs(y_true.values - y_pred),
        'pct_error': np.abs((y_true.values - y_pred) / y_true.values) * 100
    })
    
    pred_df.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved: {output_path}")


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_model_v1():
    """Train the first production model"""
    
    print("="*80)
    print(f"TRAINING MODEL {MODEL_VERSION}")
    print("="*80)
    print(f"Goal: Working, logged, reproducible baseline model")
    print(f"Model: {MODEL_NAME.upper()}")
    print(f"No tuning, no optimization - just solid baseline")
    
    # --------------------------------------------------------
    # STEP 1: Load and Validate Data
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 1: LOAD AND VALIDATE DATA")
    print("="*80)
    
    data_path = Path(project_root) / 'data' / 'raw' / 'car_sales.csv'
    
    is_valid, df, errors, warnings = validate_data(str(data_path))
    
    if not is_valid:
        print("\n[ERROR] Data validation failed!")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"✓ Data validated: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"  Warnings: {len(warnings)}")
    
    # --------------------------------------------------------
    # STEP 2: Split Data (Train / Validation / Test)
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 2: SPLIT DATA")
    print("="*80)
    
    # Prepare features and target
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    print(f"Target: {TARGET_COLUMN}")
    print(f"Features: {X.shape[1]}")
    
    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE
    )
    
    # Second split: train and validation from remaining data
    val_size_adjusted = VAL_SIZE / (1 - TEST_SIZE)  # Adjust for already separated test
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        random_state=RANDOM_STATE
    )
    
    print(f"\nSplit summary:")
    print(f"  Train:      {len(X_train):3d} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Validation: {len(X_val):3d} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test:       {len(X_test):3d} samples ({len(X_test)/len(X)*100:.1f}%)")
    print(f"  Total:      {len(X):3d} samples")
    
    # --------------------------------------------------------
    # STEP 3: Create and Fit Pipeline
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 3: CREATE AND FIT PIPELINE")
    print("="*80)
    
    # Create model
    model = Lasso(**MODEL_PARAMS)
    print(f"Model: Lasso(alpha={MODEL_PARAMS['alpha']})")
    
    # Create full pipeline (preprocessing + model)
    pipeline = create_full_pipeline(model, model_name=MODEL_NAME)
    
    print("\nPipeline structure:")
    for i, (step_name, step) in enumerate(pipeline.steps, 1):
        print(f"  {i}. {step_name}: {type(step).__name__}")
    
    # Fit on training data
    print(f"\nFitting pipeline on {len(X_train)} training samples...")
    pipeline.fit(X_train, y_train)
    print("✓ Pipeline fitted successfully")
    
    # --------------------------------------------------------
    # STEP 4: Generate Predictions
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 4: GENERATE PREDICTIONS")
    print("="*80)
    
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)
    
    print("✓ Predictions generated for all sets")
    
    # --------------------------------------------------------
    # STEP 5: Calculate Metrics
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 5: CALCULATE METRICS")
    print("="*80)
    
    train_metrics = calculate_metrics(y_train, y_train_pred, "TRAIN")
    val_metrics = calculate_metrics(y_val, y_val_pred, "VALIDATION")
    test_metrics = calculate_metrics(y_test, y_test_pred, "TEST")
    
    # --------------------------------------------------------
    # STEP 6: Check Against Baseline
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 6: COMPARE WITH BASELINE")
    print("="*80)
    
    baseline_path = Path(project_root) / 'baseline_lower_bounds.json'
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_data = json.load(f)
        
        baseline_rmse = baseline_data['baseline_lower_bounds']['min_rmse_to_beat']
        baseline_mae = baseline_data['baseline_lower_bounds']['min_mae_to_beat']
        
        print(f"Baseline RMSE: {baseline_rmse:.4f}")
        print(f"Model v1 RMSE: {test_metrics['rmse']:.4f}")
        
        improvement = ((baseline_rmse - test_metrics['rmse']) / baseline_rmse * 100)
        
        if test_metrics['rmse'] < baseline_rmse:
            print(f"✓ Model BEATS baseline by {improvement:.1f}%")
        else:
            print(f"✗ Model WORSE than baseline by {-improvement:.1f}%")
    
    # --------------------------------------------------------
    # STEP 7: Save Everything
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("STEP 7: SAVE MODEL AND ARTIFACTS")
    print("="*80)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Save pipeline
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✓ Pipeline saved: {MODEL_PATH}")
    print(f"  Size: {MODEL_PATH.stat().st_size / 1024:.2f} KB")
    
    # Save metrics
    all_metrics = {
        'model_version': MODEL_VERSION,
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics
    }
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"✓ Metrics saved: {METRICS_PATH}")
    
    # Save metadata
    metadata = create_metadata()
    metadata['data_info'] = {
        'n_total': len(X),
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features_input': X.shape[1],
        'target_column': TARGET_COLUMN
    }
    
    with open(METADATA_PATH, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Metadata saved: {METADATA_PATH}")
    
    # Save predictions
    save_predictions(
        y_test, y_test_pred, 
        'test', 
        PREDICTIONS_PATH
    )
    
    # --------------------------------------------------------
    # STEP 8: Summary
    # --------------------------------------------------------
    print("\n" + "="*80)
    print("MODEL v1 TRAINING COMPLETE")
    print("="*80)
    
    print(f"\n{'='*80}")
    print("FINAL PERFORMANCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nModel: {MODEL_NAME.upper()} {MODEL_VERSION}")
    print(f"Created: {metadata['created_at']}")
    
    print(f"\nTest Set Performance:")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE:  {test_metrics['mae']:.4f}")
    print(f"  R²:   {test_metrics['r2']:.6f}")
    print(f"  MAPE: {test_metrics['mape']:.2f}%")
    
    print(f"\nValidation Set Performance:")
    print(f"  RMSE: {val_metrics['rmse']:.4f}")
    print(f"  MAE:  {val_metrics['mae']:.4f}")
    print(f"  R²:   {val_metrics['r2']:.6f}")
    
    print(f"\nGeneralization Check:")
    train_test_gap = test_metrics['rmse'] - train_metrics['rmse']
    print(f"  Train RMSE:  {train_metrics['rmse']:.4f}")
    print(f"  Test RMSE:   {test_metrics['rmse']:.4f}")
    print(f"  Gap:         {train_test_gap:.4f}")
    
    if train_test_gap < 50:
        print(f"  Status: ✓ Good generalization")
    else:
        print(f"  Status: ⚠ Possible overfitting")
    
    print(f"\nArtifacts:")
    print(f"  Pipeline:    {MODEL_PATH}")
    print(f"  Metrics:     {METRICS_PATH}")
    print(f"  Metadata:    {METADATA_PATH}")
    print(f"  Predictions: {PREDICTIONS_PATH}")
    
    print(f"\n{'='*80}")
    print("✓ MODEL v1 IS PRODUCTION-READY")
    print(f"{'='*80}")
    
    return pipeline, all_metrics, metadata


if __name__ == "__main__":
    """
    Train Model v1 - First production model
    """
    print("\n")
    print("*" * 80)
    print("*" + " " * 78 + "*")
    print("*" + " " * 20 + "MODEL v1 TRAINING SCRIPT" + " " * 34 + "*")
    print("*" + " " * 78 + "*")
    print("*" * 80)
    
    pipeline, metrics, metadata = train_model_v1()
    
    print("\n" + "*" * 80)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("*" * 80)
    print("\nModel v1 is ready for deployment!")
    print(f"Load with: joblib.load('{MODEL_PATH}')")
    print(f"Use with:  pipeline.predict(X_new)")
    print("\n")
