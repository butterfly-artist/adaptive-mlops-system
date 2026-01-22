"""
Full Training Pipeline
Combines preprocessing (ColumnTransformer) with ML models into single Pipeline

This ensures:
- Reproducibility: Same preprocessing always applied
- No data leakage: Preprocessing fitted on train only
- Safe deployment: Preprocessing + model travel together
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import sys
import os

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.join(project_root, 'src'))

from preprocessing import create_preprocessing_pipeline


def create_full_pipeline(model, model_name='model', preprocessing_params=None):
    """
    Create full pipeline: Preprocessing + Model
    
    Args:
        model: sklearn estimator (LinearRegression, RandomForest, etc.)
        model_name: Name for the model step
        preprocessing_params: Parameters for preprocessing pipeline
        
    Returns:
        sklearn Pipeline combining preprocessing and model
    """
    # Create preprocessing pipeline
    if preprocessing_params is None:
        preprocessing_params = {}
    
    preprocessing = create_preprocessing_pipeline(**preprocessing_params)
    
    # Combine preprocessing + model
    full_pipeline = Pipeline([
        ('preprocessing', preprocessing),
        (model_name, model)
    ])
    
    return full_pipeline


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2)
    }


def train_and_evaluate_model(
    pipeline,
    X_train, y_train,
    X_test, y_test,
    model_name='model'
) -> Dict[str, Any]:
    """
    Train and evaluate a full pipeline (preprocessing + model)
    
    Args:
        pipeline: Full sklearn Pipeline
        X_train, y_train: Training data
        X_test, y_test: Test data
        model_name: Name of the model
        
    Returns:
        Dictionary with results
    """
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name}")
    print(f"{'='*70}")
    
    # Fit entire pipeline (preprocessing + model together)
    print(f"Fitting pipeline on {len(X_train)} samples...")
    pipeline.fit(X_train, y_train)
    
    # Predict on train set
    y_train_pred = pipeline.predict(X_train)
    train_metrics = calculate_metrics(y_train, y_train_pred)
    
    # Predict on test set
    y_test_pred = pipeline.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Display results
    print(f"\nTrain Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print(f"\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    return {
        'model_name': model_name,
        'pipeline': pipeline,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': y_train_pred,
            'test': y_test_pred
        }
    }


def get_ml_models() -> Dict[str, Any]:
    """
    Get dictionary of ML models to train
    
    Returns:
        Dictionary {model_name: sklearn_estimator}
    """
    return {
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42),
        'lasso': Lasso(alpha=1.0, random_state=42),
        'random_forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
    }


def train_all_models(X_train, y_train, X_test, y_test) -> Dict[str, Any]:
    """
    Train all ML models with full pipelines
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        
    Returns:
        Dictionary with all results
    """
    print("="*70)
    print("TRAINING ALL MODELS")
    print("="*70)
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    models = get_ml_models()
    results = {}
    
    for model_name, model in models.items():
        # Create full pipeline
        pipeline = create_full_pipeline(model, model_name=model_name)
        
        # Train and evaluate
        result = train_and_evaluate_model(
            pipeline, X_train, y_train, X_test, y_test, model_name
        )
        
        results[model_name] = result
    
    return results


def compare_with_baselines(results: Dict[str, Any], baseline_path='baseline_lower_bounds.json'):
    """
    Compare ML model results with baseline models
    
    Args:
        results: Results from train_all_models
        baseline_path: Path to baseline results
    """
    print("\n" + "="*70)
    print("COMPARISON WITH BASELINES")
    print("="*70)
    
    # Load baseline bounds
    baseline_file = Path(baseline_path)
    if not baseline_file.exists():
        print(f"[WARNING] Baseline file not found: {baseline_path}")
        return
    
    with open(baseline_file, 'r') as f:
        baseline_data = json.load(f)
    
    baseline_bounds = baseline_data['baseline_lower_bounds']
    
    print(f"\nBaseline Lower Bounds:")
    print(f"  RMSE: {baseline_bounds['min_rmse_to_beat']:.4f}")
    print(f"  MAE: {baseline_bounds['min_mae_to_beat']:.4f}")
    print(f"  R²: {baseline_bounds['min_r2_to_beat']:.6f}")
    
    print(f"\n{'Model':<25} {'Test RMSE':<12} {'vs Baseline':<15} {'Test R²':<12}")
    print("-" * 70)
    
    baseline_rmse = baseline_bounds['min_rmse_to_beat']
    baseline_r2 = baseline_bounds['min_r2_to_beat']
    
    for model_name, result in results.items():
        test_rmse = result['test_metrics']['rmse']
        test_r2 = result['test_metrics']['r2']
        
        improvement = ((baseline_rmse - test_rmse) / baseline_rmse * 100)
        beats_baseline = test_rmse < baseline_rmse and test_r2 > baseline_r2
        
        status = "[BEATS]" if beats_baseline else "[WORSE]"
        
        print(f"{model_name:<25} {test_rmse:<12.4f} {improvement:>6.1f}% {status:<6} {test_r2:<12.6f}")


def save_best_model(results: Dict[str, Any], output_dir='models'):
    """
    Save the best performing model pipeline
    
    Args:
        results: Results from train_all_models
        output_dir: Directory to save models
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Find best model by test RMSE
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, result in results.items():
        test_rmse = result['test_metrics']['rmse']
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_model_name = model_name
    
    if best_model_name is None:
        print("[ERROR] No best model found")
        return
    
    print(f"\n{'='*70}")
    print(f"SAVING BEST MODEL: {best_model_name}")
    print(f"{'='*70}")
    
    best_result = results[best_model_name]
    best_pipeline = best_result['pipeline']
    
    # Save pipeline
    model_path = output_path / f'{best_model_name}_pipeline.pkl'
    joblib.dump(best_pipeline, model_path)
    print(f"✓ Pipeline saved: {model_path}")
    
    # Save metrics
    metrics_path = output_path / f'{best_model_name}_metrics.json'
    metrics_data = {
        'model_name': best_model_name,
        'train_metrics': best_result['train_metrics'],
        'test_metrics': best_result['test_metrics']
    }
    
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"✓ Metrics saved: {metrics_path}")
    
    print(f"\nBest Model Performance:")
    print(f"  Test RMSE: {best_result['test_metrics']['rmse']:.4f}")
    print(f"  Test MAE: {best_result['test_metrics']['mae']:.4f}")
    print(f"  Test R²: {best_result['test_metrics']['r2']:.6f}")
    
    return best_model_name, best_pipeline


def save_all_results(results: Dict[str, Any], output_path='all_models_results.json'):
    """
    Save results for all models
    
    Args:
        results: Results from train_all_models
        output_path: Output file path
    """
    output_data = {}
    
    for model_name, result in results.items():
        output_data[model_name] = {
            'train_metrics': result['train_metrics'],
            'test_metrics': result['test_metrics']
        }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n✓ All results saved: {output_path}")


if __name__ == "__main__":
    """
    Standalone execution - train all models with full pipelines
    """
    from sklearn.model_selection import train_test_split
    
    # Add to path
    sys.path.insert(0, os.path.join(project_root, 'src'))
    from data_validation import validate_data, TARGET_COLUMN
    
    data_path = os.path.join(project_root, 'data', 'raw', 'car_sales.csv')
    
    print("="*70)
    print("FULL TRAINING PIPELINE")
    print("Preprocessing + Model combined")
    print("="*70)
    
    # Load and validate data
    print("\n[1] Loading data...")
    is_valid, df, errors, warnings = validate_data(data_path)
    
    if not is_valid:
        print("[ERROR] Data validation failed!")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"✓ Data validated: {df.shape}")
    
    # Prepare data
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Split: Train={len(X_train)}, Test={len(X_test)}")
    
    # Train all models
    print("\n[2] Training all models...")
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # Compare with baselines
    print("\n[3] Comparing with baselines...")
    baseline_path = os.path.join(project_root, 'baseline_lower_bounds.json')
    compare_with_baselines(results, baseline_path)
    
    # Save results
    print("\n[4] Saving results...")
    save_all_results(results, os.path.join(project_root, 'all_models_results.json'))
    
    # Save best model
    print("\n[5] Saving best model...")
    models_dir = os.path.join(project_root, 'models')
    save_best_model(results, models_dir)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print("\nAll models trained with full pipelines (preprocessing + model)")
    print("Ready for deployment!")
