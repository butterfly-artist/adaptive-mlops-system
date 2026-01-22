"""
Baseline Models for Car Sales Prediction

Implements simple baseline predictors:
- Mean Predictor: Predicts training mean for all samples
- Median Predictor: Predicts training median for all samples

These provide reference performance and lower bound for ML models.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any
import json


class MeanPredictor:
    """
    Baseline predictor that always predicts the mean of training data.
    
    This is the simplest possible model and represents the lower bound
    for any regression task.
    """
    
    def __init__(self):
        self.mean_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MeanPredictor':
        """
        Fit the mean predictor.
        
        Args:
            X: Features (not used, only for API compatibility)
            y: Target values
            
        Returns:
            self
        """
        self.mean_ = float(y.mean())
        self.n_samples_ = len(y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the mean value.
        
        Args:
            X: Features (not used, only for API compatibility)
            
        Returns:
            Array of predictions (all equal to training mean)
        """
        if self.mean_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return np.full(len(X), self.mean_)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'model_type': 'MeanPredictor',
            'mean_value': self.mean_,
            'n_train_samples': self.n_samples_
        }


class MedianPredictor:
    """
    Baseline predictor that always predicts the median of training data.
    
    Often more robust than mean when data has outliers.
    Minimizes Mean Absolute Error (MAE).
    """
    
    def __init__(self):
        self.median_: Optional[float] = None
        self.n_samples_: Optional[int] = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'MedianPredictor':
        """
        Fit the median predictor.
        
        Args:
            X: Features (not used, only for API compatibility)
            y: Target values
            
        Returns:
            self
        """
        self.median_ = float(y.median())
        self.n_samples_ = len(y)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the median value.
        
        Args:
            X: Features (not used, only for API compatibility)
            
        Returns:
            Array of predictions (all equal to training median)
        """
        if self.median_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        return np.full(len(X), self.median_)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters"""
        return {
            'model_type': 'MedianPredictor',
            'median_value': self.median_,
            'n_train_samples': self.n_samples_
        }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        mean_squared_error,
        mean_absolute_error,
        r2_score,
        mean_absolute_percentage_error
    )
    
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE - handle division by zero
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    except:
        mape = np.nan
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mape': float(mape)
    }


def train_and_evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Dict[str, Any]:
    """
    Train and evaluate both baseline models.
    
    Args:
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary containing models and metrics
    """
    results = {
        'mean_predictor': {},
        'median_predictor': {},
        'comparison': {}
    }
    
    # Train Mean Predictor
    print("\n" + "="*70)
    print("MEAN PREDICTOR")
    print("="*70)
    
    mean_model = MeanPredictor()
    mean_model.fit(X_train, y_train)
    
    print(f"Training mean: {mean_model.mean_:.4f}")
    
    # Evaluate on train set
    y_train_pred_mean = mean_model.predict(X_train)
    train_metrics_mean = calculate_metrics(y_train.values, y_train_pred_mean)
    
    # Evaluate on test set
    y_test_pred_mean = mean_model.predict(X_test)
    test_metrics_mean = calculate_metrics(y_test.values, y_test_pred_mean)
    
    print("\nTrain Metrics:")
    for metric, value in train_metrics_mean.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics_mean.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    results['mean_predictor'] = {
        'model': mean_model,
        'params': mean_model.get_params(),
        'train_metrics': train_metrics_mean,
        'test_metrics': test_metrics_mean
    }
    
    # Train Median Predictor
    print("\n" + "="*70)
    print("MEDIAN PREDICTOR")
    print("="*70)
    
    median_model = MedianPredictor()
    median_model.fit(X_train, y_train)
    
    print(f"Training median: {median_model.median_:.4f}")
    
    # Evaluate on train set
    y_train_pred_median = median_model.predict(X_train)
    train_metrics_median = calculate_metrics(y_train.values, y_train_pred_median)
    
    # Evaluate on test set
    y_test_pred_median = median_model.predict(X_test)
    test_metrics_median = calculate_metrics(y_test.values, y_test_pred_median)
    
    print("\nTrain Metrics:")
    for metric, value in train_metrics_median.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    print("\nTest Metrics:")
    for metric, value in test_metrics_median.items():
        print(f"  {metric.upper()}: {value:.4f}")
    
    results['median_predictor'] = {
        'model': median_model,
        'params': median_model.get_params(),
        'train_metrics': train_metrics_median,
        'test_metrics': test_metrics_median
    }
    
    # Comparison
    print("\n" + "="*70)
    print("BASELINE COMPARISON (Test Set)")
    print("="*70)
    
    comparison = pd.DataFrame({
        'Mean Predictor': test_metrics_mean,
        'Median Predictor': test_metrics_median
    }).T
    
    print(comparison.to_string())
    
    # Determine better baseline
    better_model = 'Mean' if test_metrics_mean['rmse'] < test_metrics_median['rmse'] else 'Median'
    print(f"\nBetter baseline (by RMSE): {better_model} Predictor")
    
    results['comparison'] = {
        'better_baseline': better_model,
        'comparison_table': comparison.to_dict()
    }
    
    return results


def save_baseline_results(results: Dict[str, Any], output_path: str) -> None:
    """
    Save baseline results to JSON file.
    
    Args:
        results: Results dictionary from train_and_evaluate_baselines
        output_path: Path to save JSON file
    """
    output = {
        'mean_predictor': {
            'params': results['mean_predictor']['params'],
            'train_metrics': results['mean_predictor']['train_metrics'],
            'test_metrics': results['mean_predictor']['test_metrics']
        },
        'median_predictor': {
            'params': results['median_predictor']['params'],
            'train_metrics': results['median_predictor']['train_metrics'],
            'test_metrics': results['median_predictor']['test_metrics']
        },
        'comparison': results['comparison']
    }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n[SAVED] Baseline results saved to: {output_path}")


if __name__ == "__main__":
    """
    Standalone execution for testing baseline models
    """
    import sys
    import os
    from sklearn.model_selection import train_test_split
    
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, os.path.join(project_root, 'src'))
    
    # Load and validate data
    from data_validation import validate_data, TARGET_COLUMN
    
    data_path = os.path.join(project_root, 'data', 'raw', 'car_sales.csv')
    
    print("="*70)
    print("BASELINE MODEL EVALUATION")
    print("="*70)
    
    # Validate data
    is_valid, df, errors, warnings = validate_data(data_path)
    
    if not is_valid:
        print("\n[ERROR] Data validation failed!")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"\n[OK] Data validated: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Prepare data
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train and evaluate baselines
    results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test)
    
    # Save results
    output_path = os.path.join(project_root, 'baseline_results.json')
    save_baseline_results(results, output_path)
    
    print("\n" + "="*70)
    print("BASELINE MODELS COMPLETE")
    print("="*70)
    print("\nThese results establish the LOWER BOUND for ML models.")
    print("Any ML model must beat these simple baselines to be useful!")
