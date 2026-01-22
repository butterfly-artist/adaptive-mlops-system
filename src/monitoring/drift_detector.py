"""
Drift Detection Module
Compares production data with training data to detect feature and target drift.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DriftDetector:
    """
    Detects drift between reference (training) data and current (production) data.
    """
    
    def __init__(self, reference_data_path: str, p_value_threshold: float = 0.05, psi_threshold: float = 0.2):
        self.reference_data_path = Path(reference_data_path)
        self.p_value_threshold = p_value_threshold
        self.psi_threshold = psi_threshold
        self.reference_df = self._load_reference_data()
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(exist_ok=True)
        
    def _load_reference_data(self) -> pd.DataFrame:
        """Load the reference (training) dataset."""
        if not self.reference_data_path.exists():
            raise FileNotFoundError(f"Reference data not found at {self.reference_data_path}")
        return pd.read_csv(self.reference_data_path)

    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """
        Calculate the Population Stability Index (PSI).
        """
        def scale_range(data, min_val, max_val):
            return (data - min_val) / (max_val - min_val)

        # Handle empty or constant data
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
        
        min_val = min(expected.min(), actual.min())
        max_val = max(expected.max(), actual.max())
        
        if min_val == max_val:
            return 0.0

        # Create buckets
        breakpoints = np.linspace(min_val, max_val, buckets + 1)
        
        expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
        actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)

        # Avoid division by zero
        expected_percents = np.clip(expected_percents, 0.0001, 1)
        actual_percents = np.clip(actual_percents, 0.0001, 1)

        psi_value = np.sum((expected_percents - actual_percents) * np.log(expected_percents / actual_percents))
        return float(psi_value)
    
    def detect_numerical_drift(self, current_df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Detect drift in numerical columns using KS test and PSI.
        """
        if column not in self.reference_df.columns or column not in current_df.columns:
            return {"status": "error", "message": f"Column {column} not found"}
        
        ref_data = self.reference_df[column].dropna()
        curr_data = current_df[column].dropna()
        
        if len(curr_data) < 5:
            return {"status": "insufficient_data", "column": column}
            
        ks_stat, p_value = stats.ks_2samp(ref_data, curr_data)
        psi_value = self.calculate_psi(ref_data.values, curr_data.values)
        
        is_drifted = p_value < self.p_value_threshold or psi_value > self.psi_threshold
        
        return {
            "column": column,
            "ks_statistic": float(ks_stat),
            "p_value": float(p_value),
            "psi": psi_value,
            "is_drifted": bool(is_drifted),
            "ref_mean": float(ref_data.mean()),
            "curr_mean": float(curr_data.mean()),
            "ref_std": float(ref_data.std()),
            "curr_std": float(curr_data.std())
        }
    
    def detect_categorical_drift(self, current_df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Detect drift in categorical columns.
        """
        if column not in self.reference_df.columns or column not in current_df.columns:
            return {"status": "error", "message": f"Column {column} not found"}
            
        ref_counts = self.reference_df[column].value_counts(normalize=True)
        curr_counts = current_df[column].value_counts(normalize=True)
        
        all_categories = sorted(list(set(ref_counts.index) | set(curr_counts.index)))
        ref_dist = ref_counts.reindex(all_categories, fill_value=0.0001)
        curr_dist = curr_counts.reindex(all_categories, fill_value=0.0001)
        
        # PSI for categorical
        psi_value = np.sum((ref_dist - curr_dist) * np.log(ref_dist / curr_dist))
        is_drifted = psi_value > self.psi_threshold
        
        return {
            "column": column,
            "psi": float(psi_value),
            "is_drifted": bool(is_drifted),
            "ref_distribution": ref_dist.to_dict(),
            "curr_distribution": curr_dist.to_dict()
        }

    def detect_prediction_drift(self, current_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Track prediction drift: mean shift, variance change, distribution shape.
        """
        if 'prediction' not in current_df.columns:
            return {"status": "error", "message": "Prediction column not found in logs"}
            
        # We need a reference for predictions. Since we don't have historical predictions 
        # in the training data (it's the target), we compare current predictions 
        # with the training target distribution.
        ref_target = self.reference_df['Sales_in_thousands'].dropna()
        curr_preds = current_df['prediction'].dropna()
        
        if len(curr_preds) < 5:
            return {"status": "insufficient_data"}
            
        mean_shift = float(curr_preds.mean() - ref_target.mean())
        var_change = float(curr_preds.var() - ref_target.var())
        
        # KS Test for shape
        ks_stat, p_value = stats.ks_2samp(ref_target, curr_preds)
        
        return {
            "mean_shift": mean_shift,
            "variance_change": var_change,
            "ks_p_value": float(p_value),
            "is_drifted": p_value < self.p_value_threshold,
            "ref_mean": float(ref_target.mean()),
            "curr_mean": float(curr_preds.mean()),
            "ref_std": float(ref_target.std()),
            "curr_std": float(curr_preds.std())
        }

    def run_drift_analysis(self, current_data_path: str) -> Dict[str, Any]:
        """
        Run full drift analysis and generate reports.
        """
        current_path = Path(current_data_path)
        if not current_path.exists():
            return {"status": "error", "message": "Current data (logs) not found"}
            
        current_df = pd.read_csv(current_path)
        
        # 1. Feature Drift
        exclude_cols = ['timestamp', 'model_version', 'prediction']
        cols_to_check = [c for c in current_df.columns if c in self.reference_df.columns and c not in exclude_cols]
        
        feature_drift = {}
        for col in cols_to_check:
            if self.reference_df[col].dtype in ['int64', 'float64']:
                res = self.detect_numerical_drift(current_df, col)
            else:
                res = self.detect_categorical_drift(current_df, col)
            
            # Ensure JSON serializable
            feature_drift[col] = {k: (v.item() if hasattr(v, 'item') else v) for k, v in res.items()}
        
        with open(self.reports_dir / "feature_drift.json", 'w') as f:
            json.dump(feature_drift, f, indent=2)
            
        # 2. Prediction Drift
        prediction_drift_raw = self.detect_prediction_drift(current_df)
        prediction_drift = {k: (v.item() if hasattr(v, 'item') else v) for k, v in prediction_drift_raw.items()}
        
        with open(self.reports_dir / "prediction_drift.json", 'w') as f:
            json.dump(prediction_drift, f, indent=2)
            
        # 3. Summary
        drifted_features = [col for col, res in feature_drift.items() if res.get("is_drifted", False)]
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_features_checked": len(cols_to_check),
            "drifted_features_count": len(drifted_features),
            "drifted_features": drifted_features,
            "prediction_drift_detected": bool(prediction_drift.get("is_drifted", False)),
            "overall_drift_status": bool(len(drifted_features) > 0 or prediction_drift.get("is_drifted", False))
        }
        
        with open(self.reports_dir / "drift_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
            
        return summary

if __name__ == "__main__":
    from datetime import datetime
    detector = DriftDetector("data/raw/car_sales.csv")
    logs_path = "logs/predictions.csv"
    if Path(logs_path).exists():
        summary = detector.run_drift_analysis(logs_path)
        print(json.dumps(summary, indent=2))
    else:
        print("No logs found to analyze.")
