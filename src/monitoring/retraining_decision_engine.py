"""
Retraining Decision Engine
Analyzes drift results and performance metrics to decide if retraining is necessary.
"""

import json
from pathlib import Path
import logging
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrainingDecisionEngine:
    """
    Makes the final call on whether to trigger the retraining pipeline based on strict rules.
    """
    
    def __init__(
        self, 
        feature_drift_threshold: float = 0.2,
        prediction_drift_threshold: float = 0.05, # p-value threshold
        min_samples_required: int = 50,
        cooldown_hours: int = 24
    ):
        self.feature_drift_threshold = feature_drift_threshold
        self.prediction_drift_threshold = prediction_drift_threshold
        self.min_samples_required = min_samples_required
        self.cooldown_hours = cooldown_hours
        
    def _check_cooldown(self, history_path: Path) -> Tuple[bool, str]:
        """Check if enough time has passed since the last successful retraining."""
        if not history_path.exists():
            return True, "No previous retraining history found."
            
        try:
            with open(history_path, 'r', encoding='utf-8') as f:
                history = json.load(f)
            
            if not history:
                return True, "History is empty."
                
            # Find last success
            successes = [h for h in history if h.get("decision", {}).get("retraining_status") == "success"]
            if not successes:
                return True, "No successful retraining found in history."
                
            last_success_time = datetime.fromisoformat(successes[-1]["timestamp"])
            time_since = datetime.now() - last_success_time
            
            if time_since < timedelta(hours=self.cooldown_hours):
                return False, f"Cooldown active. Last success was {time_since.total_seconds()/3600:.1f} hours ago (Required: {self.cooldown_hours}h)."
            
            return True, f"Cooldown passed. Last success was {time_since.total_seconds()/3600:.1f} hours ago."
            
        except Exception as e:
            logger.error(f"Error checking cooldown: {e}")
            return True, f"Error checking cooldown: {e}. Defaulting to True."

    def decide(
        self, 
        drift_summary: Dict[str, Any], 
        prediction_drift: Dict[str, Any],
        log_path: Path,
        history_path: Path
    ) -> Dict[str, Any]:
        """
        Evaluate rules and return a deterministic decision.
        """
        logger.info("Evaluating retraining decision rules...")
        
        # 1. Feature Drift Score
        drifted_count = drift_summary.get("drifted_features_count", 0)
        total_features = drift_summary.get("total_features_checked", 1)
        feature_drift_score = drifted_count / total_features
        feature_drift_pass = feature_drift_score >= self.feature_drift_threshold
        
        # 2. Prediction Drift Score
        # We use the KS p-value from prediction drift analysis
        # If p-value < threshold, drift is detected
        p_value = prediction_drift.get("ks_p_value", 1.0)
        prediction_drift_pass = p_value < self.prediction_drift_threshold
        
        # 3. New Samples Count
        try:
            df_logs = pd.read_csv(log_path)
            sample_count = len(df_logs)
        except Exception:
            sample_count = 0
        sample_count_pass = sample_count >= self.min_samples_required
        
        # 4. Cooldown Period
        cooldown_pass, cooldown_reason = self._check_cooldown(history_path)
        
        # Final Decision (Strict AND)
        trigger = feature_drift_pass and prediction_drift_pass and sample_count_pass and cooldown_pass
        
        # Explainability
        conditions = {
            "feature_drift": {
                "score": round(feature_drift_score, 4),
                "threshold": self.feature_drift_threshold,
                "passed": bool(feature_drift_pass)
            },
            "prediction_drift": {
                "p_value": round(p_value, 4),
                "threshold": self.prediction_drift_threshold,
                "passed": bool(prediction_drift_pass)
            },
            "sample_count": {
                "count": sample_count,
                "required": self.min_samples_required,
                "passed": bool(sample_count_pass)
            },
            "cooldown": {
                "passed": bool(cooldown_pass),
                "detail": cooldown_reason
            }
        }
        
        if trigger:
            reason = "All retraining conditions met: Significant feature drift, prediction drift, sufficient data, and cooldown passed."
        else:
            failed_reasons = []
            if not feature_drift_pass: failed_reasons.append("Feature drift below threshold")
            if not prediction_drift_pass: failed_reasons.append("Prediction drift not significant")
            if not sample_count_pass: failed_reasons.append("Insufficient new samples")
            if not cooldown_pass: failed_reasons.append("Cooldown period not passed")
            reason = "Retraining skipped: " + ", ".join(failed_reasons)
            
        decision = {
            "trigger_retraining": bool(trigger),
            "reason": reason,
            "conditions": conditions,
            "timestamp": datetime.now().isoformat()
        }
        
        if trigger:
            logger.warning(f"RETRAINING TRIGGERED: {reason}")
        else:
            logger.info(f"Retraining not required: {reason}")
            
        return decision

if __name__ == "__main__":
    # Test with dummy data
    from typing import Tuple
    engine = RetrainingDecisionEngine(min_samples_required=10)
    
    mock_drift = {"drifted_features_count": 5, "total_features_checked": 16}
    mock_pred = {"ks_p_value": 0.01}
    
    # We'd need actual files to test fully, but this shows the structure
    print("Decision Engine Initialized.")
