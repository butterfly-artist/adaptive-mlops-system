"""
Unit tests for Monitoring and Decision Engine components.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from src.monitoring.drift_detector import DriftDetector
from src.monitoring.retraining_decision_engine import RetrainingDecisionEngine

@pytest.fixture
def sample_data(tmp_path):
    # Create a reference dataset
    ref_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': np.random.choice(['A', 'B'], 100),
        'Sales_in_thousands': np.random.normal(100, 10, 100)
    })
    ref_path = tmp_path / "ref.csv"
    ref_df.to_csv(ref_path, index=False)
    return ref_path

def test_drift_detection_numerical(sample_data):
    detector = DriftDetector(str(sample_data))
    
    # Create current data with drift
    curr_df = pd.DataFrame({
        'feature1': np.random.normal(5, 1, 100), # Significant mean shift
        'feature2': np.random.choice(['A', 'B'], 100)
    })
    
    result = detector.detect_numerical_drift(curr_df, 'feature1')
    assert result['is_drifted'] is True
    assert result['psi'] > 0.2

def test_drift_detection_categorical(sample_data):
    detector = DriftDetector(str(sample_data))
    
    # Create current data with drift
    curr_df = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 100),
        'feature2': ['C'] * 100 # New category
    })
    
    result = detector.detect_categorical_drift(curr_df, 'feature2')
    assert result['is_drifted'] is True

def test_decision_engine_logic(tmp_path):
    engine = RetrainingDecisionEngine(min_samples_required=10, cooldown_hours=0)
    
    drift_summary = {"drifted_features_count": 5, "total_features_checked": 10} # 50% drift
    prediction_drift = {"ks_p_value": 0.01} # Significant drift
    
    # Create dummy log file
    log_path = tmp_path / "logs.csv"
    pd.DataFrame({'a': range(20)}).to_csv(log_path, index=False)
    
    # Create dummy history file
    history_path = tmp_path / "history.json"
    with open(history_path, 'w') as f:
        json.dump([], f)
        
    decision = engine.decide(drift_summary, prediction_drift, log_path, history_path)
    assert decision['trigger_retraining'] is True
    assert "All retraining conditions met" in decision['reason']

def test_decision_engine_cooldown(tmp_path):
    engine = RetrainingDecisionEngine(cooldown_hours=24)
    
    drift_summary = {"drifted_features_count": 5, "total_features_checked": 10}
    prediction_drift = {"ks_p_value": 0.01}
    
    log_path = tmp_path / "logs.csv"
    pd.DataFrame({'a': range(100)}).to_csv(log_path, index=False)
    
    # Create history with a very recent success
    history_path = tmp_path / "history.json"
    recent_time = pd.Timestamp.now().isoformat()
    history = [{
        "timestamp": recent_time,
        "decision": {"retraining_status": "success"}
    }]
    with open(history_path, 'w') as f:
        json.dump(history, f)
        
    decision = engine.decide(drift_summary, prediction_drift, log_path, history_path)
    assert decision['trigger_retraining'] is False
    assert "Cooldown active" in decision['conditions']['cooldown']['detail']
