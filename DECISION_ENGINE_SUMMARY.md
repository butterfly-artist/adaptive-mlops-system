# Retraining Decision Engine Summary

This document details the implementation of the rule-based decision layer for the automated retraining pipeline.

## 1. Decision Logic (`src/monitoring/retraining_decision_engine.py`)
The engine follows a strict, deterministic, and explainable rule set to decide if a model should be retrained.

### Rules (Strict AND)
| Rule | Metric | Threshold |
| :--- | :--- | :--- |
| **Feature Drift** | Drifted Features / Total Features | `> 0.2` (20%) |
| **Prediction Drift** | KS Test p-value | `< 0.05` |
| **Data Volume** | New samples in `logs/predictions.csv` | `>= 50` |
| **Cooldown** | Time since last successful retraining | `> 24 hours` |

## 2. Explainability
Every decision is logged with a detailed breakdown of which conditions passed or failed. This ensures that the system's actions are always auditable and understandable.

**Example Log Entry:**
```json
{
  "trigger_retraining": true,
  "reason": "All retraining conditions met...",
  "conditions": {
    "feature_drift": { "score": 0.3125, "threshold": 0.2, "passed": true },
    "prediction_drift": { "p_value": 0.0, "threshold": 0.05, "passed": true },
    "sample_count": { "count": 157, "required": 50, "passed": true },
    "cooldown": { "passed": true, "detail": "Cooldown passed..." }
  }
}
```

## 3. Integration
The Decision Engine is integrated into the `retraining_pipeline.py` orchestrator. It consumes reports from the `DriftDetector` and checks the `retraining_history.json` for cooldown status.

## 4. Verification
- **Scenario**: Simulated significant drift in 5 features and a 500% shift in predictions.
- **Result**: All conditions passed, and the engine correctly triggered a successful retraining event.
- **Audit Trail**: Verified in `logs/retraining_history.json`.

**The system now has a professional-grade decision layer that eliminates "gut feeling" and ensures model updates are driven by data and safety constraints.** 🛡️
