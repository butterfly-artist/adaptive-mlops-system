"""
Evaluation Gate (Mandatory)
Compares Model v1 performance against Baseline.
Decides whether to APPROVE or REJECT the model for production.
"""

import json
import sys
from pathlib import Path

def run_evaluation_gate():
    print("="*80)
    print("EVALUATION GATE: BASELINE VS MODEL v1")
    print("="*80)

    # 1. Paths
    project_root = Path(__file__).parent.parent.parent
    baseline_path = project_root / "baseline_lower_bounds.json"
    model_metrics_path = project_root / "production_models" / "v1.0" / "metrics.json"

    # 2. Load Baseline
    if not baseline_path.exists():
        print(f"✗ ERROR: Baseline metrics not found at {baseline_path}")
        sys.exit(1)
    
    with open(baseline_path, 'r') as f:
        baseline_data = json.load(f)
    
    baseline_rmse = baseline_data['baseline_lower_bounds']['min_rmse_to_beat']
    print(f"Baseline RMSE to beat: {baseline_rmse:.4f}")

    # 3. Load Model v1 Metrics
    if not model_metrics_path.exists():
        print(f"✗ ERROR: Model v1 metrics not found at {model_metrics_path}")
        sys.exit(1)

    with open(model_metrics_path, 'r') as f:
        model_data = json.load(f)
    
    model_v1_rmse = model_data['test']['rmse']
    print(f"Model v1 Test RMSE:    {model_v1_rmse:.4f}")

    # 4. Decision Logic
    improvement = baseline_rmse - model_v1_rmse
    improvement_pct = (improvement / baseline_rmse) * 100

    print("-" * 40)
    if model_v1_rmse < baseline_rmse:
        print(f"✓ DECISION: APPROVE")
        print(f"  Model v1 is better than baseline by {improvement:.4f} ({improvement_pct:.2f}%)")
        print("-" * 40)
        print("✓ Model is cleared for production deployment.")
        print("="*80)
        return True
    else:
        print(f"✗ DECISION: REJECT")
        print(f"  Model v1 is WORSE than baseline by {abs(improvement):.4f} ({abs(improvement_pct):.2f}%)")
        print("-" * 40)
        print("✗ Model does not meet quality standards. Deployment aborted.")
        print("="*80)
        return False

if __name__ == "__main__":
    approved = run_evaluation_gate()
    if not approved:
        sys.exit(1) # Fail the process
    sys.exit(0) # Pass the process
