"""
System Observability Module
Aggregates drift scores, retraining history, and model performance into a health report.
"""

import json
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, List, Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    """
    Generates a comprehensive system health report for observability.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.reports_dir = self.project_root / "reports"
        self.logs_dir = self.project_root / "logs"
        self.models_dir = self.project_root / "production_models"
        
        self.reports_dir.mkdir(exist_ok=True)
        
    def generate_health_report(self) -> Dict[str, Any]:
        """
        Aggregate data from various sources to create the health report.
        """
        logger.info("Generating System Health Report...")
        
        # 1. Load Drift Summary
        drift_summary = {}
        drift_path = self.reports_dir / "drift_summary.json"
        if drift_path.exists():
            with open(drift_path, 'r') as f:
                drift_summary = json.load(f)
                
        # 2. Load Retraining History
        retraining_history = []
        history_path = self.logs_dir / "retraining_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                retraining_history = json.load(f)
                
        # 3. Load Current Model Metrics
        current_metrics = {}
        metrics_path = self.models_dir / "v1.0" / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path, 'r') as f:
                current_metrics = json.load(f)
                
        # 4. Aggregate Observability Metrics
        total_retrains = len(retraining_history)
        successful_retrains = len([h for h in retraining_history if h.get("decision", {}).get("retraining_status") == "success"])
        failed_retrains = len([h for h in retraining_history if h.get("decision", {}).get("retraining_status") == "failed"])
        rejected_models = len([h for h in retraining_history if h.get("decision", {}).get("promotion_status") == "rejected"])
        
        # Extract last trigger reason
        last_trigger_reason = "N/A"
        if retraining_history:
            last_trigger_reason = retraining_history[-1].get("decision", {}).get("reason", "N/A")
            
        health_report = {
            "timestamp": datetime.now().isoformat(),
            "system_status": "Healthy" if not drift_summary.get("overall_drift_status", False) else "Drift Detected",
            "model_performance": {
                "current_rmse": current_metrics.get("test", {}).get("rmse", "N/A"),
                "current_r2": current_metrics.get("test", {}).get("r2", "N/A"),
                "last_updated": datetime.fromtimestamp(metrics_path.stat().st_mtime).isoformat() if metrics_path.exists() else "N/A"
            },
            "drift_observability": {
                "last_drift_check": drift_summary.get("timestamp", "N/A"),
                "drifted_features_count": drift_summary.get("drifted_features_count", 0),
                "drifted_features": drift_summary.get("drifted_features", []),
                "prediction_drift_detected": drift_summary.get("prediction_drift_detected", False)
            },
            "retraining_observability": {
                "total_retraining_attempts": total_retrains,
                "successful_retrains": successful_retrains,
                "failed_retrains": failed_retrains,
                "rejected_by_gate": rejected_models,
                "last_trigger_reason": last_trigger_reason
            },
            "stability_metrics": {
                "retrain_success_rate": (successful_retrains / total_retrains) if total_retrains > 0 else 1.0,
                "model_promotion_rate": (successful_retrains - rejected_models) / successful_retrains if successful_retrains > 0 else 1.0
            }
        }
        
        # Save the report
        output_path = self.reports_dir / "system_health.json"
        with open(output_path, 'w') as f:
            json.dump(health_report, f, indent=2)
            
        logger.info(f"System Health Report saved to {output_path}")
        return health_report

if __name__ == "__main__":
    # Test generation
    monitor = SystemHealthMonitor(".")
    report = monitor.generate_health_report()
    print(json.dumps(report, indent=2))
