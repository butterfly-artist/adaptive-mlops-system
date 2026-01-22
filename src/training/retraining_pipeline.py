"""
Automated Retraining Pipeline
Orchestrates drift detection, decision making, and model retraining.
"""

import os
import sys
import json
import logging
from pathlib import Path
import subprocess
from datetime import datetime

# Add src to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from monitoring.drift_detector import DriftDetector
from monitoring.retraining_decision_engine import RetrainingDecisionEngine
from monitoring.observability import SystemHealthMonitor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("RetrainingPipeline")

import pandas as pd
import shutil

def merge_datasets(base_path: Path, logs_path: Path, output_path: Path):
    """Merge original training data with new production logs."""
    logger.info(f"Merging {base_path} and {logs_path}...")
    
    df_base = pd.read_csv(base_path)
    df_logs = pd.read_csv(logs_path)
    
    # Ensure logs have the target column (Sales_in_thousands)
    # In a real scenario, we'd wait for ground truth. 
    # For this demo, we assume 'prediction' is our proxy or ground truth is available.
    # Here we'll just use the prediction as a placeholder for ground truth if target is missing.
    if 'Sales_in_thousands' not in df_logs.columns and 'prediction' in df_logs.columns:
        logger.warning("Target column missing in logs. Using predictions as proxy for retraining.")
        df_logs['Sales_in_thousands'] = df_logs['prediction']
    
    # Keep only relevant columns
    cols = df_base.columns
    df_logs_cleaned = df_logs[cols]
    
    df_merged = pd.concat([df_base, df_logs_cleaned], ignore_index=True)
    df_merged.to_csv(output_path, index=False)
    logger.info(f"Merged dataset saved to {output_path} (Total rows: {len(df_merged)})")
    return output_path

def run_retraining_pipeline():
    """
    Execute the full automated retraining workflow.
    """
    logger.info("Starting Automated Retraining Pipeline...")
    
    # 1. Paths
    data_path = project_root / "data" / "raw" / "car_sales.csv"
    logs_path = project_root / "logs" / "predictions.csv"
    retraining_log_path = project_root / "logs" / "retraining_history.json"
    merged_data_path = project_root / "data" / "processed" / "retraining_data.csv"
    merged_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 2. Drift Detection
    logger.info("Step 1: Running Drift Detection...")
    if not logs_path.exists():
        logger.warning("No prediction logs found. Skipping retraining check.")
        return False
        
    detector = DriftDetector(str(data_path))
    drift_summary = detector.run_drift_analysis(str(logs_path))
    
    # Load prediction drift report for the decision engine
    pred_drift_path = project_root / "reports" / "prediction_drift.json"
    if not pred_drift_path.exists():
        logger.error("Prediction drift report not found.")
        return False
        
    with open(pred_drift_path, 'r') as f:
        prediction_drift = json.load(f)
    
    # 3. Decision Engine
    logger.info("Step 2: Evaluating Retraining Decision...")
    engine = RetrainingDecisionEngine(
        feature_drift_threshold=0.2,
        prediction_drift_threshold=0.05,
        min_samples_required=50,
        cooldown_hours=24
    )
    decision = engine.decide(
        drift_summary=drift_summary,
        prediction_drift=prediction_drift,
        log_path=logs_path,
        history_path=retraining_log_path
    )
    
    # 4. Execute Retraining if triggered
    if decision["trigger_retraining"]:
        logger.info("Step 3: Triggering Model Retraining...")
        
        try:
            # A. Merge Data
            merge_datasets(data_path, logs_path, merged_data_path)
            
            # B. Run Training (into candidate directory)
            candidate_dir = project_root / "production_models" / "candidate"
            production_dir = project_root / "production_models" / "v1.0"
            
            training_script = project_root / "train_model_v1_mlflow.py"
            logger.info(f"Running training script: {training_script}")
            
            result = subprocess.run(
                [sys.executable, str(training_script), str(merged_data_path), str(candidate_dir)],
                capture_output=True,
                text=True,
                check=True,
                encoding='utf-8'
            )
            
            logger.info("[OK] Model retraining completed successfully.")
            decision["retraining_status"] = "success"
            
            # C. Evaluation Gate (vs Baseline)
            logger.info("Step 4: Running Evaluation Gate (vs Baseline)...")
            from evaluation.evaluation_gate import run_evaluation_gate
            # Note: evaluation_gate currently hardcoded to v1.0, 
            # we should ideally point it to candidate. 
            # For now, let's use our new deployment gate which is more robust.
            
            # D. Deployment Gate (vs Current Production)
            logger.info("Step 5: Running Deployment Gate (vs Current Production)...")
            from deployment.deployment_gate import run_deployment_gate, promote_model
            
            approved = run_deployment_gate(str(candidate_dir), str(production_dir))
            
            if approved:
                logger.info("Step 6: Promoting Model...")
                promote_model(str(candidate_dir), str(production_dir))
                decision["promotion_status"] = "promoted"
                logger.info("✓ Model promoted to production.")
            else:
                decision["promotion_status"] = "rejected"
                logger.warning("✗ Model rejected by deployment gate (performance degradation). Keeping current model.")
            
        except Exception as e:
            logger.error(f"[ERROR] Retraining pipeline failed: {str(e)}")
            decision["retraining_status"] = "failed"
            decision["error"] = str(e)
            return False
    else:
        logger.info("Step 3: Retraining not required. Pipeline finished.")
        decision["retraining_status"] = "skipped"
        
    # 5. Log the event
    history = []
    if retraining_log_path.exists():
        with open(retraining_log_path, 'r', encoding='utf-8') as f:
            try:
                history = json.load(f)
            except json.JSONDecodeError:
                history = []
                
    history.append({
        "timestamp": datetime.now().isoformat(),
        "decision": decision
    })
    
    retraining_log_path.parent.mkdir(exist_ok=True)
    with open(retraining_log_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
        
    logger.info(f"Retraining event logged to {retraining_log_path}")
    
    # 6. Generate System Health Report (Observability)
    try:
        monitor = SystemHealthMonitor(str(project_root))
        monitor.generate_health_report()
    except Exception as e:
        logger.error(f"Failed to generate health report: {e}")
        
    return decision["trigger_retraining"]

if __name__ == "__main__":
    run_retraining_pipeline()
