"""
Deployment Gate
Implements safe deployment logic: only promote if new model is better than current production model.
"""

import json
import logging
from pathlib import Path
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeploymentGate")

def run_deployment_gate(candidate_dir: str, production_dir: str) -> bool:
    """
    Compares candidate model with current production model.
    Returns True if candidate is better (lower RMSE), False otherwise.
    """
    logger.info("="*80)
    logger.info("DEPLOYMENT GATE: CANDIDATE VS PRODUCTION")
    logger.info("="*80)

    candidate_path = Path(candidate_dir)
    production_path = Path(production_dir)
    
    candidate_metrics_file = candidate_path / "metrics.json"
    production_metrics_file = production_path / "metrics.json"

    # 1. Load Candidate Metrics
    if not candidate_metrics_file.exists():
        logger.error(f"Candidate metrics not found at {candidate_metrics_file}")
        return False
        
    with open(candidate_metrics_file, 'r') as f:
        candidate_data = json.load(f)
    
    # Handle different JSON structures if necessary
    # In our case, it's {'test': {'rmse': ...}}
    candidate_rmse = candidate_data.get('test', {}).get('rmse')
    if candidate_rmse is None:
        logger.error("Could not find RMSE in candidate metrics.")
        return False

    # 2. Load Production Metrics
    if not production_metrics_file.exists():
        logger.warning(f"Production metrics not found at {production_metrics_file}. First deployment?")
        return True # Approve if no production model exists
        
    with open(production_metrics_file, 'r') as f:
        production_data = json.load(f)
        
    production_rmse = production_data.get('test', {}).get('rmse')
    if production_rmse is None:
        logger.warning("Could not find RMSE in production metrics. Approving candidate.")
        return True

    # 3. Decision Logic
    logger.info(f"Candidate RMSE:  {candidate_rmse:.4f}")
    logger.info(f"Production RMSE: {production_rmse:.4f}")
    
    if candidate_rmse < production_rmse:
        improvement = production_rmse - candidate_rmse
        logger.info(f"✓ APPROVE: Candidate is better by {improvement:.4f}")
        return True
    else:
        degradation = candidate_rmse - production_rmse
        logger.warning(f"✗ REJECT: Candidate is worse by {degradation:.4f}")
        return False

def promote_model(candidate_dir: str, production_dir: str):
    """
    Promotes candidate model to production by copying files.
    """
    candidate_path = Path(candidate_dir)
    production_path = Path(production_dir)
    
    logger.info(f"Promoting model from {candidate_dir} to {production_dir}...")
    
    if production_path.exists():
        # Backup old production model just in case
        backup_dir = production_path.parent / f"backup_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}" if 'pd' in globals() else production_path.parent / "production_backup"
        # Since pd might not be here, use simple backup
        backup_dir = production_path.parent / "old_production_backup"
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(production_path, backup_dir)
        shutil.rmtree(production_path)
        
    shutil.copytree(candidate_path, production_path)
    logger.info("✓ Model promotion complete.")

if __name__ == "__main__":
    # Example usage
    import sys
    if len(sys.argv) < 3:
        print("Usage: python deployment_gate.py <candidate_dir> <production_dir>")
        sys.exit(1)
        
    approved = run_deployment_gate(sys.argv[1], sys.argv[2])
    if approved:
        promote_model(sys.argv[1], sys.argv[2])
        sys.exit(0)
    else:
        sys.exit(1)
