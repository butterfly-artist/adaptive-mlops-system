"""
Model Loader Utility
Handles loading the production model pipeline
"""

import joblib
import os
from pathlib import Path
import json

def get_latest_model_path():
    """
    Find the latest production model pipeline.
    In a real scenario, this might query MLflow or a Model Registry.
    For now, we use the local production_models directory.
    """
    base_path = Path(__file__).parent.parent.parent / "production_models"
    if not base_path.exists():
        return None
    
    # Get all version directories (e.g., v1.0, v1.1)
    versions = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('v')]
    if not versions:
        return None
    
    # Sort by version name (simple string sort works for v1.0, v2.0 etc)
    latest_version = sorted(versions)[-1]
    
    # Find the .pkl file in that directory
    model_files = list(latest_version.glob("*_pipeline.pkl"))
    if not model_files:
        return None
        
    return model_files[0]

def load_production_model():
    """Load the latest production model pipeline"""
    model_path = get_latest_model_path()
    if not model_path:
        raise FileNotFoundError("No production model found in production_models/")
        
    print(f"Loading model from: {model_path}")
    return joblib.load(model_path)
