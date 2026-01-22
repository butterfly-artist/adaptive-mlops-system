"""
Training Module
Contains baseline models and full ML training pipelines
"""

from .baseline_models import (
    MeanPredictor,
    MedianPredictor,
    calculate_metrics,
    train_and_evaluate_baselines
)

from .full_pipeline import (
    create_full_pipeline,
    train_and_evaluate_model,
    train_all_models,
    get_ml_models
)

__all__ = [
    'MeanPredictor',
    'MedianPredictor',
    'calculate_metrics',
    'train_and_evaluate_baselines',
    'create_full_pipeline',
    'train_and_evaluate_model',
    'train_all_models',
    'get_ml_models'
]
