"""
Feature Engineering Pipeline

Deterministic preprocessing pipeline using sklearn Pipeline and ColumnTransformer.
Handles:
- Missing values
- Categorical encoding
- Numerical scaling
- Date feature extraction
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from pathlib import Path

try:
    from .config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        DATE_FEATURES,
        TARGET_COLUMN,
        NUMERICAL_IMPUTATION_STRATEGY,
        CATEGORICAL_IMPUTATION_STRATEGY,
        SCALING_STRATEGY,
        CATEGORICAL_ENCODING,
        ONEHOT_DROP,
        ONEHOT_HANDLE_UNKNOWN,
        DATE_FORMAT,
        DATE_FEATURES_TO_EXTRACT,
        RANDOM_STATE
    )
    from .transformers import DateFeatureExtractor
except ImportError:
    from config import (
        NUMERICAL_FEATURES,
        CATEGORICAL_FEATURES,
        DATE_FEATURES,
        TARGET_COLUMN,
        NUMERICAL_IMPUTATION_STRATEGY,
        CATEGORICAL_IMPUTATION_STRATEGY,
        SCALING_STRATEGY,
        CATEGORICAL_ENCODING,
        ONEHOT_DROP,
        ONEHOT_HANDLE_UNKNOWN,
        DATE_FORMAT,
        DATE_FEATURES_TO_EXTRACT,
        RANDOM_STATE
    )
    from transformers import DateFeatureExtractor


def get_scaler(strategy='standard'):
    """
    Get scaler based on strategy.
    
    Args:
        strategy: 'standard', 'minmax', or 'robust'
        
    Returns:
        Scaler instance
    """
    if strategy == 'standard':
        return StandardScaler()
    elif strategy == 'minmax':
        return MinMaxScaler()
    elif strategy == 'robust':
        return RobustScaler()
    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")


def create_preprocessing_pipeline(
    numerical_features=None,
    categorical_features=None,
    date_features=None,
    verbose=False
):
    """
    Create a deterministic preprocessing pipeline.
    
    Args:
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        date_features: List of date feature names
        verbose: Whether to print pipeline info
        
    Returns:
        sklearn Pipeline object
    """
    # Use defaults if not provided
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES
    if date_features is None:
        date_features = DATE_FEATURES
    
    transformers = []
    
    # ============================================================
    # Numerical Features Pipeline
    # ============================================================
    if numerical_features:
        numerical_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(
                strategy=NUMERICAL_IMPUTATION_STRATEGY,
                add_indicator=False  # Don't add missing indicator
            )),
            ('scaler', get_scaler(SCALING_STRATEGY))
        ])
        
        transformers.append((
            'numerical',
            numerical_pipeline,
            numerical_features
        ))
        
        if verbose:
            print(f"[Numerical Pipeline] {len(numerical_features)} features")
            print(f"  - Imputation: {NUMERICAL_IMPUTATION_STRATEGY}")
            print(f"  - Scaling: {SCALING_STRATEGY}")
    
    # ============================================================
    # Categorical Features Pipeline
    # ============================================================
    if categorical_features:
        if CATEGORICAL_ENCODING == 'onehot':
            categorical_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(
                    strategy=CATEGORICAL_IMPUTATION_STRATEGY,
                    fill_value='missing'
                )),
                ('encoder', OneHotEncoder(
                    drop=ONEHOT_DROP,
                    handle_unknown=ONEHOT_HANDLE_UNKNOWN,
                    sparse_output=False,  # Return dense array
                    dtype=np.float64
                ))
            ])
        else:
            raise NotImplementedError(f"Encoding '{CATEGORICAL_ENCODING}' not implemented")
        
        transformers.append((
            'categorical',
            categorical_pipeline,
            categorical_features
        ))
        
        if verbose:
            print(f"[Categorical Pipeline] {len(categorical_features)} features")
            print(f"  - Imputation: {CATEGORICAL_IMPUTATION_STRATEGY}")
            print(f"  - Encoding: {CATEGORICAL_ENCODING}")
    
    # ============================================================
    # Date Features Pipeline
    # ============================================================
    if date_features:
        date_pipeline = Pipeline(steps=[
            ('date_extractor', DateFeatureExtractor(
                date_format=DATE_FORMAT,
                features_to_extract=DATE_FEATURES_TO_EXTRACT
            )),
            ('scaler', get_scaler(SCALING_STRATEGY))
        ])
        
        transformers.append((
            'date',
            date_pipeline,
            date_features
        ))
        
        if verbose:
            print(f"[Date Pipeline] {len(date_features)} features")
            print(f"  - Extraction: {DATE_FEATURES_TO_EXTRACT}")
            print(f"  - Scaling: {SCALING_STRATEGY}")
    
    # ============================================================
    # Combine All Transformers
    # ============================================================
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop',  # Drop any columns not specified
        n_jobs=1,  # Use 1 job for reproducibility
        verbose=verbose
    )
    
    # Wrap in Pipeline for consistency
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    if verbose:
        print("\n[Pipeline Created]")
        print(f"  Total transformers: {len(transformers)}")
        print(f"  Random state: {RANDOM_STATE}")
    
    return pipeline


def fit_and_save_pipeline(X_train, output_path='preprocessing_pipeline.pkl'):
    """
    Fit preprocessing pipeline and save to disk.
    
    Args:
        X_train: Training features (DataFrame)
        output_path: Path to save the pipeline
        
    Returns:
        Fitted pipeline
    """
    print("="*70)
    print("FITTING PREPROCESSING PIPELINE")
    print("="*70)
    
    pipeline = create_preprocessing_pipeline(verbose=True)
    
    print(f"\nFitting on {X_train.shape[0]} samples...")
    pipeline.fit(X_train)
    
    print(f"\n[SAVING] Pipeline to: {output_path}")
    joblib.dump(pipeline, output_path)
    
    print("[SUCCESS] Pipeline fitted and saved!")
    
    return pipeline


def load_pipeline(pipeline_path='preprocessing_pipeline.pkl'):
    """
    Load preprocessing pipeline from disk.
    
    Args:
        pipeline_path: Path to saved pipeline
        
    Returns:
        Loaded pipeline
    """
    if not Path(pipeline_path).exists():
        raise FileNotFoundError(f"Pipeline not found: {pipeline_path}")
    
    pipeline = joblib.load(pipeline_path)
    print(f"[LOADED] Pipeline from: {pipeline_path}")
    
    return pipeline


def transform_data(pipeline, X):
    """
    Transform data using fitted pipeline.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        X: Features to transform (DataFrame)
        
    Returns:
        Transformed features (numpy array)
    """
    return pipeline.transform(X)


def get_feature_names(pipeline):
    """
    Get feature names after transformation.
    
    Args:
        pipeline: Fitted preprocessing pipeline
        
    Returns:
        List of feature names
    """
    try:
        # Get feature names from the ColumnTransformer
        preprocessor = pipeline.named_steps['preprocessor']
        
        feature_names = []
        
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            elif name == 'categorical':
                # OneHotEncoder feature names
                encoder = transformer.named_steps['encoder']
                names = encoder.get_feature_names_out(columns)
                feature_names.extend(names)
            elif name == 'date':
                # Date features
                for col in columns:
                    for feat in DATE_FEATURES_TO_EXTRACT:
                        feature_names.append(f'{col}_{feat}')
            else:
                # Numerical features keep their names
                feature_names.extend(columns)
        
        return feature_names
    except Exception as e:
        print(f"[WARNING] Could not extract feature names: {e}")
        return None


if __name__ == "__main__":
    """
    Standalone execution for testing pipeline
    """
    import sys
    import os
    from sklearn.model_selection import train_test_split
    
    # Add project root to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    sys.path.insert(0, os.path.join(project_root, 'src'))
    
    # Load and validate data
    from data_validation import validate_data, TARGET_COLUMN as VAL_TARGET
    
    data_path = os.path.join(project_root, 'data', 'raw', 'car_sales.csv')
    
    print("="*70)
    print("PREPROCESSING PIPELINE TEST")
    print("="*70)
    
    # Validate data
    is_valid, df, errors, warnings = validate_data(data_path)
    
    if not is_valid:
        print("\n[ERROR] Data validation failed!")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print(f"\n[OK] Data validated: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # Prepare data
    X = df.drop(columns=[VAL_TARGET])
    y = df[VAL_TARGET]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Create and fit pipeline
    pipeline = fit_and_save_pipeline(
        X_train,
        output_path=os.path.join(project_root, 'preprocessing_pipeline.pkl')
    )
    
    # Transform data
    print("\n" + "="*70)
    print("TRANSFORMING DATA")
    print("="*70)
    
    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)
    
    print(f"\nOriginal shape: {X_train.shape}")
    print(f"Transformed shape: {X_train_transformed.shape}")
    
    # Get feature names
    feature_names = get_feature_names(pipeline)
    if feature_names:
        print(f"\nTransformed features ({len(feature_names)}):")
        print(f"  First 10: {feature_names[:10]}")
        print(f"  Last 10: {feature_names[-10:]}")
    
    # Check for NaN values
    has_nan_train = np.isnan(X_train_transformed).any()
    has_nan_test = np.isnan(X_test_transformed).any()
    
    print(f"\nNaN check:")
    print(f"  Training set: {'FAILED - Contains NaN!' if has_nan_train else 'PASS - No NaN'}")
    print(f"  Test set: {'FAILED - Contains NaN!' if has_nan_test else 'PASS - No NaN'}")
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE TEST COMPLETE")
    print("="*70)
    
    if not has_nan_train and not has_nan_test:
        print("\n[SUCCESS] Pipeline is working correctly!")
        print("Ready for model training!")
    else:
        print("\n[WARNING] Pipeline produced NaN values - check configuration")
        sys.exit(1)
