"""
AutoML Preprocessor
Automatically detects feature types and applies appropriate transformations.
Enables "Zero-Touch" adaptation to new datasets.
"""

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logger = logging.getLogger(__name__)

def create_auto_pipeline(df: pd.DataFrame, target_column: str):
    """
    Automatically creates a preprocessing pipeline based on the dataframe structure.
    """
    X = df.drop(columns=[target_column])
    
    # 1. Identify feature types automatically
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    logger.info(f"Auto-detected {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features.")
    
    # 2. Build sub-pipelines
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # 3. Combine into ColumnTransformer
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], remainder='drop')
    
    return preprocessor, numerical_cols, categorical_cols

class AutoPreprocessor:
    """
    Wrapper for the dynamic pipeline to handle fit/transform and metadata storage.
    """
    def __init__(self):
        self.preprocessor = None
        self.target_column = None
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_names = []

    def fit(self, df: pd.DataFrame, target_column: str):
        self.target_column = target_column
        self.preprocessor, self.numerical_cols, self.categorical_cols = create_auto_pipeline(df, target_column)
        
        X = df.drop(columns=[target_column])
        self.preprocessor.fit(X)
        
        # Get feature names after one-hot encoding
        try:
            ohe = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_features = ohe.get_feature_names_out(self.categorical_cols).tolist()
            self.feature_names = self.numerical_cols + cat_features
        except:
            self.feature_names = self.numerical_cols + self.categorical_cols
            
        return self

    def transform(self, df: pd.DataFrame):
        X = df.copy()
        if self.target_column in X.columns:
            X = X.drop(columns=[self.target_column])
        return self.preprocessor.transform(X)
