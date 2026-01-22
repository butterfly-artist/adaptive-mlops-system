"""
Custom Transformers for Feature Engineering
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin


class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Extract numerical features from date columns.
    
    Converts date strings to numerical features like year, month, 
    day of year, and days since epoch.
    """
    
    def __init__(self, date_format='%m/%d/%Y', features_to_extract=None):
        """
        Args:
            date_format: Format string for parsing dates
            features_to_extract: List of features to extract
                Options: 'year', 'month', 'day', 'day_of_week', 
                        'day_of_year', 'quarter', 'days_since_epoch'
        """
        self.date_format = date_format
        self.features_to_extract = features_to_extract or ['year', 'month']
        self.epoch = datetime(1970, 1, 1)
        
    def fit(self, X, y=None):
        """Fit transformer (no-op for this transformer)"""
        return self
    
    def transform(self, X):
        """
        Transform date columns to numerical features.
        
        Args:
            X: DataFrame with date columns
            
        Returns:
            DataFrame with extracted date features
        """
        X_copy = X.copy()
        
        for col in X_copy.columns:
            # Parse dates
            dates = pd.to_datetime(X_copy[col], format=self.date_format, errors='coerce')
            
            # Extract features
            feature_dict = {}
            
            if 'year' in self.features_to_extract:
                feature_dict[f'{col}_year'] = dates.dt.year
            
            if 'month' in self.features_to_extract:
                feature_dict[f'{col}_month'] = dates.dt.month
            
            if 'day' in self.features_to_extract:
                feature_dict[f'{col}_day'] = dates.dt.day
            
            if 'day_of_week' in self.features_to_extract:
                feature_dict[f'{col}_day_of_week'] = dates.dt.dayofweek
            
            if 'day_of_year' in self.features_to_extract:
                feature_dict[f'{col}_day_of_year'] = dates.dt.dayofyear
            
            if 'quarter' in self.features_to_extract:
                feature_dict[f'{col}_quarter'] = dates.dt.quarter
            
            if 'days_since_epoch' in self.features_to_extract:
                feature_dict[f'{col}_days_since_epoch'] = (dates - self.epoch).dt.days
            
            # Drop original date column
            X_copy = X_copy.drop(columns=[col])
            
            # Add new features
            for feat_name, feat_values in feature_dict.items():
                X_copy[feat_name] = feat_values
        
        return X_copy
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if input_features is None:
            return None
        
        output_features = []
        for col in input_features:
            for feat in self.features_to_extract:
                output_features.append(f'{col}_{feat}')
        
        return np.array(output_features)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    Select specific columns from a DataFrame.
    
    Useful for applying different transformations to different column sets.
    """
    
    def __init__(self, attribute_names):
        """
        Args:
            attribute_names: List of column names to select
        """
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        """Fit transformer (no-op)"""
        return self
    
    def transform(self, X):
        """Select columns"""
        return X[self.attribute_names]
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        return np.array(self.attribute_names)


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """
    Drop specified columns from DataFrame.
    """
    
    def __init__(self, columns_to_drop):
        """
        Args:
            columns_to_drop: List of column names to drop
        """
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        """Fit transformer (no-op)"""
        return self
    
    def transform(self, X):
        """Drop columns"""
        return X.drop(columns=self.columns_to_drop, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        if input_features is None:
            return None
        return np.array([col for col in input_features if col not in self.columns_to_drop])


class DebugTransformer(BaseEstimator, TransformerMixin):
    """
    Debug transformer to print shape and info during pipeline execution.
    Useful for debugging pipelines.
    """
    
    def __init__(self, name='Debug', verbose=True):
        """
        Args:
            name: Name for this debug point
            verbose: Whether to print debug info
        """
        self.name = name
        self.verbose = verbose
        
    def fit(self, X, y=None):
        """Fit transformer"""
        if self.verbose:
            print(f"[{self.name}] Fit - Shape: {X.shape}")
        return self
    
    def transform(self, X):
        """Transform (pass through)"""
        if self.verbose:
            print(f"[{self.name}] Transform - Shape: {X.shape}")
            if isinstance(X, pd.DataFrame):
                print(f"[{self.name}] Columns: {list(X.columns)[:10]}...")
        return X
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        return input_features
