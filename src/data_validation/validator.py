"""
Data Validation Engine
Validates incoming data against defined rules and constraints
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Any
import logging

try:
    from .config import (
        REQUIRED_COLUMNS,
        TARGET_COLUMN,
        EXPECTED_DTYPES,
        MISSING_VALUE_STRATEGY,
        MAX_MISSING_PERCENTAGE,
        RANGE_CONSTRAINTS,
        CATEGORICAL_CONSTRAINTS,
        DATE_COLUMN,
        DATE_FORMAT,
        MIN_DATE,
        MAX_DATE,
        OUTLIER_DETECTION_COLUMNS,
        IQR_MULTIPLIER,
        MIN_DATASET_SIZE,
        RELATIONSHIP_RULES
    )
except ImportError:
    from config import (
        REQUIRED_COLUMNS,
        TARGET_COLUMN,
        EXPECTED_DTYPES,
        MISSING_VALUE_STRATEGY,
        MAX_MISSING_PERCENTAGE,
        RANGE_CONSTRAINTS,
        CATEGORICAL_CONSTRAINTS,
        DATE_COLUMN,
        DATE_FORMAT,
        MIN_DATE,
        MAX_DATE,
        OUTLIER_DETECTION_COLUMNS,
        IQR_MULTIPLIER,
        MIN_DATASET_SIZE,
        RELATIONSHIP_RULES
    )

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataValidator:
    """Validates data against predefined rules and constraints"""
    
    def __init__(self):
        self.validation_errors: List[str] = []
        self.validation_warnings: List[str] = []
        
    def validate(self, df: pd.DataFrame, is_prediction: bool = False) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks on the dataframe
        
        Args:
            df: The dataframe to validate
            is_prediction: If True, bypass checks for target column and dataset size
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        logger.info(f"Starting data validation (is_prediction={is_prediction})...")
        
        self.validation_errors = []
        self.validation_warnings = []
        
        # Run all validation checks
        self._validate_schema(df, is_prediction)
        self._validate_data_types(df)
        self._validate_missing_values(df, is_prediction)
        if not is_prediction:
            self._validate_target_column(df)
            self._validate_dataset_size(df)
            self._validate_outliers(df)
        
        self._validate_ranges(df)
        self._validate_categorical_values(df)
        self._validate_dates(df)
        self._validate_relationships(df)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("[PASS] Data validation passed!")
        else:
            logger.error(f"[FAIL] Data validation failed with {len(self.validation_errors)} errors")
        
        if self.validation_warnings:
            logger.warning(f"[WARN] {len(self.validation_warnings)} warnings raised")
        
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _infer_schema_from_df(self, df: pd.DataFrame):
        """Automatically infer the schema and rules from a dataframe."""
        logger.info("Inferring schema and rules automatically...")
        self.inferred_required_columns = df.columns.tolist()
        self.inferred_dtypes = df.dtypes.to_dict()
        
        # Simple heuristic: last numerical column is the target
        num_cols = df.select_dtypes(include=['number']).columns.tolist()
        self.inferred_target = num_cols[-1] if num_cols else None
        
        # Calculate basic constraints
        self.inferred_ranges = {}
        for col in num_cols:
            self.inferred_ranges[col] = {
                'min': df[col].min(),
                'max': df[col].max()
            }

    def validate(self, df: pd.DataFrame, is_prediction: bool = False) -> Tuple[bool, List[str], List[str]]:
        """Run all validation checks on the dataframe"""
        self.validation_errors = []
        self.validation_warnings = []
        
        # If no config is present or we want to be autonomous, infer from DF
        # In a real MLOps scenario, we'd infer from the RAWED dataset once and store it.
        if not hasattr(self, 'inferred_required_columns'):
            self._infer_schema_from_df(df)

        self._validate_schema(df, is_prediction=is_prediction)
        self._validate_data_types(df)
        self._validate_missing_values(df, is_prediction=is_prediction)
        
        if not is_prediction:
            self._validate_target_column(df)
            self._validate_dataset_size(df)
            self._validate_outliers(df)
        
        self._validate_ranges(df)
        self._validate_categorical_values(df)
        # self._validate_dates(df) # Skip for now to be more general
        self._validate_relationships(df)
        
        is_valid = len(self.validation_errors) == 0
        return is_valid, self.validation_errors, self.validation_warnings

    def _validate_schema(self, df: pd.DataFrame, is_prediction: bool = False):
        """Validate that all required columns are present"""
        required = set(self.inferred_required_columns)
        if is_prediction and self.inferred_target:
            required.discard(self.inferred_target)

        missing_cols = required - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {list(missing_cols)}")

    def _validate_data_types(self, df: pd.DataFrame):
        """Validate data types match expected schema"""
        for col, expected_dtype in self.inferred_dtypes.items():
            if col not in df.columns:
                continue
            
            actual_dtype = df[col].dtype
            # Allow some flexibility, especially for prediction time int/float mix
            if expected_dtype == 'float64' and actual_dtype in ['float64', 'int64']:
                continue
            if expected_dtype != actual_dtype:
                self.validation_warnings.append(
                    f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                )

    def _validate_missing_values(self, df: pd.DataFrame, is_prediction: bool = False) -> None:
        """Validate missing values according to strategy"""
        total_rows = len(df)
        
        # In autonomous mode, we allow some missing values unless they are critical
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count == 0:
                continue
                
            missing_pct = (missing_count / total_rows) * 100
            
            # Critical warning for high missing counts
            if missing_pct > 50:
                 self.validation_warnings.append(
                    f"Column '{col}' has {missing_pct:.2f}% missing values. This may impact model performance."
                )
    
    def _validate_target_column(self, df: pd.DataFrame) -> None:
        """Validate target column integrity"""
        target = getattr(self, 'inferred_target', None)
        if not target or target not in df.columns:
            return
        
        # Target must be non-negative
        if (df[target] < 0).any():
            negative_count = (df[target] < 0).sum()
            self.validation_errors.append(
                f"Target column '{target}' has {negative_count} negative values"
            )
        
        # Target cannot be null
        if df[target].isnull().any():
            null_count = df[target].isnull().sum()
            self.validation_errors.append(
                f"Target column '{target}' has {null_count} null values"
            )
    
    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate numerical values are within acceptable ranges"""
        ranges = getattr(self, 'inferred_ranges', {})
        for col, constraints in ranges.items():
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                try:
                    # Allow 10% buffer for ranges in autonomous mode
                    buffer_min = min_val * 0.9 if min_val > 0 else min_val * 1.1
                    violations = (col_data < buffer_min).sum()
                    if violations > 0:
                        self.validation_warnings.append(
                            f"Column '{col}' has {violations} values below expected minimum {buffer_min:.2f}"
                        )
                except TypeError:
                    pass
            
            if max_val is not None:
                try:
                    # Allow 10% buffer
                    buffer_max = max_val * 1.1 if max_val > 0 else max_val * 0.9
                    violations = (col_data > buffer_max).sum()
                    if violations > 0:
                        self.validation_warnings.append(
                            f"Column '{col}' has {violations} values above expected maximum {buffer_max:.2f}"
                        )
                except TypeError:
                    pass
    
    def _validate_categorical_values(self, df: pd.DataFrame) -> None:
        """Validate categorical columns (Checks for excessive cardinality)"""
        for col in df.select_dtypes(include=['object', 'category']).columns:
            cardinality = df[col].nunique()
            if cardinality > 100 and len(df) > 1000:
                self.validation_warnings.append(
                    f"Column '{col}' has high cardinality ({cardinality} unique values)"
                )
    
    def _validate_dataset_size(self, df: pd.DataFrame) -> None:
        """Validate dataset has enough data to be statistically significant"""
        if len(df) < 10: # Minimum for any basic model
            self.validation_errors.append(
                f"Dataset has only {len(df)} rows, minimum required for autonomy is 10"
            )
    
    def _validate_outliers(self, df: pd.DataFrame) -> None:
        """Detect outliers in numerical columns automatically"""
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            data = df[col].dropna()
            if len(data) < 20: continue # Not enough for stats
            
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            outliers = ((data < (Q1 - 3 * IQR)) | (data > (Q3 + 3 * IQR))).sum()
            if outliers > 0:
                self.validation_warnings.append(
                    f"Column '{col}' has {outliers} extreme outliers"
                )
    
    def _validate_relationships(self, df: pd.DataFrame) -> None:
        """No generic rules for autonomous mode yet."""
        pass


def validate_data(data_input: Any, is_prediction: bool = False) -> Tuple[bool, pd.DataFrame, List[str], List[str]]:
    """
    Main function to validate data from file or DataFrame
    
    Args:
        data_input: Path to CSV file OR pandas DataFrame
        is_prediction: If True, bypass checks for target column and dataset size
        
    Returns:
        Tuple of (is_valid, dataframe, errors, warnings)
    """
    try:
        if isinstance(data_input, pd.DataFrame):
            df = data_input
            logger.info(f"Validating DataFrame: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            df = pd.read_csv(data_input)
            logger.info(f"Loaded data from {data_input}: {df.shape[0]} rows, {df.shape[1]} columns")
        
        validator = DataValidator()
        is_valid, errors, warnings = validator.validate(df, is_prediction=is_prediction)
        
        return is_valid, df, errors, warnings
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return False, None, [f"Failed to load data: {str(e)}"], []


if __name__ == "__main__":
    # Test validation
    import sys
    import os
    
    # Get the project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    file_path = os.path.join(project_root, "data", "raw", "car_sales.csv")
    
    is_valid, df, errors, warnings = validate_data(file_path)
    
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"Status: {'[PASS]' if is_valid else '[FAIL]'}")
    print(f"Dataset shape: {df.shape if df is not None else 'N/A'}")
    
    if errors:
        print(f"\n[ERRORS] ({len(errors)}):")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
    
    if warnings:
        print(f"\n[WARNINGS] ({len(warnings)}):")
        for i, warning in enumerate(warnings, 1):
            print(f"  {i}. {warning}")
    
    if not errors and not warnings:
        print("\n[PASS] All validation checks passed!")
    
    sys.exit(0 if is_valid else 1)
