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
        
    def validate(self, df: pd.DataFrame) -> Tuple[bool, List[str], List[str]]:
        """
        Run all validation checks on the dataframe
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        logger.info("Starting data validation...")
        
        self.validation_errors = []
        self.validation_warnings = []
        
        # Run all validation checks
        self._validate_schema(df)
        self._validate_data_types(df)
        self._validate_missing_values(df)
        self._validate_target_column(df)
        self._validate_ranges(df)
        self._validate_categorical_values(df)
        self._validate_dates(df)
        self._validate_dataset_size(df)
        self._validate_outliers(df)
        self._validate_relationships(df)
        
        is_valid = len(self.validation_errors) == 0
        
        if is_valid:
            logger.info("[PASS] Data validation passed!")
        else:
            logger.error(f"[FAIL] Data validation failed with {len(self.validation_errors)} errors")
        
        if self.validation_warnings:
            logger.warning(f"[WARN] {len(self.validation_warnings)} warnings raised")
        
        return is_valid, self.validation_errors, self.validation_warnings
    
    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that all required columns are present"""
        missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(
                f"Missing required columns: {missing_cols}"
            )
        
        extra_cols = set(df.columns) - set(REQUIRED_COLUMNS)
        if extra_cols:
            self.validation_warnings.append(
                f"Extra columns found: {extra_cols}"
            )
    
    def _validate_data_types(self, df: pd.DataFrame) -> None:
        """Validate data types match expected schema"""
        for col, expected_dtype in EXPECTED_DTYPES.items():
            if col not in df.columns:
                continue
                
            actual_dtype = str(df[col].dtype)
            if actual_dtype != expected_dtype:
                self.validation_errors.append(
                    f"Column '{col}' has dtype '{actual_dtype}', expected '{expected_dtype}'"
                )
    
    def _validate_missing_values(self, df: pd.DataFrame) -> None:
        """Validate missing values according to strategy"""
        total_rows = len(df)
        
        for col, strategy in MISSING_VALUE_STRATEGY.items():
            if col not in df.columns:
                continue
            
            missing_count = df[col].isnull().sum()
            missing_pct = (missing_count / total_rows) * 100
            
            if strategy == 'reject' and missing_count > 0:
                self.validation_errors.append(
                    f"Column '{col}' has {missing_count} missing values ({missing_pct:.2f}%), "
                    f"but strategy is 'reject'"
                )
            
            if col in MAX_MISSING_PERCENTAGE:
                max_allowed = MAX_MISSING_PERCENTAGE[col]
                if missing_pct > max_allowed:
                    self.validation_errors.append(
                        f"Column '{col}' has {missing_pct:.2f}% missing values, "
                        f"exceeds maximum allowed {max_allowed}%"
                    )
    
    def _validate_target_column(self, df: pd.DataFrame) -> None:
        """Validate target column integrity"""
        if TARGET_COLUMN not in df.columns:
            self.validation_errors.append(f"Target column '{TARGET_COLUMN}' not found")
            return
        
        # Target must be non-negative
        if (df[TARGET_COLUMN] < 0).any():
            negative_count = (df[TARGET_COLUMN] < 0).sum()
            self.validation_errors.append(
                f"Target column '{TARGET_COLUMN}' has {negative_count} negative values"
            )
        
        # Target cannot be null
        if df[TARGET_COLUMN].isnull().any():
            null_count = df[TARGET_COLUMN].isnull().sum()
            self.validation_errors.append(
                f"Target column '{TARGET_COLUMN}' has {null_count} null values"
            )
    
    def _validate_ranges(self, df: pd.DataFrame) -> None:
        """Validate numerical values are within acceptable ranges"""
        for col, constraints in RANGE_CONSTRAINTS.items():
            if col not in df.columns:
                continue
            
            col_data = df[col].dropna()
            
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None:
                violations = (col_data < min_val).sum()
                if violations > 0:
                    self.validation_errors.append(
                        f"Column '{col}' has {violations} values below minimum {min_val}"
                    )
            
            if max_val is not None:
                violations = (col_data > max_val).sum()
                if violations > 0:
                    self.validation_errors.append(
                        f"Column '{col}' has {violations} values above maximum {max_val}"
                    )
    
    def _validate_categorical_values(self, df: pd.DataFrame) -> None:
        """Validate categorical columns have expected values"""
        for col, allowed_values in CATEGORICAL_CONSTRAINTS.items():
            if col not in df.columns:
                continue
            
            unique_values = set(df[col].dropna().unique())
            invalid_values = unique_values - set(allowed_values)
            
            if invalid_values:
                self.validation_errors.append(
                    f"Column '{col}' has invalid values: {invalid_values}. "
                    f"Allowed values: {allowed_values}"
                )
    
    def _validate_dates(self, df: pd.DataFrame) -> None:
        """Validate date columns"""
        if DATE_COLUMN not in df.columns:
            return
        
        try:
            dates = pd.to_datetime(df[DATE_COLUMN], format=DATE_FORMAT, errors='coerce')
            
            # Check for unparseable dates
            invalid_dates = dates.isnull().sum()
            if invalid_dates > 0:
                self.validation_errors.append(
                    f"Column '{DATE_COLUMN}' has {invalid_dates} invalid date formats"
                )
            
            # Check for future dates
            valid_dates = dates.dropna()
            future_dates = (valid_dates > MAX_DATE).sum()
            if future_dates > 0:
                self.validation_errors.append(
                    f"Column '{DATE_COLUMN}' has {future_dates} future dates"
                )
            
            # Check for too old dates
            old_dates = (valid_dates < MIN_DATE).sum()
            if old_dates > 0:
                self.validation_warnings.append(
                    f"Column '{DATE_COLUMN}' has {old_dates} dates before {MIN_DATE}"
                )
                
        except Exception as e:
            self.validation_errors.append(
                f"Error parsing dates in '{DATE_COLUMN}': {str(e)}"
            )
    
    def _validate_dataset_size(self, df: pd.DataFrame) -> None:
        """Validate dataset has minimum required size"""
        if len(df) < MIN_DATASET_SIZE:
            self.validation_errors.append(
                f"Dataset has only {len(df)} rows, minimum required is {MIN_DATASET_SIZE}"
            )
    
    def _validate_outliers(self, df: pd.DataFrame) -> None:
        """Detect and warn about outliers using IQR method"""
        for col in OUTLIER_DETECTION_COLUMNS:
            if col not in df.columns:
                continue
            
            data = df[col].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - IQR_MULTIPLIER * IQR
            upper_bound = Q3 + IQR_MULTIPLIER * IQR
            
            outliers = ((data < lower_bound) | (data > upper_bound)).sum()
            
            if outliers > 0:
                outlier_pct = (outliers / len(data)) * 100
                self.validation_warnings.append(
                    f"Column '{col}' has {outliers} ({outlier_pct:.2f}%) potential outliers"
                )
    
    def _validate_relationships(self, df: pd.DataFrame) -> None:
        """Validate business logic relationships between columns"""
        
        # Rule: Resale value should be less than original price
        if '__year_resale_value' in df.columns and 'Price_in_thousands' in df.columns:
            comparison_df = df.dropna(subset=['__year_resale_value', 'Price_in_thousands'])
            violations = (comparison_df['__year_resale_value'] > comparison_df['Price_in_thousands']).sum()
            
            if violations > 0:
                self.validation_warnings.append(
                    f"Found {violations} cases where resale value exceeds original price"
                )
        
        # Rule: Power performance factor should be positive when present
        if 'Power_perf_factor' in df.columns:
            negative_ppf = (df['Power_perf_factor'].dropna() <= 0).sum()
            if negative_ppf > 0:
                self.validation_errors.append(
                    f"Power performance factor has {negative_ppf} non-positive values"
                )


def validate_data(data_input: Any) -> Tuple[bool, pd.DataFrame, List[str], List[str]]:
    """
    Main function to validate data from file or DataFrame
    
    Args:
        data_input: Path to CSV file OR pandas DataFrame
        
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
        is_valid, errors, warnings = validator.validate(df)
        
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
