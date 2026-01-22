"""
Data Validation Configuration
Defines all validation rules and schema for car sales data
"""

from typing import List, Dict, Any
from datetime import datetime

# ============================================================
# SCHEMA DEFINITION
# ============================================================

REQUIRED_COLUMNS: List[str] = [
    'Manufacturer',
    'Model',
    'Sales_in_thousands',
    '__year_resale_value',
    'Vehicle_type',
    'Price_in_thousands',
    'Engine_size',
    'Horsepower',
    'Wheelbase',
    'Width',
    'Length',
    'Curb_weight',
    'Fuel_capacity',
    'Fuel_efficiency',
    'Latest_Launch',
    'Power_perf_factor'
]

TARGET_COLUMN: str = 'Sales_in_thousands'

# ============================================================
# DATA TYPE DEFINITIONS
# ============================================================

EXPECTED_DTYPES: Dict[str, str] = {
    'Manufacturer': 'object',
    'Model': 'object',
    'Sales_in_thousands': 'float64',
    '__year_resale_value': 'float64',
    'Vehicle_type': 'object',
    'Price_in_thousands': 'float64',
    'Engine_size': 'float64',
    'Horsepower': 'float64',
    'Wheelbase': 'float64',
    'Width': 'float64',
    'Length': 'float64',
    'Curb_weight': 'float64',
    'Fuel_capacity': 'float64',
    'Fuel_efficiency': 'float64',
    'Latest_Launch': 'object',
    'Power_perf_factor': 'float64'
}

# ============================================================
# MISSING VALUE STRATEGY
# ============================================================

MISSING_VALUE_STRATEGY: Dict[str, str] = {
    # NO missing values allowed for critical columns
    'Manufacturer': 'reject',
    'Model': 'reject',
    'Sales_in_thousands': 'reject',
    'Vehicle_type': 'reject',
    'Latest_Launch': 'reject',
    
    # Impute with median for numerical features
    'Price_in_thousands': 'median',
    'Engine_size': 'median',
    'Horsepower': 'median',
    'Wheelbase': 'median',
    'Width': 'median',
    'Length': 'median',
    'Curb_weight': 'median',
    'Fuel_capacity': 'median',
    'Fuel_efficiency': 'median',
    'Power_perf_factor': 'median',
    
    # Special case: resale value can have more missing values
    '__year_resale_value': 'median'
}

# Maximum allowed missing percentage per column before rejection
MAX_MISSING_PERCENTAGE: Dict[str, float] = {
    '__year_resale_value': 30.0,  # Can have up to 30% missing
    'Price_in_thousands': 5.0,
    'Engine_size': 5.0,
    'Horsepower': 5.0,
    'Wheelbase': 5.0,
    'Width': 5.0,
    'Length': 5.0,
    'Curb_weight': 5.0,
    'Fuel_capacity': 5.0,
    'Fuel_efficiency': 5.0,
    'Power_perf_factor': 5.0
}

# ============================================================
# VALIDATION RULES
# ============================================================

# Numerical Range Constraints
RANGE_CONSTRAINTS: Dict[str, Dict[str, float]] = {
    'Sales_in_thousands': {'min': 0.0, 'max': 1000.0},
    'Price_in_thousands': {'min': 0.0, 'max': 200.0},
    '__year_resale_value': {'min': 0.0, 'max': 100.0},
    'Engine_size': {'min': 0.5, 'max': 10.0},
    'Horsepower': {'min': 50.0, 'max': 600.0},
    'Wheelbase': {'min': 80.0, 'max': 200.0},
    'Width': {'min': 50.0, 'max': 100.0},
    'Length': {'min': 100.0, 'max': 300.0},
    'Curb_weight': {'min': 1.0, 'max': 10.0},
    'Fuel_capacity': {'min': 5.0, 'max': 50.0},
    'Fuel_efficiency': {'min': 10.0, 'max': 60.0},
    'Power_perf_factor': {'min': 0.0, 'max': 300.0}
}

# Categorical Value Constraints
CATEGORICAL_CONSTRAINTS: Dict[str, List[str]] = {
    'Vehicle_type': ['Passenger', 'Car']  # Based on EDA
}

# Date Constraints
DATE_COLUMN: str = 'Latest_Launch'
DATE_FORMAT: str = '%m/%d/%Y'
MIN_DATE: datetime = datetime(2000, 1, 1)
MAX_DATE: datetime = datetime.now()  # No future dates allowed

# ============================================================
# STATISTICAL VALIDATION RULES
# ============================================================

# Outlier detection using IQR method
OUTLIER_DETECTION_COLUMNS: List[str] = [
    'Sales_in_thousands',
    'Price_in_thousands',
    'Horsepower',
    'Power_perf_factor'
]

IQR_MULTIPLIER: float = 3.0  # Conservative outlier detection

# Minimum dataset size
MIN_DATASET_SIZE: int = 50

# ============================================================
# FEATURE RELATIONSHIPS (Business Logic Validation)
# ============================================================

# Power_perf_factor should correlate with Horsepower and Price
# If Price > 0 and Horsepower > 0, then Power_perf_factor should be calculable
RELATIONSHIP_RULES: List[str] = [
    'resale_value_less_than_price',  # Resale value should be less than original price
    'power_perf_factor_consistency'  # Power performance factor should be positive
]
