"""
Preprocessing Configuration
Defines feature types and transformation strategies
"""

from typing import List

# ============================================================
# FEATURE DEFINITIONS
# ============================================================

# Target column
TARGET_COLUMN = 'Sales_in_thousands'

# Numerical features (excluding target)
NUMERICAL_FEATURES: List[str] = [
    '__year_resale_value',
    'Price_in_thousands',
    'Engine_size',
    'Horsepower',
    'Wheelbase',
    'Width',
    'Length',
    'Curb_weight',
    'Fuel_capacity',
    'Fuel_efficiency',
    'Power_perf_factor'
]

# Categorical features
CATEGORICAL_FEATURES: List[str] = [
    'Manufacturer',
    'Model',
    'Vehicle_type'
]

# Date features (will be converted to numerical)
DATE_FEATURES: List[str] = [
    'Latest_Launch'
]

# All feature columns (excluding target)
ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES + DATE_FEATURES

# ============================================================
# TRANSFORMATION STRATEGIES
# ============================================================

# Missing value imputation strategies
NUMERICAL_IMPUTATION_STRATEGY = 'median'  # 'mean', 'median', 'most_frequent'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'  # 'most_frequent', 'constant'

# Scaling strategy for numerical features
SCALING_STRATEGY = 'standard'  # 'standard', 'minmax', 'robust'

# Categorical encoding strategy
CATEGORICAL_ENCODING = 'onehot'  # 'onehot', 'label', 'target'

# OneHotEncoder settings
ONEHOT_DROP = 'first'  # Drop first category to avoid multicollinearity
ONEHOT_HANDLE_UNKNOWN = 'ignore'  # Ignore unknown categories in test set

# Date feature extraction
DATE_FORMAT = '%m/%d/%Y'
DATE_FEATURES_TO_EXTRACT = [
    'year',
    'month',
    'day_of_year',
    'days_since_epoch'
]

# ============================================================
# FEATURE ENGINEERING OPTIONS
# ============================================================

# Create interaction features
CREATE_INTERACTIONS = False

# Create polynomial features
CREATE_POLYNOMIAL = False
POLYNOMIAL_DEGREE = 2

# Log transform skewed features
LOG_TRANSFORM_FEATURES: List[str] = []  # Can add features like 'Sales_in_thousands'

# Feature selection
APPLY_FEATURE_SELECTION = False
FEATURE_SELECTION_METHOD = 'variance'  # 'variance', 'correlation', 'model_based'
VARIANCE_THRESHOLD = 0.0

# ============================================================
# PIPELINE SETTINGS
# ============================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Verbosity
VERBOSE = True

# Number of jobs for parallel processing (-1 = all cores)
N_JOBS = -1
