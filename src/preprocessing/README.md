# Preprocessing Pipeline Documentation

## Overview
Deterministic feature engineering pipeline built with `sklearn.Pipeline` and `ColumnTransformer`. No notebooks - pure Python scripts only.

## Pipeline Operations

### 1. **Missing Value Imputation**
- **Numerical features**: Median imputation
- **Categorical features**: Most frequent value
- **Strategy**: Defined in `config.py`

### 2. **Categorical Encoding**
- **Method**: OneHotEncoding
- **Settings**:
  - Drop first category (avoid multicollinearity)
  - Handle unknown categories in test set (ignore)
  - Output: Dense array (not sparse)

### 3. **Date Feature Extraction**
- **Input**: `Latest_Launch` (MM/DD/YYYY format)
- **Extracted Features**:
  - Year
  - Month
  - Day of year
  - Days since epoch (1970-01-01)
- **Output**: 4 numerical features per date column

### 4. **Numerical Scaling**
- **Method**: StandardScaler (z-score normalization)
- **Formula**: `(x - mean) / std`
- Applied to:
  - Original numerical features
  - Extracted date features

## Architecture

```python
ColumnTransformer
├── Numerical Pipeline
│   ├── SimpleImputer (median)
│   └── StandardScaler
├── Categorical Pipeline
│   ├── SimpleImputer (most_frequent)
│   └── OneHotEncoder
└── Date Pipeline
    ├── DateFeatureExtractor (custom)
    └── StandardScaler
```

## Files

### Core Implementation
- **`config.py`** - Configuration and feature definitions
  - Feature lists (numerical, categorical, date)
  - Imputation strategies
  - Scaling methods
  - OneHotEncoder settings

- **`transformers.py`** - Custom transformers
  - `DateFeatureExtractor` - Extract numerical features from dates
  - `DataFrameSelector` - Select columns from DataFrame
  - `DropColumnsTransformer` - Drop specified columns
  - `DebugTransformer` - Debug pipeline execution

- **`pipeline.py`** - Main pipeline implementation
  - `create_preprocessing_pipeline()` - Create pipeline
  - `fit_and_save_pipeline()` - Fit and save
  - `load_pipeline()` - Load from disk
  - `transform_data()` - Transform new data
  - `get_feature_names()` - Get transformed feature names

- **`__init__.py`** - Module exports

## Usage

### Creating and Fitting Pipeline

```python
from src.preprocessing import create_preprocessing_pipeline
from sklearn.model_selection import train_test_split

# Load data
X = df.drop(columns=['Sales_in_thousands'])
y = df['Sales_in_thousands']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create pipeline
pipeline = create_preprocessing_pipeline(verbose=True)

# Fit on training data
pipeline.fit(X_train)

# Transform
X_train_transformed = pipeline.transform(X_train)
X_test_transformed = pipeline.transform(X_test)
```

### Saving and Loading

```python
import joblib

# Save
joblib.dump(pipeline, 'preprocessing_pipeline.pkl')

# Load
pipeline = joblib.load('preprocessing_pipeline.pkl')
```

### Standalone Execution

```bash
# Test pipeline
python src/preprocessing/pipeline.py

# Direct test
python test_pipeline_direct.py
```

## Feature Transformation

### Input Features (15)
**Numerical (11)**:
- `__year_resale_value`
- `Price_in_thousands`
- `Engine_size`
- `Horsepower`
- `Wheelbase`
- `Width`
- `Length`
- `Curb_weight`
- `Fuel_capacity`
- `Fuel_efficiency`
- `Power_perf_factor`

**Categorical (3)**:
- `Manufacturer`
- `Model`
- `Vehicle_type`

**Date (1)**:
- `Latest_Launch`

### Output Features (~198)
- **Numerical**: 11 (scaled)
- **Date extracted**: 4 (year, month, day_of_year, days_since_epoch)
- **Categorical encoded**: ~183 (one-hot encoded Manufacturer, Model, Vehicle_type)

Total features increase due to one-hot encoding of categorical variables.

## Determinism Guarantees

### Why Deterministic?
1. **Fixed random state**: `RANDOM_STATE = 42`
2. **No randomization** in transformers
3. **Same input → Same output** always
4. **Reproducible** across runs

### Verification
```python
# Transform twice
X_test_1 = pipeline.transform(X_test)
X_test_2 = pipeline.transform(X_test.copy())

# Should be identical
assert np.allclose(X_test_1, X_test_2)
```

## Data Quality Guarantees

### No Missing Values
- All missing values imputed
- **Guaranteed**: `np.isnan(output).sum() == 0`

### No Infinite Values
- StandardScaler handles extreme values
- **Guaranteed**: `np.isinf(output).sum() == 0`

### Consistent Shapes
- Same transformation applied to train/test
- Feature count preserved across batches

## Configuration

All settings in `config.py`:

```python
# Imputation
NUMERICAL_IMPUTATION_STRATEGY = 'median'
CATEGORICAL_IMPUTATION_STRATEGY = 'most_frequent'

# Scaling
SCALING_STRATEGY = 'standard'  # 'standard', 'minmax', 'robust'

# Encoding
CATEGORICAL_ENCODING = 'onehot'
ONEHOT_DROP = 'first'
ONEHOT_HANDLE_UNKNOWN = 'ignore'

# Date features
DATE_FORMAT = '%m/%d/%Y'
DATE_FEATURES_TO_EXTRACT = ['year', 'month', 'day_of_year', 'days_since_epoch']
```

## Testing

### Automated Tests
```bash
python test_pipeline_direct.py
```

**Checks**:
1. ✓ Pipeline creation
2. ✓ Fitting on training data
3. ✓ Transformation (train/test)
4. ✓ No NaN values
5. ✓ No Inf values
6. ✓ Determinism (reproducibility)
7. ✓ Shape consistency

### Manual Checks
```python
# Check for data leakage
assert pipeline.fit(X_train) is pipeline  # Fit returns self
X_test_clean = pipeline.transform(X_test)  # Only transform test

# Check feature count
n_features = X_transformed.shape[1]
feature_names = get_feature_names(pipeline)
assert len(feature_names) == n_features
```

## Integration with Training

```python
from src.preprocessing import create_preprocessing_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

# Create full pipeline (preprocessing + model)
full_pipeline = Pipeline([
    ('preprocessing', create_preprocessing_pipeline()),
    ('model', LinearRegression())
])

# Fit end-to-end
full_pipeline.fit(X_train, y_train)

# Predict
predictions = full_pipeline.predict(X_test)
```

## Common Issues

### Issue: Pipeline not loading
**Cause**: Module import path issues with joblib
**Solution**: Ensure transformers.py is importable or recreate pipeline

### Issue: OneHotEncoder unknown categories
**Setting**: `handle_unknown='ignore'`
**Behavior**: Unknown categories in test set get all zeros

### Issue: Feature count mismatch
**Cause**: Different categories in train/test
**Solution**: OneHotEncoder with `drop='first'` and `handle_unknown='ignore'`

## Next Steps

1. ✓ Pipeline created and tested
2. → Integrate with model training
3. → Version pipeline with DVC
4. → Add to CI/CD for testing
5. → Monitor for data drift

---

**Remember**: The pipeline is DETERMINISTIC - same input always produces same output. This is crucial for reproducibility in MLOps!
