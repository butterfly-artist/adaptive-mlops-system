# Full Training Pipeline Documentation

## Overview
Complete end-to-end training pipeline that **combines preprocessing and models** into a single, deployable unit.

## Why Combine Preprocessing + Model?

### Production Benefits

1. **No Data Leakage**
   - Preprocessing fitted on training data only
   - Same transformations applied to test and production data
   - Impossible to accidentally use test statistics

2. **Reproducibility**
   - Single pipeline ensures identical preprocessing
   - No manual steps that could be forgotten
   - Version control for entire prediction workflow

3. **Safe Deployment**
   - Preprocessing and model travel together
   - One `.pkl` file contains everything
   - No risk of version mismatch

4. **Simplified Production Code**
   - No separate preprocessing step
   - Single `predict()` call handles everything
   - Less code = fewer bugs

## Pipeline Structure

```
Pipeline
│
├── Step 1: preprocessing (ColumnTransformer)
│   ├── Numerical: Imputation → Scaling
│   ├── Categorical: Imputation → OneHotEncoding
│   └── Date: Feature Extraction → Scaling
│
└── Step 2: model (ML Estimator)
    └── LinearRegression / Ridge / Lasso / RandomForest / GradientBoosting
```

## Implementation

### Core Function: `create_full_pipeline`

```python
from src.training import create_full_pipeline
from sklearn.linear_model import Ridge

# Create full pipeline
pipeline = create_full_pipeline(
    model=Ridge(alpha=1.0, random_state=42),
    model_name='ridge'
)

# Fit on RAW data (preprocessing happens automatically)
pipeline.fit(X_train, y_train)

# Predict on RAW data (preprocessing happens automatically)
predictions = pipeline.predict(X_test)
```

### Trained Models

All models trained with full pipelines:

1. **Linear Regression**
   - Basic linear model
   - No regularization
   - Fast training

2. **Ridge Regression**
   - L2 regularization (alpha=1.0)
   - Prevents overfitting
   - Good for multicollinearity

3. **Lasso Regression**
   - L1 regularization (alpha=1.0)
   - Feature selection
   - Sparse solutions

4. **Random Forest**
   - 100 trees, max_depth=10
   - Non-linear relationships
   - Feature importance

5. **Gradient Boosting**
   - 100 estimators, learning_rate=0.1
   - Sequential ensemble
   - High accuracy potential

## Results

### Performance (Test Set)

| Model | Test RMSE | Test MAE | Test R² | vs Baseline |
|-------|-----------|----------|---------|-------------|
| **Lasso** | **77.68** | **45.26** | **0.381** | **-21.3%** ✓ |
| Linear Regression | 77.96 | 49.98 | 0.376 | -21.1% ✓ |
| Ridge | 78.82 | 48.88 | 0.362 | -20.2% ✓ |
| Random Forest | 83.19 | 40.84 | 0.290 | -15.8% ✓ |
| Gradient Boosting | 87.48 | 45.72 | 0.214 | -11.5% ✓ |
| **Baseline (Mean)** | **98.80** | **53.01** | **-0.002** | Reference |

### Key Findings

1. **All models beat baseline** ✓
2. **Best model**: Lasso (RMSE = 77.68)
3. **Best improvement**: 21.3% over baseline
4. **Best MAE**: Random Forest (40.84)
5. **All models have positive R²** (baseline was negative)

### Model Selection

**Winner**: **Lasso Regression**
- Test RMSE: 77.68
- 21.3% improvement over baseline
- Good balance: not overfitting, decent performance
- Simple, interpretable model

## Files Created

### Training Module

- **`src/training/full_pipeline.py`** - Main implementation
  - `create_full_pipeline()` - Build combined pipeline
  - `train_and_evaluate_model()` - Train single model
  - `train_all_models()` - Train all models
  - `get_ml_models()` - Get model configurations
  - Standalone executable

- **`src/training/__init__.py`** - Module exports (updated)

### Results

- **`all_models_results.json`** - All model metrics
- **`models/lasso_pipeline.pkl`** - Best model pipeline (9.3 KB)
- **`models/lasso_metrics.json`** - Best model metrics

### Analysis

- **`simple_analysis.py`** - Results visualization

## Usage Examples

### Training Full Pipeline

```python
from src.training import create_full_pipeline, train_and_evaluate_model
from sklearn.ensemble import RandomForestRegressor

# Create pipeline
model = RandomForestRegressor(n_estimators=100, random_state=42)
pipeline = create_full_pipeline(model, model_name='random_forest')

# Train on RAW data
result = train_and_evaluate_model(
    pipeline,
    X_train, y_train,
    X_test, y_test,
    model_name='RandomForest'
)

# Access results
print(f"Test RMSE: {result['test_metrics']['rmse']}")
```

### Training All Models

```python
from src.training import train_all_models

# Train all at once
results = train_all_models(X_train, y_train, X_test, y_test)

# Compare
for model_name, result in results.items():
    print(f"{model_name}: RMSE = {result['test_metrics']['rmse']:.2f}")
```

### Standalone Execution

```bash
# Train all models with full pipelines
python src/training/full_pipeline.py

# Analyze results
python simple_analysis.py
```

### Production Usage

```python
import joblib

# Load complete pipeline
pipeline = joblib.load('models/lasso_pipeline.pkl')

# Predict on RAW data (preprocessing happens automatically!)
predictions = pipeline.predict(X_new)

# That's it! No manual preprocessing needed
```

## Data Flow

```
Raw Data (X_train)
    ↓
[Fit Pipeline]
    ↓
Preprocessing fitted on train data
    ↓
Transformed features
    ↓
Model fitted on transformed features
    ↓
Pipeline ready!

Raw Data (X_test)
    ↓
[Predict with Pipeline]
    ↓
Preprocessing applied (using train statistics)
    ↓
Transformed features
    ↓
Model predicts on transformed features
    ↓
Predictions!
```

## Advantages Over Separate Steps

### ❌ Bad Practice (Separate Steps)

```python
# Fit preprocessing
preprocessor.fit(X_train)

# Transform train
X_train_transformed = preprocessor.transform(X_train)

# Fit model
model.fit(X_train_transformed, y_train)

# In production - easy to make mistakes!
X_new_transformed = preprocessor.transform(X_new)  # What if you forget this?
predictions = model.predict(X_new_transformed)

# Risk: Model gets raw data instead of transformed data
# Result: Bad predictions!
```

### ✓ Good Practice (Combined Pipeline)

```python
# Fit full pipeline
pipeline.fit(X_train, y_train)

# In production - impossible to make mistakes
predictions = pipeline.predict(X_new)

# Preprocessing happens automatically!
# Risk: Zero - pipeline handles everything
```

## Overfitting Analysis

| Model | Train RMSE | Test RMSE | Gap |
|-------|-----------|-----------|-----|
| Linear Regression | 0.00 | 77.96 | 77.96 (perfect fit - many features) |
| Ridge | 17.65 | 78.82 | 61.17 |
| Lasso | 36.93 | 77.68 | 40.75 ✓ Good |
| Random Forest | 26.99 | 83.19 | 56.20 |
| Gradient Boosting | 1.85 | 87.48 | 85.63 (overfitting!) |

**Best generalization**: Lasso (smallest train-test gap relative to performance)

## Integration with MLOps

### Version Control

```bash
# Version the entire pipeline
dvc add models/lasso_pipeline.pkl
git add models/lasso_pipeline.pkl.dvc
git commit -m "Add Lasso pipeline v1.0"
```

### CI/CD

```yaml
# Test pipeline
- name: Test Pipeline
  run: |
    python -c "
    import joblib
    pipeline = joblib.load('models/lasso_pipeline.pkl')
    assert hasattr(pipeline, 'predict')
    "
```

### Monitoring

```python
# Log predictions
import mlflow

mlflow.sklearn.log_model(pipeline, "lasso_pipeline")
```

## Next Steps

1. ✓ Full pipelines created and trained
2. ✓ Best model selected and saved
3. → Deploy best pipeline to production
4. → Set up model serving (API)
5. → Monitor predictions and retrain triggers
6. → A/B test against baseline

---

**Remember**: Never separate preprocessing from model in production. They must travel together as a single pipeline!
