# Baseline Models for Car Sales Prediction

## Overview
Before building complex ML models, we establish **baseline performance metrics** using simple statistical predictors. These baselines:
- Provide **reference performance** for comparison
- Establish the **lower bound** that ML models must beat
- Require **no training** and **no feature engineering**
- Serve as sanity checks for ML model performance

## Implemented Baseline Models

### 1. Mean Predictor
**Strategy**: Predicts the training set mean for all samples

**Characteristics**:
- Simplest possible predictor
- Minimizes Mean Squared Error (MSE) in expectation
- R² = 0 by definition on training set

**Results**:
- **Prediction Value**: 52.05 thousand units
- **Test RMSE**: 98.80
- **Test MAE**: 53.01
- **Test R²**: -0.0022

### 2. Median Predictor
**Strategy**: Predicts the training set median for all samples

**Characteristics**:
- More robust to outliers than mean
- Minimizes Mean Absolute Error (MAE) in expectation
- Better when target distribution is skewed

**Results**:
- **Prediction Value**: 30.70 thousand units
- **Test RMSE**: 102.06
- **Test MAE**: 46.41 ✓ [BEST]
- **Test R²**: -0.0694

## Performance Comparison

### Test Set Metrics

| Metric | Mean Predictor | Median Predictor | Winner |
|--------|---------------|------------------|---------|
| **RMSE** | 98.80 | 102.06 | Mean ✓ |
| **MAE** | 53.01 | 46.41 | Median ✓ |
| **R²** | -0.0022 | -0.0694 | Mean ✓ |
| **MAPE** | 647.15% | 369.70% | Median ✓ |

### Key Findings

1. **Mean Predictor** is better overall (by RMSE)
2. **Median Predictor** performs better on MAE and MAPE (more robust to outliers)
3. Both have **negative R²** on test set (expected for constant predictors)
4. Large MAPE values indicate high relative errors (many small sales values)

## Lower Bounds Established

**Any ML model MUST beat these metrics to be useful:**

```json
{
  "min_rmse_to_beat": 98.80,
  "min_mae_to_beat": 46.41,
  "min_r2_to_beat": -0.0022
}
```

### Why These Are Important

- **Complexity Justification**: If an ML model can't beat a constant predictor, it's not learning anything useful
- **Feature Value**: Shows that features contain predictive information beyond central tendency
- **Sanity Check**: Prevents accepting poorly performing models
- **Benchmark**: Clear target for minimum acceptable performance

## Files

### Implementation
- **`src/training/baseline_models.py`** - Baseline model classes and evaluation
  - `MeanPredictor` class
  - `MedianPredictor` class
  - `calculate_metrics()` function
  - `train_and_evaluate_baselines()` function

### Results
- **`baseline_results.json`** - Full baseline metrics (train and test)
- **`baseline_lower_bounds.json`** - Minimum thresholds for ML models

### Analysis
- **`analyze_baselines.py`** - Results visualization and analysis

## Usage

### Training Baselines

```python
from src.training import train_and_evaluate_baselines
from sklearn.model_selection import train_test_split

# Load data
X = df.drop(columns=['Sales_in_thousands'])
y = df['Sales_in_thousands']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train and evaluate
results = train_and_evaluate_baselines(X_train, y_train, X_test, y_test)
```

### Using Predictors

```python
from src.training import MeanPredictor, MedianPredictor

# Mean predictor
mean_model = MeanPredictor()
mean_model.fit(X_train, y_train)
predictions = mean_model.predict(X_test)

# Median predictor
median_model = MedianPredictor()
median_model.fit(X_train, y_train)
predictions = median_model.predict(X_test)
```

### Standalone Execution

```bash
# Run baseline evaluation
python src/training/baseline_models.py

# Analyze results
python analyze_baselines.py
```

## Interpretation

### Why R² is Negative?

- R² compares model performance to mean baseline
- For constant predictors on test set, R² ≈ 0 or slightly negative
- Negative R² means model is slightly worse than training mean
- This is **expected** and **normal** for baseline models

### Mean vs Median: Which to Use?

**Use Mean Predictor when**:
- Target distribution is approximately normal
- You care about MSE/RMSE
- Outliers are not a major concern

**Use Median Predictor when**:
- Target distribution is skewed (like car sales)
- You care about MAE
- Dataset has significant outliers
- Robustness is important

### Current Dataset Insights

- **Mean (52.05) > Median (30.70)** → Right-skewed distribution
- Few high-sales vehicles pull the mean up
- Most vehicles have sales below 30.70 thousand units
- Median predictor is more robust to outliers (better MAE)

## Next Steps

1. ✓ Baselines established
2. → Build ML models (Linear Regression, Random Forest, etc.)
3. → Compare ML models against these baselines
4. → Only deploy ML models that **significantly** beat baselines
5. → Use baseline metrics in monitoring (drift detection)

---

**Remember**: If your fancy ML model can't beat a constant predictor, something is wrong! These baselines are the **minimum bar** for model acceptance.
