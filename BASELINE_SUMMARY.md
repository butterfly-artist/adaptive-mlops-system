# Baseline Models - Task Complete ✓

## Task Summary
Implemented mandatory baseline models (Mean and Median predictors) before building ML models. These establish reference performance and lower bounds.

---

## 🎯 Implemented Models

### 1. Mean Predictor
- **Type**: Constant predictor
- **Strategy**: Always predicts training mean
- **Value**: **52.05** thousand units

**Test Performance**:
- RMSE: **98.80** ← LOWER BOUND
- MAE: 53.01
- R²: -0.0022

### 2. Median Predictor
- **Type**: Constant predictor  
- **Strategy**: Always predicts training median
- **Value**: **30.70** thousand units

**Test Performance**:
- RMSE: 102.06
- MAE: **46.41** ← LOWER BOUND
- R²: -0.0694

---

## 📊 Results Summary

### Best Baseline
**Mean Predictor** (by RMSE)

### Lower Bounds Established

```
ML models MUST beat these metrics:
✓ RMSE < 98.80
✓ MAE < 46.41  
✓ R² > -0.0022
```

### Key Insights

1. **Distribution is right-skewed**
   - Mean (52.05) > Median (30.70)
   - Most vehicles have lower sales
   - Few high-sales vehicles pull mean up

2. **Median is more robust**
   - Better MAE (46.41 vs 53.01)
   - Better MAPE (369.70% vs 647.15%)
   - Less sensitive to outliers

3. **Both models are weak** (intentionally!)
   - Negative R² on test set
   - High prediction errors
   - No feature information used

---

## 📁 Files Created

### Core Implementation
- **`src/training/baseline_models.py`**
  - MeanPredictor class
  - MedianPredictor class
  - Metrics calculation
  - Training & evaluation pipeline
  - Standalone executable

- **`src/training/__init__.py`**
  - Module exports

### Documentation
- **`src/training/BASELINE_README.md`**
  - Complete documentation
  - Usage examples
  - Interpretation guide

### Results
- **`baseline_results.json`**
  - Full metrics (train + test)
  - Model parameters
  - Comparison data

- **`baseline_lower_bounds.json`**
  - Minimum thresholds for ML models
  - Reference predictions
  - Best baseline identifier

### Analysis
- **`analyze_baselines.py`**
  - Results visualization
  - Performance comparison
  - Summary generation

---

## 💡 Why Baselines Matter

### 1. Reference Performance
Provides objective comparison point for ML models

### 2. Complexity Justification
If ML can't beat constant predictors → not learning anything useful

### 3. Sanity Check
Prevents accepting poorly performing models

### 4. Lower Bound
Clear minimum bar for model acceptance

### 5. Feature Value Proof
ML models beating baselines proves features contain predictive information

---

## 🔄 Usage

### Quick Test
```bash
python src/training/baseline_models.py
```

### Analysis
```bash
python analyze_baselines.py
```

### In Code
```python
from src.training import MeanPredictor, MedianPredictor

# Train
mean_model = MeanPredictor()
mean_model.fit(X_train, y_train)

# Predict
predictions = mean_model.predict(X_test)
```

---

## 📈 What's Next

1. ✓ Baseline models complete
2. → Preprocess features
3. → Train ML models (Linear Regression, Random Forest, XGBoost)
4. → **Compare ML models vs baselines**
5. → Only accept models that **significantly** beat baselines
6. → Use in drift detection & monitoring

---

## 🎓 Key Learnings

### Mean vs Median Choice

**Our Dataset**:
- Right-skewed distribution (confirmed)
- Presence of outliers (validation found them)
- Mean predictor wins on RMSE
- Median predictor wins on MAE

**Recommendation**:
Use **Mean Predictor** as primary baseline (better RMSE), but also track MAE where Median excels.

### Expected ML Improvement

For useful ML models, we expect:
- RMSE reduction: **> 20%** (below ~79)
- MAE reduction: **> 20%** (below ~37)
- R² improvement: **> 0.5** (positive, meaningful)

---

## ✅ Success Criteria Met

- [x] Mean predictor implemented ✓
- [x] Median predictor implemented ✓
- [x] Both models tested on train/test splits ✓
- [x] Metrics calculated (MSE, RMSE, MAE, R², MAPE) ✓
- [x] Reference performance established ✓
- [x] Lower bounds documented ✓
- [x] Results saved for comparison ✓
- [x] Code is executable, not notes ✓

---

**Status**: ✅ BASELINE MODELS COMPLETE

These simple predictors establish the **floor** that all ML models must exceed. Without beating these baselines, an ML model adds no value over simply predicting a constant!

Ready to build ML models with clear success criteria! 🚀
