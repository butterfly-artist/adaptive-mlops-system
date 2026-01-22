# Data Understanding & Validation - COMPLETED ✓

## Task Summary
Performed EDA to define validation rules (not fancy plots) and implemented comprehensive data validation as executable code.

## Dataset Overview
- **File**: `data/raw/car_sales.csv`
- **Size**: 157 rows × 16 columns
- **Target**: `Sales_in_thousands`

## Columns Breakdown

### Categorical (4 columns)
- `Manufacturer` - Car manufacturer name
- `Model` - Car model name  
- `Vehicle_type` - Type of vehicle (Passenger/Car)
- `Latest_Launch` - Launch date (MM/DD/YYYY format)

### Numerical (12 columns)
- `Sales_in_thousands` - **TARGET VARIABLE** (must be ≥ 0)
- `__year_resale_value` - Resale value
- `Price_in_thousands` - Original price (must be > 0)
- `Engine_size` - Engine size in liters
- `Horsepower` - Engine horsepower
- `Wheelbase` - Wheelbase measurement
- `Width` - Vehicle width
- `Length` - Vehicle length
- `Curb_weight` - Vehicle weight
- `Fuel_capacity` - Fuel tank capacity
- `Fuel_efficiency` - MPG rating
- `Power_perf_factor` - Calculated performance metric

## Implemented Validation Rules

### 1. Schema Validation
✓ All 16 required columns must be present
✓ Data types must match expected schema

### 2. Missing Value Strategy
**REJECT (0% allowed)**:
- Manufacturer
- Model  
- Sales_in_thousands (TARGET)
- Vehicle_type
- Latest_Launch

**MEDIAN IMPUTATION** (with thresholds):
- Resale value: max 30% missing
- All other numerical features: max 5% missing

### 3. Range Constraints
```
Sales_in_thousands:     [0, 1000]
Price_in_thousands:     [0, 200]
__year_resale_value:    [0, 100]
Engine_size:            [0.5, 10.0]
Horsepower:             [50, 600]
Wheelbase:              [80, 200]
Width:                  [50, 100]
Length:                 [100, 300]
Curb_weight:            [1.0, 10.0]
Fuel_capacity:          [5.0, 50.0]
Fuel_efficiency:        [10, 60]
Power_perf_factor:      [0, 300]
```

### 4. Categorical Constraints
- `Vehicle_type` must be in: ['Passenger', 'Car']

### 5. Date Validation
- Format: MM/DD/YYYY
- **NO FUTURE DATES** allowed
- Minimum date: 2000-01-01

### 6. Business Logic Rules
✓ Resale value < Original price
✓ Power performance factor > 0 when present
✓ Sales must be non-negative

### 7. Statistical Validation
- Minimum dataset size: 50 rows
- Outlier detection using IQR method (multiplier: 3.0)
- Monitored columns: Sales, Price, Horsepower, Power_perf_factor

## Files Created

### `src/data_validation/config.py`
- All validation configuration
- Schema definitions
- Range constraints
- Missing value strategies

### `src/data_validation/validator.py`
- Core validation engine
- DataValidator class with 10 validation checks
- Standalone executable

### `src/data_validation/__init__.py`
- Module initialization
- Exports for easy import

### `src/data_validation/README.md`
- Comprehensive documentation
- Usage examples
- Design decisions

### `src/data_validation/RULES.py`
- Quick reference guide
- All rules in concise format

## Validation Results (Current Data)

**Status**: ✓ PASS

**Errors**: 0

**Warnings**: 4
1. Sales_in_thousands: 5 (3.18%) potential outliers
2. Price_in_thousands: 3 (1.94%) potential outliers  
3. Horsepower: 1 (0.64%) potential outliers
4. Power_perf_factor: 1 (0.65%) potential outliers

## Usage

### As a Module
```python
from src.data_validation import validate_data

is_valid, df, errors, warnings = validate_data('data/raw/car_sales.csv')

if is_valid:
    # Proceed with preprocessing
else:
    # Handle validation errors
```

### As Standalone Script
```bash
python src/data_validation/validator.py
```
Exit code: 0 (pass), 1 (fail)

## Key Design Principles

1. **Rules, Not Plots**: All validation is rule-based and executable
2. **Code, Not Notes**: Everything is implemented as working Python code
3. **Fail Fast**: Critical violations (schema, target) cause immediate failure
4. **Warnings vs Errors**: Outliers warn but don't fail validation
5. **Business Logic**: Domain-specific rules are enforced
6. **Reproducible**: All rules are version-controlled and deterministic

## Next Steps

1. ✓ Data validation module complete
2. → Move to preprocessing pipeline
3. → Integrate validation into training workflow
4. → Add to CI/CD for automated quality checks
5. → Use in retraining triggers

---

**Rule Enforced**: Raw data is protected by DVC and never modified directly ✓
