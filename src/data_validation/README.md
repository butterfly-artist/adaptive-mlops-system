# Data Validation Module

## Overview
This module implements comprehensive data validation for car sales data, ensuring data quality before training ML models.

## Components

### 1. `config.py` - Validation Configuration
Defines all validation rules and constraints:

- **Schema Definition**
  - 16 required columns
  - Target column: `Sales_in_thousands`
  
- **Data Types**
  - 4 object (string) columns: Manufacturer, Model, Vehicle_type, Latest_Launch
  - 12 float64 columns: Sales, resale value, price, engine specs, dimensions, efficiency
  
- **Missing Value Strategy**
  - **Reject**: Critical columns (Manufacturer, Model, Sales_in_thousands, Vehicle_type, Latest_Launch)
  - **Median imputation**: Numerical features (Price, Engine_size, Horsepower, etc.)
  - **Maximum thresholds**: Resale value can have up to 30% missing, others 5%

- **Range Constraints**
  ```
  Sales_in_thousands: [0, 1000]
  Price_in_thousands: [0, 200]
  Engine_size: [0.5, 10.0]
  Horsepower: [50, 600]
  Fuel_efficiency: [10, 60]
  ... and more
  ```

- **Categorical Constraints**
  - Vehicle_type must be 'Passenger' or 'Car'

- **Date Validation**
  - Format: MM/DD/YYYY
  - Range: 2000-01-01 to present
  - **No future dates allowed**

- **Outlier Detection**
  - IQR method with multiplier 3.0
  - Applied to Sales, Price, Horsepower, Power_perf_factor

- **Business Logic Rules**
  - Resale value < Original price
  - Power performance factor > 0

### 2. `validator.py` - Validation Engine

#### Main Class: `DataValidator`

**Validation Checks:**
1. ✓ Schema validation (all required columns present)
2. ✓ Data type validation
3. ✓ Missing value validation
4. ✓ Target column integrity (non-negative, non-null)
5. ✓ Range validation for all numeric columns
6. ✓ Categorical value validation
7. ✓ Date format and range validation
8. ✓ Minimum dataset size (50 rows)
9. ✓ Outlier detection (warnings)
10. ✓ Business logic relationships

#### Usage

```python
from data_validation import validate_data

# Validate a CSV file
is_valid, df, errors, warnings = validate_data('path/to/data.csv')

if is_valid:
    print("Data is valid!")
else:
    for error in errors:
        print(f"Error: {error}")
```

#### Command Line

```bash
python src/data_validation/validator.py
```

Returns exit code 0 if validation passes, 1 if fails.

## Validation Results (Current Dataset)

**Dataset**: 157 rows × 16 columns

**Status**: ❌ FAILED

**Errors** (1):
- Column 'Vehicle_type' has invalid values: {'Car'}. Allowed values: ['Passenger', 'Car']

**Warnings** (3):
1. Column 'Sales_in_thousands' has 1 (0.65%) potential outliers
2. Column 'Price_in_thousands' has 6 (3.90%) potential outliers
3. Column 'Horsepower' has 1 (0.65%) potential outliers

## Key Design Decisions

1. **No Fancy Plots**: Validation is rule-based, not exploratory
2. **Executable Code**: All rules are code, not documentation
3. **Fail Fast**: Critical violations cause immediate failure
4. **Warnings vs Errors**: Outliers warn but don't fail; schema violations fail
5. **Business Logic**: Validates domain-specific relationships
6. **Reproducibility**: All rules are version-controlled and deterministic

## Next Steps

1. Fix validation errors in data if needed
2. Use this validator in preprocessing pipeline
3. Integrate into CI/CD for automated data quality checks
4. Add to retraining triggers (if validation fails, alert don't train)
