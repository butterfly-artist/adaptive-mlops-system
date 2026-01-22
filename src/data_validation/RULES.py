"""
Quick Reference: Data Validation Rules
Execute: python -m src.data_validation.validator
"""

# TARGET
TARGET = 'Sales_in_thousands'

# REQUIRED COLUMNS (16 total)
COLUMNS = [
    'Manufacturer', 'Model', 'Sales_in_thousands', '__year_resale_value',
    'Vehicle_type', 'Price_in_thousands', 'Engine_size', 'Horsepower',
    'Wheelbase', 'Width', 'Length', 'Curb_weight', 'Fuel_capacity',
    'Fuel_efficiency', 'Latest_Launch', 'Power_perf_factor'
]

# RULES

## 1. NON-NEGATIVE VALUES
# - Sales_in_thousands >= 0 (TARGET - CRITICAL)
# - Price_in_thousands > 0
# - All physical measurements > 0

## 2. REALISTIC RANGES
# Sales: 0-1000 thousand units
# Price: 0-200 thousand dollars
# Engine: 0.5-10.0 liters
# Horsepower: 50-600 HP
# Fuel efficiency: 10-60 MPG

## 3. NO MISSING VALUES IN
# - Manufacturer
# - Model
# - Sales_in_thousands (TARGET)
# - Vehicle_type
# - Latest_Launch

## 4. DATES
# - Format: MM/DD/YYYY
# - No future dates
# - Range: 2000 onwards

## 5. CATEGORICAL
# - Vehicle_type in ['Passenger', 'Car']

## 6. BUSINESS LOGIC
# - Resale value < Original price
# - Power_perf_factor > 0 when present

## 7. DATA QUALITY
# - Minimum 50 rows
# - Maximum 5% missing for most numerical features
# - Maximum 30% missing for resale_value
