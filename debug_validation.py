import pandas as pd
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_validation import validate_data

df = pd.DataFrame([{
    "Manufacturer": "Acura",
    "Model": "Integra",
    "Vehicle_type": "Passenger",
    "Price_in_thousands": 21.5,
    "Engine_size": 1.8,
    "Horsepower": 140,
    "Wheelbase": 101.2,
    "Width": 67.3,
    "Length": 172.4,
    "Curb_weight": 2.639,
    "Fuel_capacity": 13.2,
    "Fuel_efficiency": 28.0,
    "Latest_Launch": "2/2/2012",
    "Power_perf_factor": 58.28,
    "__year_resale_value": 16.36
}])

try:
    is_valid, validated_df, errors, warnings = validate_data(df)
    print(f"Is valid: {is_valid}")
    print(f"Errors: {errors}")
except Exception as e:
    import traceback
    traceback.print_exc()
