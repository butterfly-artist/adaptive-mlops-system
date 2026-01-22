"""
Manual Prediction Logging Test
"""

import requests
import time
import subprocess
import os
from pathlib import Path

def test_logging():
    print("="*80)
    print("TESTING PREDICTION LOGGING")
    print("="*80)
    
    # Start API
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parent / "src")
    
    process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        time.sleep(5) # Wait for startup
        
        payload = {
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
        }
        
        print("Sending prediction request...")
        response = requests.post("http://127.0.0.1:8000/predict", json=payload)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        print("Waiting for background logging task...")
        time.sleep(2)
        
        log_file = Path("logs/predictions.csv")
        if log_file.exists():
            print(f"✓ Success! Log file created at {log_file}")
            print("\nLog content sample:")
            with open(log_file, 'r') as f:
                print(f.read())
        else:
            print("✗ Failure: Log file not found.")
            
    finally:
        process.terminate()
        stdout, stderr = process.communicate()
        if stderr:
            print("\nServer Stderr:")
            print(stderr)

if __name__ == "__main__":
    test_logging()
