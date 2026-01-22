"""
API Test Script
Starts the API and sends test requests
"""

import requests
import time
import subprocess
import os
import sys
from pathlib import Path

def test_api():
    print("="*80)
    print("STARTING API TEST")
    print("="*80)
    
    # 1. Start the API in a separate process
    # We use uvicorn to run the FastAPI app
    api_path = "api.main:app"
    print(f"Starting API server: {api_path}...")
    
    # Set PYTHONPATH to include src
    env = os.environ.copy()
    project_root = str(Path(__file__).parent)
    env["PYTHONPATH"] = os.path.join(project_root, "src") + os.pathsep + env.get("PYTHONPATH", "")
    
    process = subprocess.Popen(
        ["uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server to start with retries
        print("Waiting for server to initialize...")
        max_retries = 30
        for i in range(max_retries):
            try:
                response = requests.get("http://127.0.0.1:8000/", timeout=1)
                if response.status_code == 200:
                    print(f"Server started after {i+1} seconds.")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
                if i == max_retries - 1:
                    print("[ERROR] Server failed to start in time.")
                    process.terminate()
                    stdout, stderr = process.communicate()
                    print("\n--- SERVER STDOUT ---")
                    print(stdout)
                    print("\n--- SERVER STDERR ---")
                    print(stderr)
                    sys.exit(1)
        
        # 2. Test Root Endpoint
        print("\n[1] Testing Root Endpoint...")
        response = requests.get("http://127.0.0.1:8000/")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # 3. Test Health Endpoint
        print("\n[2] Testing Health Endpoint...")
        response = requests.get("http://127.0.0.1:8000/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # 4. Test Valid Prediction
        print("\n[3] Testing Valid Prediction...")
        valid_data = {
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
        
        response = requests.post("http://127.0.0.1:8000/predict", json=valid_data)
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
            print(f"[OK] Prediction: {response.json()['prediction']}")
        else:
            print(f"Error: {response.text}")
            
        # 5. Test Invalid Prediction (Data Validation Check)
        print("\n[4] Testing Invalid Prediction (Negative Price)...")
        invalid_data = valid_data.copy()
        invalid_data["Price_in_thousands"] = -10.0 # Should fail validation
        
        response = requests.post("http://127.0.0.1:8000/predict", json=invalid_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        if response.status_code == 400:
            print("[OK] Validation correctly caught the error")
            
    finally:
        # Terminate the server process
        print("\nShutting down API server...")
        process.terminate()
        stdout, stderr = process.communicate()
        # print(stdout)
        
    print("\n" + "="*80)
    print("API TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    test_api()
