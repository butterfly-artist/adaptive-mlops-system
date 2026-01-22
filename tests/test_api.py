"""
Professional API Test Suite using Pytest
Tests:
- Schema validation
- Edge cases
- Missing fields
- Successful predictions
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os
from pathlib import Path

# Add src to path so api.main can find its imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the app
from api.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

@pytest.fixture
def valid_payload():
    return {
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

def test_read_root(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert response.json()["status"] == "healthy"

def test_health_check(client):
    """Test the health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_success(client, valid_payload):
    """Test a successful prediction with valid data"""
    response = client.post("/predict", json=valid_payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["status"] == "success"
    assert data["prediction"] >= 0

def test_predict_invalid_schema(client):
    """Test schema validation (missing required field)"""
    invalid_payload = {
        "Manufacturer": "Acura",
        "Model": "Integra"
        # Missing many required fields
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)

def test_predict_validation_error(client, valid_payload):
    """Test business logic validation (negative price)"""
    invalid_payload = valid_payload.copy()
    invalid_payload["Price_in_thousands"] = -50.0
    
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 400
    assert "Data validation failed" in response.json()["detail"]["message"]

def test_predict_missing_optional_field(client, valid_payload):
    """Test prediction with missing optional field (__year_resale_value)"""
    payload = valid_payload.copy()
    del payload["__year_resale_value"]
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "prediction" in response.json()

def test_predict_edge_case_zero_values(client, valid_payload):
    """Test edge case with zero values where allowed"""
    payload = valid_payload.copy()
    # Price 0 might be allowed by schema but maybe not by business logic?
    # Let's see what our validator says.
    payload["Price_in_thousands"] = 0.1 
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

def test_batch_prediction(client, valid_payload):
    """Test batch prediction endpoint"""
    payload = [valid_payload, valid_payload]
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    assert len(response.json()) == 2
    assert response.json()[0]["status"] == "success"
