from fastapi.testclient import TestClient
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.serve import app

client = TestClient(app)

def test_root():
    res = client.get("/")
    assert res.status_code == 200
    assert res.json() == {"message": "MLflow Iris Model API is running."}

def test_valid_prediction():
    res = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
    assert res.status_code == 200
    assert "prediction" in res.json()

def test_invalid_input_type():
    res = client.post("/predict", json={"features": "invalid"})
    assert res.status_code == 422  # Pydantic validation

def test_wrong_length():
    res = client.post("/predict", json={"features": [1.0, 2.0]})
    assert res.status_code == 400
    assert "detail" in res.json()
    assert res.json()["detail"] == "Input feature length must be 4."

