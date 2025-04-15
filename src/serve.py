import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc

# Fix Windows path issue for local MLflow tracking URI
mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}")

# Model registry name & stage
MODEL_NAME = "iris_rf_model"
MODEL_STAGE = "Production"

# Try to load model
try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    print(f"(!) Failed to load model from registry: {e}")
    model = None

# FastAPI app
app = FastAPI(
    title="MLflow Iris Classifier API",
    description="Serve a RandomForest model from MLflow Registry",
    version="1.0"
)

# Input schema
class InputData(BaseModel):
    features: list[float]  # Expected length = 4

@app.get("/")
def root():
    return {"message": "MLflow Iris Model API is running."}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    if len(data.features) != 4:
        raise HTTPException(status_code=400, detail="Input feature length must be 4.")

    pred = model.predict([data.features])
    return {"prediction": int(pred[0])}
