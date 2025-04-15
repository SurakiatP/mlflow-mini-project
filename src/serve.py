import os
from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.pyfunc

import os
import mlflow

# Fix Windows path issue for tracking_uri
mlflow.set_tracking_uri(f"file:///{os.path.abspath('mlruns').replace(os.sep, '/')}")

# Load model from registry
MODEL_NAME = "iris_rf_model"
MODEL_STAGE = "Production"

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_STAGE}")
except Exception as e:
    print(f" Failed to load model from registry: {e}")
    model = None

# FastAPI App
app = FastAPI(
    title="MLflow Iris Classifier API",
    description="Serve a RandomForest model from MLflow Registry",
    version="1.0"
)

# Input data structure
class InputData(BaseModel):
    features: list  # Expected: list of 4 float values (iris features)

@app.get("/")
def root():
    return {"message": "MLflow Iris Model API is running."}

@app.post("/predict")
def predict(data: InputData):
    if model is None:
        return {"error": "Model not loaded."}
    pred = model.predict([data.features])
    return {"prediction": int(pred[0])}

