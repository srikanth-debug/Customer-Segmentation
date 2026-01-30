from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # only needed locally

app = FastAPI(title="MLflow Model API", version="1.0")

MODEL_PATH = "churn_model"

model = None
try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
except Exception as e:
    logging.error(f"Failed to load MLflow model at {MODEL_PATH}: {e}")

class PredictionRequest(BaseModel):
    Frequency : float
    Monetary : float
    Tenure : float
    AvgOrderValue : float
    # Add other features here

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.pyfunc
import os
import logging
from dotenv import load_dotenv

load_dotenv()  # only needed locally

app = FastAPI(title="MLflow Model API", version="1.0")



MODEL_URI = os.getenv("MODEL_URI", "models:/MyModel/Production")
model = None
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    logging.error(f"Failed to load MLflow model at {MODEL_URI}: {e}")

class PredictionRequest(BaseModel):
    Frequency : float
    Monetary : float
    Tenure : float
    AvgOrderValue : float
    # Add other features here

@app.get("/")
def root():
    return {"message" : "API running"}

@app.post("/predict")
def predict(request: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        data = [[request.Frequency, request.Monetary, request.Tenure, request.AvgOrderValue]]  # match your model input
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
