from fastapi import FastAPI, HTTPException
import mlflow.pyfunc
from pydantic import BaseModel
import os

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Initialize FastAPI ---
app = FastAPI(title="MLflow Model API", version="1.0")

# --- Load model once at startup ---
MODEL_URI = os.getenv("MODEL_URI", "models:/MyModel/Production")
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
except Exception as e:
    raise RuntimeError(f"Failed to load MLflow model: {e}")

# --- Request schema ---
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    # Add your features here

# --- Prediction endpoint ---
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        data = [[request.feature1, request.feature2]]  # match your model input
        prediction = model.predict(data)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Health check endpoint ---
@app.get("/health")
def health():
    return {"status": "ok"}
