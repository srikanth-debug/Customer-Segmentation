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

@app.get("/")
def root():
    return {"message" : "API running"}

@app.post("/predict")
def predict(request: PredictionRequest):

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request → dataframe (pyfunc prefers dataframe)
        data = {
            "Frequency": [request.Frequency],
            "Monetary": [request.Monetary],
            "Tenure": [request.Tenure],
            "AvgOrderValue": [request.AvgOrderValue]
        }

        import pandas as pd
        df = pd.DataFrame(data)

        pred = model.predict(df)[0]

        #***---Buisness-layer logic***
        customer_value = request.Monetary
        freq = request.Frequency

        if customer_value > 500:
            value_tier = "HIGH_VALUE"
        elif customer_value > 200:
            value_tier = "MID_VALUE"
        else:
            value_tier = "LOW_VALUE"

        if pred == 1 and value_tier == "HIGH_VALUE":
            action = "Immediate retention campaign"
            discount = "20%"
        elif pred == 1:
            action = "Send retention offer"
            discount = "10%"
        else:
            action = "No action required"
            discount = "0%"

        explanation = (
            f"Customer shows churn risk based on purchase freequency"
            f"({freq}) and total spend ({customer_value})"
        )                     

        return {
            "churn_prediction": int(pred),
            "risk_level": "HIGH" if pred == 1 else "LOW",
            "customer_value_tier" : value_tier,
            "recommonded_action": action,
            "suggested_discount" : discount,
            "explanation" : explanation
    
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    

@app.get("/health")
def health():
    return {"status": "ok"}
