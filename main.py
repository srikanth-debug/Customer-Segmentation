from fastapi import FastAPI
from pydantic import BaseModel
import mlflow.sklearn
import pandas as pd
import mlflow.pyfunc
mlflow.set_tracking_uri("sqlite:///mlflow.db")


app = FastAPI(Title="Churn Prediction API" )

#Load the model
model = mlflow.sklearn.load_model("churn_model")


class CustomerFeatures(BaseModel):
    Frequency : float
    Monetary : float
    Tenure : float
    AvgOrderValue : float

@app.get("/")
def home():
      return {"status : API running"}


@app.post("/predict")
def predict_churn(data: CustomerFeatures):
        df = pd.DataFrame([data.dict()])
        churn_prob = model.predict_proba(df)[0][1]
        pred = model.predict(df)[0]

        return {
              "churn_probability" : round(float(churn_prob), 3),
              "prediction" : int(pred),
              "risk_level": "HIGH" if churn_prob > 0.7 else "Low"
        }