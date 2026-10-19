"""Serve the trained model as an API.  Run:  uvicorn api:app --reload   then open http://127.0.0.1:8000/docs"""
import joblib
import pandas as pd
from fastapi import FastAPI

from features import add_features
from logger import log_prediction

app = FastAPI(title="Credit Default Risk API")
model = joblib.load("models/model.joblib")     # created by train.py


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(applicant: dict):
    """Send the 23 raw columns as JSON (LIMIT_BAL, SEX, EDUCATION, ... PAY_AMT6)."""
    row = add_features(pd.DataFrame([applicant]))
    probability = float(model.predict_proba(row)[0, 1])
    log_prediction(row, probability)
    return {"default_probability": round(probability, 4), "prediction": int(probability >= 0.5)}
