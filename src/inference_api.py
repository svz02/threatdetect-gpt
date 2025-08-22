# deployment
from fastapi import FastAPI, Request
import joblib
import pandas as pd

app = FastAPI()
pipeline = joblib.load("models/rf_pipeline.joblib")

@app.post("/predict")
async def predict(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    pred = pipeline.predict(df)[0]
    return {"threat": bool(pred)}