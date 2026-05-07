from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    return {
        "prediction": prediction[0]
    }