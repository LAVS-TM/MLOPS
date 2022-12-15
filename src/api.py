from fastapi import FastAPI
import joblib
import pandas as pd

from data_generator import generate_dataframe
from train_model import train_model

app = FastAPI()

@app.get("/predict")
async def predict(size: int, nb_rooms: int, garden: bool, orientation: int):
    model = joblib.load("model.joblib")
    X = [[size, nb_rooms, garden, orientation]]
    y_pred = model.predict(X)
    return {"y_pred": y_pred[0]}

@app.get("/retrain")
async def retrain(nb_samples: int):
    # df = generate_dataframe(nb_samples)
    df = pd.read_csv("../data/new_houses.csv").sample(nb_samples)
    df["orientation"] = df["orientation"].map({"Nord": 0, "Est": 1, "Sud": 2, "Ouest": 3})
    train_model(df)
    
    return {"message": "Model retrained"}