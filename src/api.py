from fastapi import FastAPI
import joblib
import pandas as pd
import numpy as np
import os

from data_generator import generate_dataframe
from train_model import train_model

app = FastAPI()


@app.get("/predict")
async def predict(size: int, nb_rooms: int, garden: bool, orientation: int):
    if not os.path.exists("model.joblib"):
        df = pd.read_csv("../data/houses.csv")
        df["orientation"] = df["orientation"].map(
            {"Nord": 0, "Est": 1, "Sud": 2, "Ouest": 3})
        train_model(df)
    model = joblib.load("model.joblib")
    X = [[size, nb_rooms, garden, orientation]]
    y_pred = model.predict(X)
    return {"y_pred": y_pred[0]}


@app.get("/retrain")
async def retrain(nb_samples: int):
    if not os.path.exists("model.joblib"):
        df = pd.read_csv("../data/houses.csv")
        df["orientation"] = df["orientation"].map(
            {"Nord": 0, "Est": 1, "Sud": 2, "Ouest": 3})
        train_model(df)
    """
    while (nb_samples > 0):
        df = pd.read_csv("../data/new_houses.csv")
        if len(df) < nb_samples:
            sample_size = len(df) % nb_samples
        else:
            sample_size = nb_samples
        df = df.sample(sample_size)
        # Add bias to the new data
        df["price"] = df["price"] + np.random.randint(low=
                                                      -df["price"].min() / 2, high=df["price"].min() / 2, size=sample_size)
        train_model(df)
        nb_samples -= sample_size
    """
    df = generate_dataframe(nb_samples)
    train_model(df)
    return {"message": "Model retrained"}
