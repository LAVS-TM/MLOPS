from fastapi import FastAPI
import joblib

model = joblib.load("model.joblib")

app = FastAPI()

@app.get("/predict")
async def predict(size: int, nb_rooms: int, garden: bool):
    X = [[size, nb_rooms, garden]]
    y_pred = model.predict(X)
    return {"y_pred": y_pred[0]}