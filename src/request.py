import requests

def predict_request(size, nb_rooms, garden, orientation):
    orientation = 0 if orientation == "North" else 1 if orientation == "East" else 2 if orientation == "South" else 3
    payload = {'size': size, 'nb_rooms': nb_rooms, 'garden': garden, 'orientation': orientation}
    response = requests.get("http://127.0.0.1:8000/predict", params=payload)
    return response.json()

def retrain_request(nb_samples):
    payload = {'nb_samples': nb_samples}
    response = requests.get("http://127.0.01:8000/retrain", params=payload)
    return response.json()