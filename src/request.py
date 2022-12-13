import requests

def predict_request(size, nb_rooms, garden):
    payload = {'size': size, 'nb_rooms': nb_rooms, 'garden': garden}
    response = requests.get("http://127.0.0.1:8000/predict", params=payload)
    return response.json()