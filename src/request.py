import requests

size = 30
nb_rooms = 2
garden = True
payload = {'size': size, 'nb_rooms': nb_rooms, 'garden': garden}

# Send a request
response = requests.get("http://127.0.0.1:8000/predict", params=payload)

print(response.json())