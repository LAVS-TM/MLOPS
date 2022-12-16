import numpy as np
import pandas as pd

def generate_dataframe(n_samples):
    df = pd.read_csv("../data/new_houses.csv")
    size = np.random.randint(low=df['size'].min() / 2, high=3 * df['size'].max() / 2, size=n_samples) + np.random.random()
    nb_rooms = np.random.randint(low=df['nb_rooms'].min() / 2, high=3 * df['nb_rooms'].max() / 2, size=n_samples)
    garden = np.random.randint(low=0, high=2, size=n_samples)
    orientation = np.random.randint(low=0, high=4, size=n_samples)
    price = size * 2000 + nb_rooms * 10000 + garden * 50000 * (1 + orientation) * 1000 
    data = {
        'size': size,
        'nb_rooms': nb_rooms,
        'garden': garden,
        "orientation": orientation,
        "price": price
    }
    df = pd.DataFrame(data)
    return df