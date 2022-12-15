import numpy as np
import pandas as pd

def generate_dataframe(n_samples):
    size = np.random.randint(low=15, high=1000, size=n_samples) + np.random.random()
    nb_rooms = np.random.randint(low=1, high=10, size=n_samples)
    garden = np.random.randint(low=0, high=2, size=n_samples)
    orientation = np.random.randint(low=0, high=4, size=n_samples)
    price = size * 1000 + nb_rooms * 10000 + garden * 50000 * orientation * 1000 + np.random.randint(low=0, high=100000, size=n_samples)
    data = {
        'size': size,
        'nb_rooms': nb_rooms,
        'garden': garden,
        "orientation": orientation,
        "price": price
    }
    df = pd.DataFrame(data)
    return df