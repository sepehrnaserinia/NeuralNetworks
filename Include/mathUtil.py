from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def rand(mu, sigma, len):
    return mu + sigma * np.random.randn(len)

def shuffle_in_unison(array_1, array_2, array3=None):
    rng_state = np.random.get_state()
    np.random.shuffle(array_1)
    np.random.set_state(rng_state)
    np.random.shuffle(array_2)
    if array3 is not None:
        np.random.set_state(rng_state)
        np.random.shuffle(array3)

def distance(point):
    return sqrt((point[0][0] - point[1][0])**2 + (point[0][1] - point[1][1])**2)

if __name__ == "__main__":
    print("Header File...")
