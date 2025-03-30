import numpy as np
import matplotlib.pyplot as plt


def build_logistic_map(r, x):
    logistic_map = [x]
    for i in range(1, 13000):
        x = r * x * (1 - x)
        logistic_map.append(x)
    return np.array(logistic_map)


def draw():
    x = build_logistic_map(3.7, 0.3)
    plt.plot(x[:300])
    plt.show()

