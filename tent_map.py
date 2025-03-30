import numpy as np
import matplotlib.pyplot as plt


def build_tent_map(mu, x):
    tent_map = [x]
    for i in range(1, 13000):
        x = mu * min(x, 1 - x)
        tent_map.append(x)
    return np.array(tent_map)


def draw():
    x = build_tent_map(1.9, 0.4)
    plt.plot(x[:300])
    plt.show()
