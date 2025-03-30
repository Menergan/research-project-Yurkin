import numpy as np
import matplotlib.pyplot as plt


def build_henon_map(x1, x2):
    a = 1.4
    b = 0.3
    henon_map = [x1, x2]
    for i in range(2, 13000):
        x = 1 - a * henon_map[i - 1] ** 2 + b * henon_map[i - 2]
        henon_map.append(x)
    return np.array(henon_map)


def draw():
    x = build_henon_map(1, 0.5)
    plt.plot(x[:300])
    plt.show()
