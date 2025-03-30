import numpy as np
import matplotlib.pyplot as plt


def build_gauss_series(loc=0.5, scale=0.2):
    y_gauss = np.random.normal(loc=loc, scale=scale, size=13000)
    y_gauss_norm = (y_gauss - np.min(y_gauss)) / (np.max(y_gauss) - np.min(y_gauss))
    return y_gauss_norm


def build_cumsum_gauss_series(loc=0.5, scale=0.2):
    y_cumsum = build_gauss_series(loc=loc, scale=scale)
    y_cumsum_norm = (y_cumsum - np.min(y_cumsum)) / (np.max(y_cumsum) - np.min(y_cumsum))
    return y_cumsum_norm


def draw():
    x = build_cumsum_gauss_series()
    plt.plot(x)
    plt.show()
