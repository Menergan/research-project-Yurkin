import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def build_sin_series():
    y_sin = np.sin(np.arange(0, 1000, .01))
    y_norm_sin = (y_sin - np.min(y_sin)) / (np.max(y_sin) - np.min(y_sin))
    return y_norm_sin[0:13000]


def build_sin_waves(f=(2 * np.pi), ampl=1):
    y_sin_wave = ampl * np.sin(f * np.linspace(0, 10, 13000))
    y_norm_sin_wave = (y_sin_wave - np.min(y_sin_wave)) / (np.max(y_sin_wave) - np.min(y_sin_wave))
    return y_norm_sin_wave


def build_saw_waves(f=(2 * np.pi), ampl=1):
    y_saw_wave = np.array(ampl * signal.sawtooth(f * np.linspace(0, 10, 13000)))
    y_norm_saw_wave = (y_saw_wave - np.min(y_saw_wave)) / (np.max(y_saw_wave) - np.min(y_saw_wave))
    return y_norm_saw_wave


def build_composite_sin(n):
    x = np.linspace(0, 10, 13000)
    y = np.zeros_like(x)

    for i in range(n):
        ampl = np.random.uniform(0.5, 3.0)
        f = np.random.uniform(1.0, 5.0)
        phase = np.random.uniform(0, 2 * np.pi)

        y = ampl * np.sin(f * x + phase)

    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))

    return y_norm


def draw():
    x = build_sin_series()
    plt.plot(x[:5000])
    plt.show()
