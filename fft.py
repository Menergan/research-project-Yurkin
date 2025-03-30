import numpy as np


def fft(x, low_rate, high_rate, x_rate):
    high_rate = min(high_rate, x_rate)

    f = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), d=1 / x_rate)

    fft_filtered = f * np.logical_and(np.abs(freq) >= low_rate, np.abs(freq) <= high_rate)
    x_filtered = np.fft.ifft(fft_filtered).real

    return np.array(x_filtered)


def fft_2(x, low_rate1, high_rate1, low_rate2, high_rate2, x_rate):
    return fft(x, low_rate1, high_rate1, x_rate) + fft(x, low_rate2, high_rate2, x_rate)


def max_amplitude(x, x_rate):
    f = np.fft.fft(x)
    ampls = (np.abs(f) / len(x))[:len(x) // 2]
    freqs = np.fft.fftfreq(len(x), 1 / x_rate)[:len(x) // 2]
    return freqs[np.argmax(ampls)]
