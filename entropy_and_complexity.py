import numpy as np
from lorenz_time_series import build_lorenz_time_series
from henon_map import build_henon_map
from sin_time_series import build_sin_series
from gauss_time_series import build_gauss_series
from tent_map import build_tent_map
from logistic_map import build_logistic_map
from itertools import permutations, product


def get_entropy_and_complexity_sorting(x):
    m = 4
    n = x.shape[0]
    k = n - m + 1
    cnt = list(permutations(np.arange(m)))
    r = len(cnt)
    p = np.zeros(r)

    for i in range(k):
        z_i = x[i:i + m]
        sorted_indices = np.argsort(z_i)
        p[np.all(cnt == sorted_indices, axis=1)] += 1

    probabilities_p = p / k
    probabilities_pe = np.full(r, 1 / r)
    probabilities_p_pe = (probabilities_p + probabilities_pe) / 2

    S_p = -np.sum(np.log(probabilities_p + 10 ** (-13)) * probabilities_p)
    S_max = -np.sum(np.log(probabilities_pe + 10 ** (-13)) * probabilities_pe)
    S_p_pe = -np.sum(np.log(probabilities_p_pe + 10 ** (-13)) * probabilities_p_pe)

    Q_0 = -2 * ((r + 1) * (np.log(r + 1)) / r - 2 * np.log(2 * r) + np.log(r)) ** (-1)
    Q_p_pe = (S_p_pe - S_p / 2 - S_max / 2) * Q_0

    H_p = S_p / S_max
    C_p = H_p * Q_p_pe

    return H_p, C_p


def get_entropy_and_complexity_slicing(x):
    return 0
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    m = 4
    n = x.shape[0]
    k = n - m + 1
    eps = 0.05
    combs = product(range(21), repeat=m)
    cnt = [np.array(i) for i in combs]
    r = len(cnt)
    print(r)
    p = np.zeros(r)

    for i in range(k):
        z_i = x[i:i + m]
        sorted_indices = z_i // eps
        p[np.all(cnt == sorted_indices, axis=1)] += 1

    probabilities_p = p / k
    probabilities_pe = np.full(r, 1 / r)
    probabilities_p_pe = (probabilities_p + probabilities_pe) / 2

    S_p = -np.sum(np.log(probabilities_p + 10 ** (-13)) * probabilities_p)
    S_max = -np.sum(np.log(probabilities_pe + 10 ** (-13)) * probabilities_pe)
    S_p_pe = -np.sum(np.log(probabilities_p_pe + 10 ** (-13)) * probabilities_p_pe)

    Q_0 = -2 * ((r + 1) * (np.log(r + 1)) / r - 2 * np.log(2 * r) + np.log(r)) ** (-1)
    Q_p_pe = (S_p_pe - S_p / 2 - S_max / 2) * Q_0

    H_p = S_p / S_max
    C_p = H_p * Q_p_pe

    return H_p, C_p


def count_for_lorenz():
    return get_entropy_and_complexity_sorting(build_lorenz_time_series()), get_entropy_and_complexity_slicing(
        build_lorenz_time_series())


def count_for_henon():
    return get_entropy_and_complexity_sorting(build_henon_map(0.5, 1)), get_entropy_and_complexity_slicing(
        build_henon_map(0.5, 1))


def count_for_sin():
    return get_entropy_and_complexity_sorting(build_sin_series()), get_entropy_and_complexity_slicing(
        build_sin_series())


def count_for_gauss():
    return get_entropy_and_complexity_sorting(build_gauss_series()), get_entropy_and_complexity_slicing(
        build_gauss_series())


def count_for_tent():
    return get_entropy_and_complexity_sorting(build_tent_map(1.9, 0.4)), get_entropy_and_complexity_slicing(
        build_tent_map(1.9, 0.4))


def count_for_logistic():
    return get_entropy_and_complexity_sorting(build_logistic_map(3.7, 0.3)), get_entropy_and_complexity_slicing(
        build_logistic_map(3.7, 0.3))


def count_for_all():
    print(count_for_lorenz())
    print(count_for_henon())
    print(count_for_sin())
    print(count_for_gauss())
    print(count_for_tent())
    print(count_for_logistic())
