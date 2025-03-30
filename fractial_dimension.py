import numpy as np
import matplotlib.pyplot as plt


def count_with_eps(x, eps):
    n = x.shape[0]
    boxes = set()

    for i in range(n):
        cur_x = x[i] // eps
        cur_y = i // (n * eps)
        if cur_x == 1 // eps:
            cur_x -= 1
        if cur_y == n // (n * eps):
            cur_y -= 1
        boxes.add((cur_x, cur_y))

    return len(boxes)


def get_frac_dimension(x, draw=False):
    res = []
    epss = []

    for eps in range(20, 76):
        cur = count_with_eps(x, 1 / eps)
        if len(res) == 0 or np.log2(cur) != res[-1]:
            res.append(np.log2(cur))
            epss.append(np.log2(1 / (1 / eps)))

    res = np.array(res)
    epss = np.array(epss)
    a, b = np.polyfit(epss, res, deg=1)
    angle_rad = np.arctan(a)

    if draw:
        res_trend = a * epss + b
        plt.scatter(epss, res)
        plt.plot(epss, res_trend, color='red')
        plt.show()

    return angle_rad
