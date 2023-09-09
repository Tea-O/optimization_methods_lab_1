import numpy as np


def Dichotomy(l, r, eps, func):
    while(r - l > eps):
        c = (l + r) / 2
        if func(r) * func(c) < 0:
            l = c
        else:
            r = c
    return (r + l) / 2

def F(x):
    return 4 - np.exp(x) - 2 * x ** 2

print(Dichotomy(0, 2, 0.001, F))