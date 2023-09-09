import numpy as np
import numpy.linalg as la

import scipy.optimize as sopt

import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import axes3d

def f(x):
    return 3 * x[0] ** 2 + 7 * x[1] ** 2 + x[1] * x[0] - x[0]

def df(x):
    return np.array([6 * x[0] + x[1] - 1, x[0] + 14 * x[1]])

guesses = [np.array([0.5, 0.1])]

x = guesses[-1]
s = -df(x)
eps = 0.0001
def f1d(alpha):
    return f(x + alpha*s)
while True:
    alpha_opt = sopt.golden(f1d)
    next_guess = x - alpha_opt * df(x)
    if abs(f(next_guess) - f(guesses[-1])) < eps:
        break
    guesses.append(next_guess)

print(next_guess)