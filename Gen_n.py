import numpy as np
from datetime import datetime

import numpy.array_api

start_time = datetime.now()


# vec - array of coefficients
def gen_vector(n, k):
    vec = np.random.uniform(1, k, n)
    vec[0] = 1
    vec[n - 1] = k
    return vec


def random_fun(n, x, vec):
    sum = 0
    for i in range(n):
        sum += x[i] ** 2 * vec[i]
    return sum


def grad_random_fun(n, x, vector):
    for i in range(n):
        vector[i] = 2 * x[i] * vector[i]
    return vector


st = [-20, 10]


def steepest_descent_with_dichotomy(n, f, grad_f, initial, epsilon, max_iterations, alpha_bounds, vec):
    p = initial
    iteration = 1
    while iteration < max_iterations:
        gradient = grad_f(n, p, vec.copy())

        # Метод дихотомии для определения оптимального шага alpha
        a = alpha_bounds[0]
        b = alpha_bounds[1]
        while abs(b - a) > epsilon:
            alpha1 = (2 * a + b) / 3
            alpha2 = (a + 2 * b) / 3
            if f(n, p - alpha1 * gradient, vec) < f(n, p - alpha2 * gradient, vec):
                b = alpha2
            else:
                a = alpha1
            alpha_star = (a + b) / 2
        p_new = p - alpha_star * gradient
        if abs(f(n, p_new, vec) - f(n, p, vec)) < epsilon:
            break
        p = p_new
        iteration += 1

    return iteration


# print(steepest_descent_with_dichotomy(random_fun, grad_random_fun, initial, 0.00001, 50000, [0.001, 0.4]))

# n = [2, 5, 10, 25, 50, 100, 200, 350, 500, 750, 1000]
# k = [1, 5, 10, 50, 100, 250, 375, 500, 750, 1000]
n = [2, 5, 10, 25, 50, 100, 250, 500, 1000]
k = [1, 5, 10, 50, 100, 250, 500, 900]
#n = [2, 1]
#k = [1, 2]
maxItr = 100
bounds = [0, 0.4]
epsilon = 0.01
table = []
iteration = 0
def Gr_m(initial, F, grad_f, vec, n, eps, alpha, max_iteration):
    i = 0
    iteration = 1
    X_prev = initial
    X = X_prev - alpha * grad_f(n, X_prev, vec.copy())
    while np.linalg.norm(X - X_prev) > eps:
        X_prev = X.copy()
        i = i + 1
        #print(i, ":", X)
        X = X_prev - alpha * grad_f(n, X_prev, vec.copy())  # Формула
        iteration += 1
        if iteration > max_iteration:
            return iteration
    print("Iteration:", iteration)
    return iteration
def descentWithWolfe(initial, F, grad_f, vec, n, a1 = 1e-4, a2 = 0.9, eps = 1e-2, maxIter = 1000,):
    iter = 0
    x = initial
    while iter <= maxIter:
        gradient = grad_f(n, x, vec.copy())
        alpha = 1

        while True:
            xk = x - alpha * gradient
            if F(n, xk, vec.copy()) <= F(n, x, vec.copy()) + a1 * alpha * gradient.T @ (-gradient):
                if (grad_f(n, xk, vec.copy()) @ (-gradient)) >= a2 * gradient.T @ (-gradient):
                    break
            alpha /= 2
        if np.linalg.norm(gradient) < eps:
            break
        iter += 1
        x = xk
    print(iter)
    return iter

for i in n:
    initial = np.random.uniform(-1, 1, i)
    result = []
    for j in k:
        avg_res = 0
        for step in range(5):
            vec = gen_vector(i, j)
            # avg_res += steepest_descent_with_dichotomy(i, random_fun, grad_random_fun, initial, epsilon,
            #                                            maxItr, bounds, vec)
            #avg_res += Gr_m(initial, random_fun, grad_random_fun, vec, i, epsilon, 0.001, maxItr)
            avg_res += descentWithWolfe(initial, random_fun, grad_random_fun, vec, i)
        result.append(np.hstack((i, j, avg_res / 5)))
    table.append(np.array(result))
print(table)

colors = ['blue', 'red', 'yellow', 'brown', 'grey', 'green', 'purple', 'orange',
           'black', 'pink']
import matplotlib.pyplot as plt

for i in range(len(n)):
    plt.plot(table[i].T[2], table[i].T[1], linestyle='-', marker='o', color=colors[i], label='n=' + str(table[i][0][0]))
plt.grid()
plt.legend()
plt.show()
end = datetime.now() - start_time
print(end)
