import numpy as np


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
n = [2, 5, 10]
k = [1, 5, 10]
table = []
for i in n:
    initial = np.random.uniform(0, 10, i)
    avg_result = []
    for j in k:
        vec = gen_vector(i, j)
        avg_result.append(np.hstack((i, j,
                                     steepest_descent_with_dichotomy(i, random_fun, grad_random_fun, initial, 0.00001,
                                                                     50000, [0, 0.4], vec))))
    table.append(np.array(avg_result))

print(table)

print(table[0].T[2])
colors = ['blue', 'red', 'yellow']
import matplotlib.pyplot as plt

for i in range(len(n)):
    plt.plot(table[i].T[2], table[i].T[1], linestyle='-', marker='o', color=colors[i], label=str(table[i][0][0]))
plt.grid()
plt.legend()
plt.show()

#
# x = np.arange(0, 1000)
# plt.figure(figsize=(10, 5))
# plt.xlabel(r'$K$', fontsize=14)
# plt.ylabel(r'$N$', fontsize=14)
# plt.grid(True)
# plt.legend(loc='best', fontsize=12)
# plt.show()
