import numpy as np
from datetime import datetime
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
table = []
for i in n:
    initial = np.random.uniform(-1, 1, i)
    result = []
    for j in k:
        avg_res = 0
        for step in range(5):
            vec = gen_vector(i, j)
            avg_res += steepest_descent_with_dichotomy(i, random_fun, grad_random_fun, initial, 0.0001,
                                                       5000, [0, 0.4], vec)
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
