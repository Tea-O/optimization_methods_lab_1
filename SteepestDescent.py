import time
from datetime import datetime

import numpy as np

start_time = datetime.now()
st = [-20, 10]
eposh = 41
points= np.zeros((eposh, 2))
points[0] = st


def steepest_descent_with_dichotomy(f, grad_f, initial_x, initial_y, epsilon, max_iterations, alpha_bounds):
    p = [initial_x, initial_y]
    iteration = 1

    while iteration < max_iterations:
        gradient = grad_f(p)

        # Метод дихотомии для определения оптимального шага alpha
        a = alpha_bounds[0]
        b = alpha_bounds[1]
        while abs(b - a) > epsilon:
            alpha1 = (2 * a + b) / 3
            alpha2 = (a + 2 * b) / 3
            if f(p - alpha1 * gradient) < f(p - alpha2 * gradient):
                b = alpha2
            else:
                a = alpha1
            alpha_star = (a + b) / 2

        x_new = p[0] - alpha_star * gradient[0]
        y_new = p[1] - alpha_star * gradient[1]
        if abs(f([x_new, y_new]) - f([p[0], p[1]])) < epsilon:
            break

        p = x_new, y_new
        points[iteration] = p
        iteration += 1

    return p[0], p[1], iteration


def F(x):
    return 7 * (x[0] + 3) ** 2 + 0.5 * (x[1] + 7) ** 2



def grad(x):
    return np.array([14 * x[0] + 42, x[1] + 7])


result = steepest_descent_with_dichotomy(F, grad, -20, 10, 0.00001, 50, np.array([0.01, 0.4]))
print(result)
print(points)
end = datetime.now() - start_time
print(end)
import matplotlib.pyplot as plt

X_estimate, Y_estimate = points[:, 0], points[:, 1]
# print(X_estimate)
# print(Y_estimate)
Z_estimate = F(np.array([X_estimate, Y_estimate]))

min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
t = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(t, t)
plt.contour(X, Y, F([X, Y]), levels=sorted([F(p) for p in points]))
plt.plot(X_estimate, Y_estimate, '.-', color="r")

# fig = plt.figure(figsize=(35, 35))
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# t = np.linspace(-50, 50, 100)
# X, Y = np.meshgrid(t, t)
# ax.contour3D(X, Y, F(np.array([X, Y])), 60, cmap='viridis')
# ax.plot(X_estimate, Y_estimate, '.-', color='red')
# #ax.scatter(X_estimate, Y_estimate, F(np.array([X_estimate, Y_estimate])), marker='o', color='blue', linewidth=3)
# ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
# ax.view_init(60, 180)
plt.show()

# t = np.linspace(-10, 10, 20)
# X, Y = np.meshgrid(t, t)
# ax = plt.figure().add_subplot(projection='3d')
# ax.plot_surface(X, Y, F(np.array([X, Y])))
# # plt.plot(ans[:, 0], ans[:, 1], 'o-')
# # plt.contour(X, Y, F(X, Y), levels=sorted(F(p[0], p[1]) for p in ans))
#
# plt.show()

