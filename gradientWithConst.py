import numdifftools as nd
import numpy as np

eposh = 200
st = np.array([-5, -20])
points = np.zeros((eposh, 2))
points[0] = st
xscale = 1.5
yscale = 0.4
# def F(x):
#     #return 2 * (x[0] - 1) ** 2 + 10 * (x[1] + 10) ** 2
#     return 2 * (x[0] - 1) ** 2 + 30 * (x[1] + 10) ** 2
def F(x):
    #return 2 * (x[0] - 1) ** 2 + 10 * (x[1] + 10) ** 2
    a = x[0] * xscale
    b = x[1] * yscale
    return 2 * (a - 1) ** 2 + 30 * (b + 10) ** 2
def Grad(func, x):
    return np.array(nd.Gradient(func)(x))

    # сам метод


import matplotlib.pyplot as plt


def Gr_m(x1, x2):
    i = 0
    iteration = 1
    alpha = 0.03  # Шаг сходимости
    eps = 0.001  # точность
    max_iteration = 1000 #максимальное количество итераций
    X_prev = np.array([x1, x2])
    X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])

    while np.linalg.norm(X - X_prev) > eps:
        X_prev = X.copy()
        i = i + 1
        print(i, ":", X)
        X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])  # Формула
        points[iteration] = X
        iteration += 1
        if iteration > max_iteration:
            return X
    print("Iteration:", iteration)
    return X


result = Gr_m(st[0], st[1])
filtered_points = points[~np.all(points == 0, axis=1)]
X_estimate, Y_estimate = filtered_points[:, 0], filtered_points[:, 1]

min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
# print(X_estimate)
# print(Y_estimate)
Z_estimate = F(np.array([X_estimate, Y_estimate]))

min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
t = np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(t, t)

#print(filtered_points)
plt.contour(X, Y, F([X, Y]), levels=sorted(set([F(p) for p in filtered_points])))

plt.plot(X_estimate, Y_estimate, '.-', color="r")
plt.show()

print(result)
