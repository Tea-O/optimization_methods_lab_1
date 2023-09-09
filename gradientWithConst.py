import numdifftools as nd
import numpy as np
eposh = 200
st = np.array([-20, 10])
points= np.zeros((eposh, 2))
points[0] = st
def F(x):
    return 10 * (x[0] - 14) ** 2 + (x[1] - 7) ** 2


def Grad(func, x):
    return np.array(nd.Gradient(func)(x))

    # сам метод


import matplotlib.pyplot as plt

def Gr_m(x1, x2):
    i = 0
    iteration = 1
    alpha = 0.03  # Шаг сходимости
    eps = 0.0001  # точность
    X_prev = np.array([x1, x2])
    X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])

    while np.linalg.norm(X - X_prev) > eps:
        X_prev = X.copy()
        i = i + 1
        print(i, ":", X)
        X = X_prev - alpha * Grad(F, [X_prev[0], X_prev[1]])  # Формула
        points[iteration] = X
        iteration += 1
    return X


result = Gr_m(-20, 10)

X_estimate, Y_estimate = points[:, 0], points[:, 1]


min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))

fig = plt.figure(figsize=(35, 35))
ax = fig.add_subplot(1, 2, 1, projection='3d')
t = np.linspace(-60, 60, 100)
X, Y = np.meshgrid(t, t)
ax.contour3D(X, Y, F(np.array([X, Y])), 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate,  color='red', linewidth=1)
#ax.scatter(X_estimate, Y_estimate, F(np.array([X_estimate, Y_estimate])), marker='o', color='blue', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.view_init(-90, 90)
plt.show()

print(result)
