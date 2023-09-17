import numpy as np
from datetime import datetime
import numdifftools as nd
start_time = datetime.now()

def F(x):
    return 2 * (x[0] - 1) ** 2 + 100 * (x[1] + 10) ** 2



def grad(func, x):
    return np.array(nd.Gradient(func)(x))

st = [-20, 10]
eposh = 2000
points= np.zeros((eposh, 2))
points[0] = st
# def WolfeConditions(x_k, f, grad, a1 = 1e-4, a2 = 0.9):

def descentWithWolfe(x, f, grad,initial, a1 = 1e-4, a2 = 0.9, eps = 1e-5, maxIter = eposh - 1):
    iter = 0

    while iter <= maxIter:
        gradient = grad(f, x)
        alpha = 1

        while True:
            xk = x - alpha * gradient
            if f(xk) <= f(x) + a1 * alpha * gradient.T @ (-gradient):
                if (grad(f, xk) @ (-gradient)) >= a2 * gradient.T @ (-gradient):
                    break
            alpha /= 2
        if np.linalg.norm(gradient) < eps:
            break
        iter += 1
        x = xk
        if(iter > maxIter):
            return iter
        points[iter] = xk
    return x, iter

result = descentWithWolfe(st, F, grad, [0, 1])
print(result)
import matplotlib.pyplot as plt

filtered_points = points[~np.all(points == 0, axis=1)]
X_estimate, Y_estimate = filtered_points[:, 0], filtered_points[:, 1]
# print(X_estimate)
# print(Y_estimate)
Z_estimate = F(np.array([X_estimate, Y_estimate]))

min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
t = np.linspace(-25, 25, 1000)
X, Y = np.meshgrid(t, t)
plt.contour(X, Y, F([X, Y]), levels=sorted([F(p) for p in filtered_points]))
plt.plot(X_estimate, Y_estimate, '.-', color="r")
plt.show()


end = datetime.now() - start_time
print(end)