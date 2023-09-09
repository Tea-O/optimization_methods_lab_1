import numpy as np
from datetime import datetime
start_time = datetime.now()

def F(x):
    return 7 * (x[0] + 3) ** 2 + 0.5 * (x[1] + 7) ** 2



def grad(x):
    return np.array([14 * x[0] + 42, x[1] + 7])

st = [-20, 10]
eposh = 103
points= np.zeros((eposh, 2))
points[0] = st
# def WolfeConditions(x_k, f, grad, a1 = 1e-4, a2 = 0.9):

def descentWithWolfe(x, f, grad,initial, a1 = 1e-4, a2 = 0.9, eps = 1e-5, maxIter = 1e+5):
    iter = 0

    while iter <= maxIter:
        gradient = grad(x)
        alpha = 1

        while True:
            xk = x - alpha * gradient
            if f(xk) <= f(x) + a1 * alpha * gradient.T @ (-gradient):
                if (grad(xk) @ (-gradient)) >= a2 * gradient.T @ (-gradient):
                    break
            alpha /= 2
        if np.linalg.norm(gradient) < eps:
            break
        iter += 1
        x = xk
        points[iter] = xk
    return x, iter

result = descentWithWolfe(np.array([-20, 10]), F, grad, [0, 1])
print(result)
import matplotlib.pyplot as plt

X_estimate, Y_estimate = points[:, 0], points[:, 1]
# print(X_estimate)
# print(Y_estimate)
Z_estimate = F(np.array([X_estimate, Y_estimate]))

min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
t = np.linspace(-25, 25, 1000)
X, Y = np.meshgrid(t, t)
plt.contour(X, Y, F([X, Y]), levels=sorted([F(p) for p in points]))
plt.plot(X_estimate, Y_estimate, '.-', color="r")
plt.show()


end = datetime.now() - start_time
print(end)