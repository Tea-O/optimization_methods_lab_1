import time
from datetime import datetime
import numdifftools as nd
import numpy as np
def pretty_table(data, cell_sep=' | ', header_separator=True) -> str:
    rows = len(data)
    cols = len(data[0])
    col_width = []
    for col in range(cols):
        columns = [str(data[row][col]) for row in range(rows)]
        col_width.append(len(max(columns, key=len)))

    separator = "-+-".join('-' * n for n in col_width)

    lines = []

    for i, row in enumerate(range(rows)):
        result = []
        for col in range(cols):
            item = str(data[row][col]).rjust(col_width[col])
            result.append(item)

        lines.append(cell_sep.join(result))

        if i == 0 and header_separator:
            lines.append(separator)

    return '\n'.join(lines)
start_time = datetime.now()
st = [-5, -20]
eposh = 100
points = np.zeros((eposh, 2))
points[0] = st
accuracy = 0.001
xscale = 1.5
yscale = 0.4
def steepest_descent_with_dichotomy(f, grad_f, initial_x, initial_y, epsilon, max_iterations, alpha_bounds):
    p = [initial_x, initial_y]
    iteration = 1
    numOfFunc = 0
    while iteration < max_iterations:
        gradient = grad_f(f, p)

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
            numOfFunc = numOfFunc + 2
        x_new = p[0] - alpha_star * gradient[0]
        y_new = p[1] - alpha_star * gradient[1]
        if abs(f([x_new, y_new]) - f([p[0], p[1]])) < epsilon:
            break
        p = x_new, y_new
        points[iteration] = p
        iteration += 1
    return p[0], p[1], iteration


def F(x):
    a = x[0] * xscale
    b = x[1] * yscale
    return 2 * (a - 1) ** 2 + 30 * (b + 10) ** 2

def grad(func, x):
    return np.array(nd.Gradient(func)(x))


result = steepest_descent_with_dichotomy(F, grad, st[0], st[1], accuracy, eposh, np.array([0.01, 0.4]))
print(result)
end = datetime.now() - start_time
import matplotlib.pyplot as plt
filtered_points = points[~np.all(points == 0, axis=1)]
X_estimate, Y_estimate = filtered_points[:, 0], filtered_points[:, 1]
Z_estimate = F(np.array([X_estimate, Y_estimate]))
min_x0, min_x1 = np.meshgrid(result[0], result[1])
min_z = F(np.stack([min_x0, min_x1]))
t = np.linspace(-20, 20, 1000)
X, Y = np.meshgrid(t, t)
plt.contour(X, Y, F([X, Y]), levels=sorted([F(p) for p in filtered_points]))
plt.plot(X_estimate, Y_estimate, '.-', color="r")
plt.show()

